#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo机器人控制示例脚本 (Python版)
完全等价于原 Bash 脚本，支持交互控制、暂停、恢复、停止、日志查看等功能。
"""

import os
import sys
import signal
import subprocess
import yaml
from pathlib import Path
from time import sleep
import select, time, threading, queue
# 全局变量
current_proc = None
LOG_DIR = None


# ========== 信号处理 ==========
def cleanup(signum, frame):
    global current_proc
    print("\n⏹️ 捕获到 Ctrl+C，开始终止任务")
    if current_proc and current_proc.poll() is None:
        print(f"⏹️ 正在终止任务 (PID: {current_proc.pid})...")
        current_proc.terminate()
        try:
            current_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_proc.kill()
        print("✅ 任务已终止")
    sys.exit(130)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


# ========== 工具函数 ==========
def print_header():
    print("=== Kuavo机器人控制示例 ===")
    print("此脚本展示如何使用命令行参数控制不同的任务")
    print("支持暂停、继续、停止功能\n")
    print("📋 控制功能说明:")
    print("  🔄 暂停/恢复: 发送 SIGUSR1 信号")
    print("  ⏹️  停止任务: 发送 SIGUSR2 信号")
    print("  📊 查看日志: tail -f log/kuavo_deploy/kuavo_deploy.log\n")


def get_script_paths():
    script_dir = Path(__file__).resolve().parent
    script = script_dir / "src" / "scripts" / "script.py"
    auto_test = script_dir / "src" / "scripts" / "script_auto_test.py"
    return script_dir, script, auto_test


def ensure_log_dir(script_dir):
    log_root = script_dir.parent / "log" / "kuavo_deploy"
    log_root.mkdir(parents=True, exist_ok=True)
    return log_root


# ========== 交互控制 ==========
def input_listener(input_queue, stop_event):
    """后台线程：持续监听用户输入"""
    sys.stdout.write(f"\r🟢 任务运行中，输入命令以暂停/停止等(p/s/l/h): >")
    sys.stdout.flush()
    while not stop_event.is_set():
        # 显示提示符
        inputok, _, _ = select.select([sys.stdin], [], [], 0.1)
        if inputok:
            line = sys.stdin.readline()
            # print("line: ", line)  # 换行
            input_queue.put(line.strip().lower())

def interactive_controller():
    global current_proc, LOG_DIR

    print("🎮 交互式控制器已启动")
    print(f"任务PID: {current_proc.pid}\n")
    print("📋 可用命令:")
    print("  p/pause    - 暂停/恢复任务")
    print("  s/stop     - 停止任务")
    print("  l/log      - 查看实时日志")
    print("  h/help     - 显示帮助\n")

    input_queue = queue.Queue()
    stop_event = threading.Event()

    # 启动输入监听线程
    threading.Thread(
        target=input_listener, args=(input_queue, stop_event), daemon=True
    ).start()

    while True:
        # 检查子进程状态
        if current_proc.poll() is not None:
            retcode = current_proc.returncode  # 获取退出码

            if retcode == 0:
                print("\n✅ 任务已正常结束")
            else:
                print(f"\n❌ 任务异常退出，错误码：{retcode}")
                print("📄 请查看日志文件：log/kuavo_deploy/kuavo_deploy.log")

            current_proc = None
            stop_event.set()
            break


        try:
            cmd = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # 无输入则继续检测进程
        # cmd = input(f"🟢 任务运行中 (PID: {current_proc.pid}) > ").strip().lower()  # 会阻塞等待输入
        # if cmd != "":
            # stop_event.set()
        if cmd in ("p", "pause"):
            print("🔄 发送暂停/恢复信号...")
            os.kill(current_proc.pid, signal.SIGUSR1)

        elif cmd in ("s", "stop"):
            print("⏹️  发送停止信号...")
            os.kill(current_proc.pid, signal.SIGUSR2)
            try:
                current_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_proc.kill()
            print("✅ 任务已强制停止")
            stop_event.set()
            break

        elif cmd in ("l", "log"):
            log_path = LOG_DIR / "kuavo_deploy.log"
            if log_path.exists():
                print("📊 显示最新日志 (Ctrl+C 返回):")
                os.system(f"tail -n 20 {log_path}")
            else:
                print("❌ 日志文件不存在")

        elif cmd in ("h", "help"):
            print("📋 可用命令:")
            print("  p/pause    - 暂停/恢复任务")
            print("  s/stop     - 停止任务")
            print("  l/log      - 查看实时日志")
            print("  h/help     - 显示帮助\n")

        else:
            print(f"❌ 未知命令: {cmd}")


# ========== YAML 解析 ==========
def parse_config(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        inf = cfg.get("inference", {})
        task = inf.get("task", "N/A")
        method = inf.get("method", "N/A")
        timestamp = inf.get("timestamp", "N/A")
        epoch = inf.get("epoch", "N/A")
        model_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
        print("📋 模型配置信息:")
        print(f"   Task: {task}")
        print(f"   Method: {method}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Epoch: {epoch}")
        print(f"📂 完整模型路径: {model_path}")
        if model_path.exists():
            print("✅ 模型路径存在")
        else:
            print("❌ 模型路径不存在")
        return cfg
    except Exception as e:
        print(f"❌ 解析配置文件失败: {e}")
        sys.exit(1)


def print_task_menu(config_path="<config_path>", use_color=True):
    """
    打印 Kuavo 任务菜单，说明在前，统一在最后输出命令行模板。
    
    Args:
        config_path (str): 默认配置文件路径，用于命令行模板显示。
        use_color (bool): 是否使用终端颜色，默认 True。
    """
    # 终端颜色定义
    GREEN  = "\033[32m" if use_color else ""
    BLUE   = "\033[34m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    RESET  = "\033[0m"  if use_color else ""

    tasks = [
        ("go (dry_run)", "普通任务: 先插值到bag第一帧的位置, 再回放bag包前往工作位置 (什么也不会发生) "),
        ("go", "普通任务: 先插值到bag第一帧的位置, 再回放bag包前往工作位置"),
        ("run", "普通任务: 从当前位置直接运行模型"),
        ("go_run", "普通任务: 到达工作位置后直接运行模型"),
        ("here_run", "普通任务: 插值至bag的最后一帧状态后直接开始运行模型"),
        ("back_to_zero", "普通任务: 中断模型推理后，倒放bag包回到0位"),
        ("go (verbose)", "普通任务: 执行选项2的执行, 并启用详细输出"),
        ("auto_test", "自动测试任务：仿真中自动测试模型，执行 eval_episodes 次"),
        ("退出", ""),
    ]

    print(f"\n🟢 可选择的任务示例如下:")
    for idx, (name, desc) in enumerate(tasks, 1):
        if desc:
            print(f"{GREEN}{idx}. {name:<15}{RESET} : {BLUE}{desc}{RESET}")
        else:
            print(f"{GREEN}{idx}. {name}{RESET}")

    # 统一输出命令行模板
    print(f"📋 数字选择后，会自动执行的命令示例:{RESET}")
    print(f"普通任务:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script.py --task <chosen_task> --config {config_path}{RESET}")
    print(f"自动测试任务:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config {config_path}{RESET}")



# ========== 主逻辑 ==========
def main():
    global current_proc, LOG_DIR

    print_header()
    script_dir, script, auto_test = get_script_paths()
    LOG_DIR = ensure_log_dir(script_dir)

    if not script.exists():
        print(f"错误: 找不到 script.py 文件: {script}")
        sys.exit(1)
    if not auto_test.exists():
        print(f"错误: 找不到 script_auto_test.py 文件: {auto_test}")
        sys.exit(1)

    print("1. 执行: python script.py --help")
    print("2. 执行: python script_auto_test.py --help")
    print("3. 进一步选择示例\n")

    choice = input("请选择要执行的示例 (1-3) 或按 Enter 退出: ").strip()
    if choice == "1":
        subprocess.run(["python3", str(script), "--help"])
        return
    elif choice == "2":
        subprocess.run(["python3", str(auto_test), "--help"])
        return
    elif choice == "":
        print("退出")
        return
    elif choice != "3":
        print("无效选择")
        return

    config_path = input("请输入自定义配置文件路径: ").strip()
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    parse_config(config_path)

    while True:
        print_task_menu(config_path=config_path, use_color=True)

        sub_choice = input("请选择要执行的示例 (1-9): ").strip()

        def start_task(cmd):
            global current_proc
            log_path = LOG_DIR / "kuavo_deploy.log"
            with open(log_path, "w") as f:
                current_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"任务已启动，PID: {current_proc.pid}")
            interactive_controller()

        if sub_choice == "1":
            subprocess.run(["python3", str(script), "--task", "go", "--dry_run", "--config", config_path])
        elif sub_choice == "2":
            start_task(["python3", str(script), "--task", "go", "--config", config_path])
        elif sub_choice == "3":
            start_task(["python3", str(script), "--task", "run", "--config", config_path])
        elif sub_choice == "4":
            start_task(["python3", str(script), "--task", "go_run", "--config", config_path])
        elif sub_choice == "5":
            start_task(["python3", str(script), "--task", "here_run", "--config", config_path])
        elif sub_choice == "6":
            start_task(["python3", str(script), "--task", "back_to_zero", "--config", config_path])
        elif sub_choice == "7":
            start_task(["python3", str(script), "--task", "go", "--verbose", "--config", config_path])
        elif sub_choice == "8":
            start_task(["python3", str(auto_test), "--task", "auto_test", "--config", config_path])
        elif sub_choice == "9":
            print("退出")
            break
        else:
            print("❌ 无效选择: ", sub_choice)


if __name__ == "__main__":
    main()
