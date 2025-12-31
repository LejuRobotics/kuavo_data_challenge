#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo机器人控制示例脚本 (Python版)
完全等价于原 Bash 脚本，支持交互控制、暂停、恢复、停止、日志查看等功能。
Kuavo Robot Control Demostration Script (for Python)
Equivalent to the old Bash based scripts. Supports interactive controls, pause, resume, stop, view logs, etc.
"""

import os
import sys
import signal
import subprocess
import yaml
from pathlib import Path
from time import sleep
import select, time, threading, queue
# 全局变量 Global Variables
current_proc = None
LOG_DIR = None


# ========== 信号处理 Signal Processing ==========
def cleanup(signum, frame):
    global current_proc
    print("\n⏹️ Ctrl+C detected, stopping task")
    if current_proc and current_proc.poll() is None:
        print(f"⏹️ Now stopping task (PID: {current_proc.pid})...")
        current_proc.terminate()
        try:
            current_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_proc.kill()
        print("✅ Task successfully terminated!")
    sys.exit(130)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


# ========== 工具函数 ==========
def print_header():
    print("=== Kuavo Robot Control Demostration ===")
    print("This script shows how to use command parameters to execute various tasks")
    print("Supports pause, play and stop.\n")
    print("📋 Control Explaination:")
    print("  🔄 Pause/Resume: Send SIGUSR1 signal")
    print("  ⏹️  Stop: Send SIGUSR2 signal")
    print("  📊 View Log: tail -f log/kuavo_deploy/kuavo_deploy.log\n")


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
    """Background thread: Continuously monitors user input"""
    sys.stdout.write(f"\r🟢 The task is running. Enter command to pause/stop/etc. (p/s/l/h): >")
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

    print("🎮 Interactive Control has Started")
    print(f"Task PID: {current_proc.pid}\n")
    print("📋 Available Commands:")
    print("  p/pause    - Pause/Resume")
    print("  s/stop     - Stop")
    print("  l/log      - View Log")
    print("  h/help     - Display Help\n")

    input_queue = queue.Queue()
    stop_event = threading.Event()

    # 启动输入监听线程
    # Starts input monitoring thread
    threading.Thread(
        target=input_listener, args=(input_queue, stop_event), daemon=True
    ).start()

    while True:
        # 检查子进程状态
        # Check subprocess status
        if current_proc.poll() is not None:
            retcode = current_proc.returncode  # 获取退出码 Exit code

            if retcode == 0:
                print("\n✅ The task has successfully completed")
            else:
                print(f"\n❌ Abnormal termination detected! Error code: {retcode}")
                print("📄 Please see the event log: log/kuavo_deploy/kuavo_deploy.log")

            current_proc = None
            stop_event.set()
            break


        try:
            cmd = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # 无输入则继续检测进程 No input detected, moving on...
        # cmd = input(f"🟢 任务运行中 (PID: {current_proc.pid}) > ").strip().lower()  # 会阻塞等待输入
        # if cmd != "":
            # stop_event.set()
        if cmd in ("p", "pause"):
            print("🔄 Sending pause/resume signal...")
            os.kill(current_proc.pid, signal.SIGUSR1)

        elif cmd in ("s", "stop"):
            print("⏹️  Sending stop signal...")
            os.kill(current_proc.pid, signal.SIGUSR2)
            try:
                current_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_proc.kill()
            print("✅ The task has successfully stopped")
            stop_event.set()
            break

        elif cmd in ("l", "log"):
            log_path = LOG_DIR / "kuavo_deploy.log"
            if log_path.exists():
                print("📊 Displaying the latest entries in the log file (Ctrl+C to return):")
                os.system(f"tail -n 20 {log_path}")
            else:
                print("❌ The log file is not found!")

        elif cmd in ("h", "help"):
            print("📋 Available Commands:")
            print("  p/pause    - Pause/Resume")
            print("  s/stop     - Stop")
            print("  l/log      - View Log")
            print("  h/help     - Display Help\n")

        else:
            print(f"❌ Unknown Command: {cmd}")


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
        print("📋 Model Configuration Info:")
        print(f"   Task: {task}")
        print(f"   Method: {method}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Epoch: {epoch}")
        print(f"📂 Full Model Path: {model_path}")
        if model_path.exists():
            print("✅ Model Path Exists")
        else:
            print("❌ Model Path Does not Exist")
        return cfg
    except Exception as e:
        print(f"❌ Failed to Parse Configuration File: {e}")
        sys.exit(1)


def print_task_menu(config_path="<config_path>", use_color=True):
    """
    打印 Kuavo 任务菜单，说明在前，统一在最后输出命令行模板。
    Prints Kuavo task menu, with description up front.
    
    Args:
        config_path (str): 默认配置文件路径，用于命令行模板显示。 Default configuration file path, used for displaying this template
        use_color (bool): 是否使用终端颜色，默认 True。 Whether colour is used in terminal. Defaults to True.
    """
    # 终端颜色定义 Terminal colour definition
    GREEN  = "\033[32m" if use_color else ""
    BLUE   = "\033[34m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    RESET  = "\033[0m"  if use_color else ""

    tasks = [
        ("go (dry_run)", "Normal Task (Dry run): First insert values towards the first frame of the rosbag file, then starts playback of the rosbag file (Dry run, nothing happens)."),
        ("go", "Normal Task: First insert values towards the first frame of the rosbag file, then starts playback of the rosbag file (towards its working position)."),
        ("run", "Normal Task: Starts running the model at the current position."),
        ("go_run", "Normal Task: First gets to its working position, then start running model"),
        ("here_run", "Normal Task: Insert values towards the last frame of the rosbag file, then start running model"),
        ("back_to_zero", "Normal Task: After interrputing the running model, play the rosbag file in reverse to its zero position"),
        ("go (verbose)", "Normal Task: Same as Option 2 but with detailed outputs"),
        ("auto_test", "Simulator auto-test: Auto-testing inside the simulator, with number of iterations specified as eval_episode"),
        ("Exit", ""),
    ]

    print(f"\n🟢 Here are the available task options:")
    for idx, (name, desc) in enumerate(tasks, 1):
        if desc:
            print(f"{GREEN}{idx}. {name:<15}{RESET} : {BLUE}{desc}{RESET}")
        else:
            print(f"{GREEN}{idx}. {name}{RESET}")

    # 统一输出命令行模板 Template
    print(f"📋 After selection, the following command will be executed: {RESET}")
    print(f"Normal task:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script.py --task <chosen_task> --config {config_path}{RESET}")
    print(f"Auto-testing Task:{RESET}")
    print(f"{YELLOW}  python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config {config_path}{RESET}")



# ========== 主逻辑 Main logic ==========
def main():
    global current_proc, LOG_DIR

    print_header()
    script_dir, script, auto_test = get_script_paths()
    LOG_DIR = ensure_log_dir(script_dir)

    if not script.exists():
        print(f"Error: script.py not found: {script}")
        sys.exit(1)
    if not auto_test.exists():
        print(f"Error: script_auto_test.py not found: {auto_test}")
        sys.exit(1)

    print("1. Execute: python script.py --help")
    print("2. Execute: python script_auto_test.py --help")
    print("3. Task Selection Menu\n")

    choice = input("Please select an option (1-3) or press Enter to exit: ").strip()
    if choice == "1":
        subprocess.run(["python3", str(script), "--help"])
        return
    elif choice == "2":
        subprocess.run(["python3", str(auto_test), "--help"])
        return
    elif choice == "":
        print("Exiting")
        return
    elif choice != "3":
        print("Invalid Option")
        return

    config_path = input("Please specify filepath to the configuration file: ").strip()
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    parse_config(config_path)

    while True:
        print_task_menu(config_path=config_path, use_color=True)

        sub_choice = input("Please select one of the following options (1-9): ").strip()

        def start_task(cmd):
            global current_proc
            log_path = LOG_DIR / "kuavo_deploy.log"
            with open(log_path, "w") as f:
                current_proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"Task started, PID: {current_proc.pid}")
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
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice: ", sub_choice)


if __name__ == "__main__":
    main()
