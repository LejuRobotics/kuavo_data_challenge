#!/usr/bin/env python3
"""
Kuavo机器人控制指令发送器
用于向正在运行的script.py进程发送控制指令

使用示例:
  python controller.py pause    # 暂停机械臂运动
  python controller.py resume   # 恢复机械臂运动
  python controller.py stop     # 停止机械臂运动
  python controller.py status   # 查看进程状态
"""

import os
import sys
import signal
import psutil
import argparse
from pathlib import Path

def find_example_process():
    """
    查找正在运行的script.py进程
    
    Returns:
        psutil.Process: 找到的进程对象，如果没找到返回None
    """
    target_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查进程名或命令行参数
            if (proc.info['name'] == 'python' or proc.info['name'] == 'python3') and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                
                # 精确匹配 kuavo_deploy/src/scripts/script.py 路径
                if 'kuavo_deploy/src/scripts/script.py' in cmdline:
                    target_processes.append((proc, 'exact'))
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if len(target_processes) != 1:
        print(f"❌ 找到 {len(target_processes)} 个匹配的进程，请使用 --pid 参数指定进程ID")
        sys.exit(1)
    else:
        return target_processes[0][0]

def send_signal_to_process(proc, signal_type):
    """
    向指定进程发送信号
    
    Args:
        proc: psutil.Process对象
        signal_type: 信号类型 ('pause', 'resume', 'stop')
    """
    try:
        if signal_type == 'pause':
            proc.send_signal(signal.SIGUSR1)
            print(f"✅ 已发送暂停信号到进程 {proc.pid}")
        elif signal_type == 'resume':
            proc.send_signal(signal.SIGUSR1)
            print(f"✅ 已发送恢复信号到进程 {proc.pid}")
        elif signal_type == 'stop':
            proc.send_signal(signal.SIGUSR2)
            print(f"✅ 已发送停止信号到进程 {proc.pid}")
        else:
            print(f"❌ 未知的信号类型: {signal_type}")
            return False
        return True
    except psutil.NoSuchProcess:
        print(f"❌ 进程 {proc.pid} 不存在")
        return False
    except psutil.AccessDenied:
        print(f"❌ 没有权限向进程 {proc.pid} 发送信号")
        return False
    except Exception as e:
        print(f"❌ 发送信号时发生错误: {e}")
        return False

def show_process_status(proc):
    """
    显示进程状态信息
    
    Args:
        proc: psutil.Process对象
    """
    try:
        print(f"📊 进程信息:")
        print(f"  PID: {proc.pid}")
        print(f"  状态: {proc.status()}")
        print(f"  创建时间: {proc.create_time()}")
        print(f"  CPU使用率: {proc.cpu_percent()}%")
        print(f"  内存使用: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"  命令行: {' '.join(proc.cmdline())}")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"❌ 无法获取进程信息: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Kuavo机器人控制指令发送器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python controller.py pause    # 暂停机械臂运动
  python controller.py resume   # 恢复机械臂运动
  python controller.py stop     # 停止机械臂运动
  python controller.py status   # 查看进程状态

控制指令说明:
  pause   - 暂停机械臂运动 (发送SIGUSR1信号)
  resume  - 恢复机械臂运动 (发送SIGUSR1信号)
  stop    - 停止机械臂运动 (发送SIGUSR2信号)
  status  - 显示当前运行的script.py进程状态
        """
    )
    
    parser.add_argument(
        "command",
        type=str,
        choices=["pause", "resume", "stop", "status"],
        help="控制指令"
    )
    
    parser.add_argument(
        "--pid",
        type=int,
        help="指定进程PID (如果不指定，将自动查找script.py进程)"
    )
    
    args = parser.parse_args()
    
    # 查找目标进程
    target_proc = None
    
    if args.pid:
        # 使用指定的PID
        try:
            target_proc = psutil.Process(args.pid)
            # 验证进程是否运行script.py
            cmdline = ' '.join(target_proc.cmdline())
            if 'script.py' not in cmdline:
                print(f"❌ 进程 {args.pid} 不是script.py进程")
                print(f"   命令行: {cmdline}")
                sys.exit(1)
        except psutil.NoSuchProcess:
            print(f"❌ 进程 {args.pid} 不存在")
            sys.exit(1)
        except psutil.AccessDenied:
            print(f"❌ 没有权限访问进程 {args.pid}")
            sys.exit(1)
    else:
        # 自动查找script.py进程
        print("🔍 正在查找运行中的script.py进程...")
        target_proc = find_example_process()
        
        if not target_proc:
            print("❌ 未找到运行中的script.py进程")
            print("💡 请确保script.py正在运行，或使用 --pid 参数指定进程ID")
            print("💡 预期的进程路径: kuavo_deploy/src/scripts/script.py")
            sys.exit(1)
        
        # 显示找到的进程信息
        cmdline = ' '.join(target_proc.cmdline())
        if 'kuavo_deploy/src/scripts/script.py' in cmdline:
            print(f"✅ 找到精确匹配的进程: {target_proc.pid}")
        else:
            print(f"⚠️  找到部分匹配的进程: {target_proc.pid}")
            print(f"   命令行: {cmdline}")
    
    # 执行命令
    if args.command == "status":
        show_process_status(target_proc)
    else:
        print(f"🎯 目标进程: {target_proc.pid}")
        success = send_signal_to_process(target_proc, args.command)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
