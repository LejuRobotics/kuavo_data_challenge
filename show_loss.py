import argparse
import time
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import tkinter as tk

def get_screen_size():
    """获取屏幕宽高"""
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def main():
    parser = argparse.ArgumentParser(description="Dynamic TensorBoard Loss Viewer")
    parser.add_argument("logdir", type=str, help="Path to TensorBoard log directory")
    args = parser.parse_args()

    # 载入日志（设置 size_guidance，避免加载太多历史事件）
    ea = event_accumulator.EventAccumulator(
        args.logdir,
        size_guidance={"scalars": 0}  # 0 表示按需加载，不做上限
    )
    ea.Reload()

    # 获取屏幕分辨率
    screen_w, screen_h = get_screen_size()
    screen_w, screen_h = 1440, 1080
    win_w, win_h = 640, 480
    pos_x = (screen_w - win_w) // 2 + 1920 - 1440 + 1920
    pos_y = (screen_h - win_h) // 2

    # 创建窗口并居中
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")  # TkAgg
    except Exception:
        try:
            manager.window.setGeometry(pos_x, pos_y, win_w, win_h)  # QtAgg
        except Exception:
            print("⚠️ 当前 matplotlib 后端不支持自动定位窗口，请手动拖动")

    plt.ion()

    last_len = 0
    while True:
        ea.Reload()
        try:
            events = ea.Scalars("train/loss")
        except KeyError:
            print("找不到 tag: train/loss，请确认 log 文件里的标记名")
            time.sleep(5)
            continue

        # 只取新增数据
        if len(events) > last_len:
            new_events = events[last_len:]
            steps = [e.step for e in events][-2500:]
            values = [e.value for e in events][-2500:]
            last_len = len(events)

            ax.clear()
            ax.plot(steps, values, label="train/loss", color="blue")
            ax.set_xlim(0, 2500)
            ax.set_ylim(0, 1.2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("train")
            ax.legend()
            ax.grid(True)

            plt.draw()

        plt.pause(5)  # 每 5 秒刷新一次，减少 CPU 占用

if __name__ == "__main__":
    main()
