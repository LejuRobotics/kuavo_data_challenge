from kuavo_deploy.utils.logging_utils import setup_logger
from std_msgs.msg import Bool
from kuavo_deploy.utils.ros_manager import ROSManager
import threading, time


log_robot = setup_logger("robot")

class ControlSignalManager:
    """控制信号管理器"""
    def __init__(self):
        self.ros_manager = ROSManager()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        self.ros_manager.register_subscriber('/kuavo/pause_state', Bool, self._pause_callback)
        self.ros_manager.register_subscriber('/kuavo/stop_state', Bool, self._stop_callback)
    
    def _pause_callback(self, msg):
        """暂停回调"""
        if msg.data:
            self.pause_flag.set()
        else:
            self.pause_flag.clear()

    def _stop_callback(self, msg):
        """停止回调"""
        if msg.data:
            self.stop_flag.set()
    
    def check_control_signals(self):
        """检查控制信号"""
        # 检查暂停状态
        while self.pause_flag.is_set():
            log_robot.info("🔄 机械臂运动已暂停")
            time.sleep(0.1)
            if self.stop_flag.is_set():
                log_robot.info("🛑 机械臂运动被停止")
                return False
        
        # 检查是否需要停止
        if self.stop_flag.is_set():
            log_robot.info("🛑 收到停止信号，退出机械臂运动")
            return False
            
        return True  # 正常继续
    
    def close(self):
        """释放资源"""
        self.ros_manager.close()
        self.stop_flag.clear()
        self.pause_flag.clear()