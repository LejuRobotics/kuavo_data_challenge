import rospy
from sensor_msgs.msg import CompressedImage, JointState
from kuavo_deploy.utils.logging_utils import setup_logger
log_robot = setup_logger("robot")
class ROSManager:
    def __init__(self):
        self.subscribers = []
        self.services = []
        self.publishers = []

    def register_subscriber(self, topic, msg_type, callback):
        if msg_type == CompressedImage:
            sub = rospy.Subscriber(topic, msg_type, callback, queue_size=1, tcp_nodelay=True, buff_size=2**20)
        else:
            sub = rospy.Subscriber(topic, msg_type, callback, queue_size=1, tcp_nodelay=True)
        self.subscribers.append(sub)
        return sub

    def register_publisher(self, topic, msg_type, queue_size=10):
        pub = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        self.publishers.append(pub)
        return pub

    def register_service(self, name, srv_type, handler):
        srv = rospy.Service(name, srv_type, handler)
        self.services.append(srv)
        return srv

    def close(self):
        for sub in self.subscribers:
            sub.unregister()
        for srv in self.services:
            srv.shutdown()
        for pub in self.publishers:
            pub.unregister()
        self.subscribers.clear()
        self.services.clear()
        self.publishers.clear()
        log_robot.info("All ROS resources released.")
    
    def __del__(self):
        self.close()