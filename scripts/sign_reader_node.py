#!/usr/bin/env python3

import os
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from tensorflow.keras.models import load_model
from ocr_utils import read_sign

class SignReaderNode:
    def __init__(self):
        rospy.init_node("sign_reader_node")

        self.bridge = CvBridge()

        pkg_path = rospkg.RosPack().get_path("your_ros_pkg")
        model_path = os.path.join(pkg_path, "models", "my_ocr_model.keras")
        self.model = load_model(model_path)

        self.image_sub = rospy.Subscriber(
            "/B1/pi_camera/image_raw",
            Image,
            self.image_cb,
            queue_size=1
        )

        self.score_pub = rospy.Publisher(
            "/score_tracker",
            String,
            queue_size=10
        )

        rospy.loginfo("sign_reader_node ready")

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"cv_bridge failed: {e}")
            return

        type_text, clue_text = read_sign(img, self.model)

        if len(type_text) == 0 or len(clue_text) == 0:
            return

        out = f"{type_text},{clue_text}"
        rospy.loginfo(f"read sign: {out}")
        self.score_pub.publish(out)

if __name__ == "__main__":
    SignReaderNode()
    rospy.spin()