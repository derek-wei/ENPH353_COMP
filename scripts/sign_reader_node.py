#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from tensorflow.keras.models import load_model

from ocr_utils import read_sign, type_to_location

TEAM_ID = "piggy"
TEAM_PASSWORD = "pissy"


class SignReaderNode:
    def __init__(self):
        rospy.init_node("sign_reader_node")

        self.bridge = CvBridge()
        self.started = False
        self.stopped = False
        self.published_locations = set()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(os.path.dirname(script_dir), "models", "my_ocr_model.keras")
        model_path = rospy.get_param("~model_path", default_model_path)

        if not os.path.isfile(model_path):
            rospy.logfatal(f"Model not found: {model_path}")
            raise FileNotFoundError(model_path)

        self.model = load_model(model_path)

        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)

        self.image_sub = rospy.Subscriber(
            "/B1/pi_camera/image_raw",
            Image,
            self.image_cb,
            queue_size=1
        )

        rospy.loginfo(f"sign_reader_node ready, model={model_path}")

    def publish_score(self, location, clue, repeats=1):
        payload = f"{TEAM_ID},{TEAM_PASSWORD},{location},{clue}"
        for _ in range(repeats):
            self.score_pub.publish(payload)
            rospy.sleep(0.05)
        rospy.loginfo(f"published: {payload}")

    def start_once(self):
        if not self.started:
            self.publish_score(0, "NA", repeats=3)
            self.started = True

    def stop_once(self):
        if not self.stopped:
            self.publish_score(-1, "NA", repeats=3)
            self.stopped = True

    def image_cb(self, msg):
        if self.stopped:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"cv_bridge failed: {e}")
            return

        self.start_once()

        type_text, clue_text = read_sign(img, self.model)
        if not type_text or not clue_text:
            return

        location = type_to_location(type_text)
        if location is None:
            return

        if location in self.published_locations:
            return

        self.publish_score(location, clue_text, repeats=2)
        self.published_locations.add(location)

        if len(self.published_locations) == 8:
            self.stop_once()


if __name__ == "__main__":
    SignReaderNode()
    rospy.spin()