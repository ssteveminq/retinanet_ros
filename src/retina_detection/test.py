#!/usr/bin/env python3
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
# sys.path.append('/usr/lib/python2.7/dist-packages')
import numpy as np
# print("numpy_version", np.version)
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

from model.detector import *
TARGET_SIZE=512


class ObjectDectors(object):
    def __init__(self, wait=0.0):
        # image_topic = "/camera/rgb/image_raw"
        # self.model_ = Detector(timestamp="2021-04-22T11.25.25")
        self.model_ = Detector(timestamp="2021-07-01T22.24.44")
        self.model_.eval()
        self.img_pub =rospy.Publisher("detected_image", Image, queue_size=10)
        self.ret_pub =rospy.Publisher("retina_ros/bounding_boxes", BoundingBoxes, queue_size=10)
        self.bridge = CvBridge()
        print("model-created")
        # image_topic = "/hsrb/head_rgbd_sensor/rgb/image_raw"
        image_topic = "camera/color/image_raw"
        rospy.Subscriber(image_topic, Image, self.image_callback)
        self.savefigure=False
        self.listener()
        # self.bridge = CvBridge()



    def normalize(self, 
        img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,):


        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator

        return img

    def listener(self,wait=0.0):
        rospy.spin() 

    def image_callback(self,msg):
        # print("Received an image!")
        try:
            # Convert your ROS Image message to numpy image data type
            cv2_img= np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            cv2_img_ori= cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            image_before_resize= cv2_img_ori
            # image_before= cv2.cvtColor(image_before, cv2.COLOR_RGB2BGR)
            cv2_img_ori= cv2.resize(cv2_img_ori, (512, 512))
            image_ori = cv2.resize(cv2_img, (512, 512))
            o_height, o_width=  cv2_img.shape[:2]
            # print("o_width", o_width)
            # print("o_height", o_height)
            x_scale = o_width/TARGET_SIZE
            y_scale = o_height/TARGET_SIZE


            #remove alpha channel if it exists
            if image_ori.shape[-1]==4:
                image_ori= image_ori[...,:3]
            #normalize the image
            image = self.normalize(image_ori)

            with torch.no_grad():
                image = torch.Tensor(image)
                if torch.cuda.is_available():
                    image = image.cuda()
                boxes = self.model_.get_boxes(image.permute(2, 0, 1).unsqueeze(0))

            Boxes_msg=BoundingBoxes()
            Boxes_msg.image_header = msg.header
            Boxes_msg.header.stamp = rospy.Time.now()
            for box in boxes[0]:
                # print(box.confidence)
                confidence = float(box.confidence)
                box = (box.box * torch.Tensor([512] * 4)).int().tolist()
                box[0]=int(np.around(box[0]*x_scale))
                box[2]=int(np.around(box[2]*x_scale))
                box[1]=int(np.around(box[1]*y_scale))
                box[3]=int(np.around(box[3]*y_scale))
     
                if confidence>0.1:
                    cv2.rectangle(image_before_resize, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                    cv2.putText(image_before_resize, str(confidence)[:4], (box[0]-2, box[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    # msg_frame = self.bridge.cv2_to_imgmsg(image_ori)
                    detection_box = BoundingBox()
                    # detection_box.Class=str(box.class_id)

                    detection_box.xmin = box[0]
                    detection_box.ymin = box[1]
                    detection_box.xmax = box[2]
                    detection_box.ymax = box[3]
                    detection_box.probability = confidence
                    Boxes_msg.bounding_boxes.append(detection_box)

            # msg_frame = self.bridge.cv2_to_imgmsg(image_ori)
            msg_frame = self.bridge.cv2_to_imgmsg(image_before_resize)
            self.img_pub.publish(msg_frame )
            self.ret_pub.publish(Boxes_msg)


        except CvBridgeError:
            print("error")
        if self.savefigure:
            cv2.imwrite('new_image.jpeg', cv2_img)
            print('save picture')
            # self.detected_msg.data="Take a photo"
            self.savefigure=False


    def test(self):

        save_path = "/home/mk/workspaces/py38_ws/src/retina_detection/data/result/"
        testdata_path = "/home/mk/workspaces/py38_ws/src/retina_detection/data/"
        num_data = 8
        for i in range(num_data):
            filename= testdata_path+str(i)+".jpg"
            print("test filename", filename)
            image = cv2.imread(filename)
            image_ori = cv2.resize(image, (512, 512))
            image = normalize(image_ori)


            model = Detector(timestamp="2021-04-22T11.25.25")
            # model = detector.Detector(timestamp="2021-04-22T11.25.25")
            model.eval()

            with torch.no_grad():
                image = torch.Tensor(image)
                if torch.cuda.is_available():
                    image = image.cuda()
                boxes = model.get_boxes(image.permute(2, 0, 1).unsqueeze(0))

            for box in boxes[0]:
                # print(box)
                # print(box.confidence)
                confidence = float(box.confidence)
                box = (box.box * torch.Tensor([512] * 4)).int().tolist()
                if confidence>0.35:
                    print(box)
                    cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            savefilename=save_path+str(i)+".jpg"
            cv2.imwrite(savefilename, image_ori)

if __name__ == '__main__':
    rospy.init_node('object_recognition_test')
    detection_manager = ObjectDectors()
    # print("manager-created")
    # detection_manager.listener()	
