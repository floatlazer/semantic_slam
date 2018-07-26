#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from skimage.transform import resize


import fcrn as models

class DepthCnn:
    def __init__(self, model_data_path):

        # Default input size
        self.height = 228
        self.width = 304
        self.channels = 3
        self.batch_size = 1

        with tf.device('/device:GPU:1'):

            # Create a placeholder for the input image
            self.input_node = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels))

            # Construct the network
            self.net = models.ResNet50UpProj({'data': self.input_node}, self.batch_size, 1, False)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # Use another device if current device not exist
        # Load the converted parameters
        print('Loading the model')
        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(self.sess, model_data_path)

        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image

        self.depth_pub = rospy.Publisher("/pred_depth", Image, queue_size = 1)

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image, self.predict, queue_size = 1, buff_size = 30*480*640)


    def predict(self, color_img_ros):

        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "rgb8") # Convert ros msg to numpy array
        except CvBridgeError as e:
            print(e)

        # Prepare image
        img = resize(color_img, (self.height, self.width), mode = 'reflect', anti_aliasing=True, preserve_range = True) # Give float64
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)

        # Evalute the network for the given image
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(pred[0,:,:,0], encoding="32FC1")
        depth_msg.header = color_img_ros.header
        self.depth_pub.publish(depth_msg)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    args = parser.parse_args()
    #model_path = '~/xuan/pre_catkin_ws/src/pre2018/depth_cnn/include/fcrn/NYU_FCRN'
    rospy.init_node('seg_cnn', anonymous=True)
    seg_cnn = DepthCnn(args.model_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
