#!/usr/bin/env python
'''
Send a simple 2d plan point cloud
'''
from __future__ import division, print_function
import rospy
from color_pcl_generator import ColorPclGenerator
import numpy as np
from sensor_msgs.msg import PointCloud2

def single_color_img(width, height, (b, g, r)):
    img = np.ones((height, width,3), dtype = np.uint8)
    img[:,:,0] = np.ones((height, width), dtype = np.uint8) * b
    img[:,:,1] = np.ones((height, width), dtype = np.uint8) * g
    img[:,:,2] = np.ones((height, width), dtype = np.uint8) * r
    return img

class SemanticFusionTest:
    def __init__(self):
        # Camera intrinsic matrix
        fx = 544.771755
        fy = 546.966312
        cx = 322.376103
        cy = 245.357925
        width = 640
        height = 480
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        bgr = WHITE
        confidence1 = 0.7#0.7
        confidence2 = 0.2#0.2
        confidence3 = 0.05#0.05
        self.intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
        self.pcl_pub = rospy.Publisher("semantic_pcl",PointCloud2, queue_size = 1)
        self.cloud_gen = ColorPclGenerator(width, height, frame_id = 'world')
        # Color image
        color_img = single_color_img(width, height, bgr)
        # Depth image
        depth_img = np.ones((height, width), dtype = np.float32) * 1.
        # Semantic colors
        semantic_colors = np.zeros((3, height, width, 3), dtype = np.uint8)
        # Confidences
        confidences = np.ones((3, height, width), dtype = np.float32)
        confidences[0] *= confidence1
        confidences[1] *= confidence2
        confidences[2] *= confidence3
        COLOR_SEQ = []
        COLOR_SEQ.append([RED, GREEN, BLUE]) # Init color
        COLOR_SEQ.append([GREEN, RED, BLUE]) # Update with same color set
        COLOR_SEQ.append([BLUE, RED, GREEN]) # Update with same color set
        COLOR_SEQ.append([WHITE, BLUE, RED]) # Update with a new color
        # Publish
        r = rospy.Rate(2)
        i = 0
        while not rospy.is_shutdown():
            # Change color
            semantic_colors[0] = single_color_img(width, height, COLOR_SEQ[i][0])
            semantic_colors[1] = single_color_img(width, height, COLOR_SEQ[i][1])
            semantic_colors[2] = single_color_img(width, height, COLOR_SEQ[i][2])
            since = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - since).to_sec() < 5:
                # Produce point cloud with rgb colors, semantic colors and confidences
                cloud_ros = self.cloud_gen.generate_cloud(color_img, depth_img, semantic_colors, confidences, self.intrinsic, rospy.Time.now())
                print('Publish point cloud. Color(bgr):', bgr, 'Semantic colors:', COLOR_SEQ[i][0], COLOR_SEQ[i][1], COLOR_SEQ[i][2], 'Confidences:', confidence1, confidence2, confidence3)
                self.pcl_pub.publish(cloud_ros)
                r.sleep()
            i += 1
            if i == len(COLOR_SEQ):
                #i = 0
                break

if __name__ == '__main__':
        rospy.init_node('semantic_fusion_test')
        sft = SemanticFusionTest()
