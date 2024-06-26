"""
Scripts to drive a robopilot car remotely

Usage:
    remote_cam_view.py --name=<robot_name> --broker="localhost" [--record=<path>]


Options:
    -h --help     Show this screen.
"""
import os
import time
import math
from docopt import docopt
import robopilot as dk
import cv2

from robopilot.parts.cv import CvImageView, ImgBGR2RGB, ImgRGB2BGR, ImageScale, ImgWriter, ArrowKeyboardControls
from robopilot.parts.salient import SalientVis
from robopilot.parts.network import MQTTValuePub, MQTTValueSub
from robopilot.parts.transform import Lambda
from robopilot.parts.image import JpgToImgArr

V = dk.vehicle.Vehicle()
args = docopt(__doc__)
print(args)

V.add(MQTTValueSub(name="robopilot/%s/camera" % args["--name"], broker=args["--broker"]), outputs=["jpg"])
V.add(JpgToImgArr(), inputs=["jpg"], outputs=["img_arr"]) 
V.add(ImgBGR2RGB(), inputs=["img_arr"], outputs=["rgb"])
V.add(ImageScale(4.0), inputs=["rgb"], outputs=["lg_img"])
V.add(CvImageView(), inputs=["lg_img"])

V.add(ArrowKeyboardControls(), outputs=["control"])
V.add(MQTTValuePub(name="robopilot/%s/controls" % args["--name"]), inputs=["control"])

record_path = args["--record"]
if record_path is not None:
    class ImageSaver:
        def __init__(self, path):
            self.index = 0
            self.path = path
        
        def run(self, img_arr):
            if img_arr is None:
                return
            dest_path = os.path.join(self.path, "img_%d.jpg" % self.index)
            self.index += 1
            cv2.imwrite(dest_path, img_arr)
    
    V.add(ImageSaver(record_path), inputs=["rgb"])


V.start(rate_hz=20)

