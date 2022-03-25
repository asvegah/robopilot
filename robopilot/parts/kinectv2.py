"""
Author: Ahmad Vegah
File: kinectv2.py
Date: July 12 2021
Notes: Robopilot part for the Microsoft Kinect v2 depth camera.
"""
import os
#export DISPLAY=:0
#export LD_LIBRARY_PATH=/home/jetson-1/freenect2/lib
#export LIBFREENECT2_INSTALL_PREFIX=/home/jetson-1/freenect2
#os.environ["export LIBFREENECT2_INSTALL_PREFIX"] = "/home/jetson-1/freenect2"
#os.environ["export LD_LIBRARY_PATH"] = "/home/jetson-1/freenect2/lib"
#os.environ["export DISPLAY"] = ":0"

import time
import logging

import numpy as np
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

""" try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except: """
try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()
except:
    from pylibfreenect2 import CpuPacketPipeline
    pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)
fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)
print('cameras', num_devices)

#
# NOTE: Jetson Nano users should clone the Jetson Hacks project
#       https://github.com/JetsonHacksNano/installLibfreenect2
#       and install_ https://github.com/r9y9/pylibfreenect2 from source_ in order to get the python bindings.
#

#
# The Kinect v2 will not output images with arbitrarily chosen dimensions.
# The dimensions must be from the allowed list.  I've chosen a reasonable
# resolution that I know is supported by the camera.
# If the part is initialized with different sizes, then opencv
# will be used to resize the image before it is returned.
#
WIDTH = 424
HEIGHT = 240
CHANNELS = 3

class MSKinectV2(object):
    """
    Robopilot part for the Microsoft Kinect v2 depth camera.
    The Kinect v2 camera is a device which uses a wide angle camera,
    an IR stream, a depth map along with an rgb image
    NOTE: this ALWAYS captures 424 pixels wide x 240 pixels high x RGB at 60fps.
          If an image width, height or depth are passed with different values,
          the the outputs will be scaled to the requested size on the way out.
    """

    def __init__(self, width = WIDTH, height = HEIGHT, channels = CHANNELS, enable_rgb=True, enable_depth=True, device_id = None):
        self.device_id = fn.getDeviceSerialNumber(0)  # "923322071108" # serial number of device to use or None to use default
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        self.width = width
        self.height = height
        self.channels = channels
        self.resize = (width != WIDTH) or (height != height) or (channels != CHANNELS)
        if self.resize:
            print("The output images will be resized from {} to {}.  This requires opencv.".format((WIDTH, HEIGHT, CHANNELS), (self.width, self.height, self.channels)))

        # Configure streams
        self.pipeline = None
        self.config = fn.openDevice(self.device_id, pipeline=pipeline) 
        
        listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)
        
        # Register listeners
        self.config.setColorFrameListener(listener)
        self.config.setIrAndDepthFrameListener(listener)

        if self.enable_depth or self.enable_rgb:             
            
            # Start streaming
            if self.enable_rgb and self.enable_depth:
                self.config.start()
            else:
                config.startStreams(rgb=self.enable_rgb, depth=self.enable_depth)

            # NOTE: must be called after device.start()
            if self.enable_depth:
                self.registration = Registration(self.config.getIrCameraParams(),
                                            self.config.getColorCameraParams())

            self.undistorted = Frame(512, 424, 4)
            self.registered = Frame(512, 424, 4)
            self.color_depth_map = np.zeros((424, 512),  np.int32).ravel()                                 

            # eat some frames to allow auto-exposure to settle
            #for i in range(0, 5):
                #self.pipeline.wait_for_frames()
                #listener.waitForNewFrame()
                #listener.release(frames)  
            
            self.pipeline = listener 
            print('Kinect Camera loaded.. .warming camera')
            time.sleep(2)   # let camera warm up

              

        # initialize frame state
        self.color_image = None
        self.depth_image = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_time = self.start_time

        self.running = True

    def _stop_pipeline(self):
        if self.pipeline is not None:
            self.config.stop()
            self.config.close()  
            self.pipeline = None

    def _poll(self):
        import cv2

        last_time = self.frame_time
        self.frame_time = time.time() - self.start_time
        self.frame_count += 1

        #
        # get the frames
        #
        try:
            if self.enable_rgb or self.enable_depth:
                frames = self.pipeline.waitForNewFrame()        
        except Exception as e:
            logging.error(e)
            return

        #
        # convert camera frames to images
        #
        if self.enable_rgb or self.enable_depth:
            # Align the depth frame to color frame
            depth_frame = frames["depth"]
            color_frame = frames["color"]
            self.registration.apply(color_frame, depth_frame, self.undistorted, self.registered, color_depth_map=self.color_depth_map)
            # Convert depth to 8bit array, RGB into 8bit planar array
            self.depth_image = depth_frame.asarray(np.uint8)
            flipped_frame = cv2.flip(color_frame.asarray(np.uint8), 1)
            self.color_image = cv2.cvtColor(flipped_frame, cv2.COLOR_RGBA2RGB)
            #self.color_image = self.color_depth_map.reshape(424, 512)
            #self.depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16) if self.enable_depth else None
            #self.color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8) if self.enable_rgb else None

            if self.resize:
                import cv2
                if self.width != WIDTH or self.height != HEIGHT:
                    self.color_image = cv2.resize(self.color_image, (self.width, self.height), cv2.INTER_NEAREST) if self.enable_rgb else None
                    self.depth_image = cv2.resize(self.depth_image, (self.width, self.height), cv2.INTER_NEAREST) if self.enable_depth else None
                if self.channels != CHANNELS:
                    self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2GRAY) if self.enable_rgb else None

        self.pipeline.release(frames) 

    def update(self):
        """
        When running threaded, update() is called from the background thread
        to update the state.  run_threaded() is called to return the latest state.
        """
        while self.running:
            self._poll()

    def run_threaded(self):
        """
        Return the lastest state read by update().  This will not block.
        All 4 states are returned, but may be None if the feature is not enabled when the camera part is constructed.
        For gyroscope, x is pitch, y is yaw and z is roll.
        :return: (rbg_image: nparray, depth_image: nparray, acceleration: (x:float, y:float, z:float), gyroscope: (x:float, y:float, z:float))
        """
        return self.color_image, self.depth_image

    def run(self):
        """
        Read and return frame from camera.  This will block while reading the frame.
        see run_threaded() for return types.
        """
        self._poll()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(2) # give thread enough time to shutdown

        # done running
        self._stop_pipeline()