"""
Author: Ahmad Vegah
File: kinectv2.py
Date: July 12 2021
Notes: Robopilot part for the Microsoft Kinect v2 depth camera.
"""

#export DISPLAY=:0
#export LD_LIBRARY_PATH=/home/jetson-1/freenect2/lib
#export LIBFREENECT2_INSTALL_PREFIX=/home/jetson-1/freenect2
#os.environ["export LIBFREENECT2_INSTALL_PREFIX"] = "/home/jetson-1/freenect2"
#os.environ["export LD_LIBRARY_PATH"] = "/home/jetson-1/freenect2/lib"
#os.environ["export DISPLAY"] = ":0"

import sys
import os
import json
import math

import cv2
import numpy as np

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

COLOR = 0
DEPTH = 1
IR = 2
RAW_COLOR = 3
RAW_DEPTH = 4

class dotdict(dict):
    """dot.notation access to dictionary attributes.
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DEPTH_SHAPE = (int(512), int(424), int(4))
COLOR_SHAPE = (int(1920), int(1080))

class Kinect():
    def __init__(self, params=None, device_index=0, headless=False, pipeline=None):
        '''
            Kinect:  Kinect interface using the libfreenect library. Will query
                libfreenect for available devices, if none are found will throw
                a RuntimeError. Otherwise will open the device for collecting
                color, depth, and ir data. 

                Use kinect.get_frame([<frame type>]) within a typical opencv 
                capture loop to collect data from the kinect.

                When instantiating, optionally pass a parameters file or dict 
                with the kinect's position and orientation in the worldspace 
                and/or the kinect's intrinsic parameters. An example dictionary 
                is shown below:

                    kinect_params = {
                        "position": {
                            "x": 0,
                            "y": 0,
                            "z": 0.810,
                            "roll": 0,
                            "azimuth": 0,
                            "elevation": -34
                        }
                        "intrinsic_parameters": {
                            "cx": 256.684,
                            "cy": 207.085,
                            "fx": 366.193,
                            "fy": 366.193
                        }
                    }
                
                These can also be adjusted using the intrinsic_parameters and 
                position properties. 

            ARGUMENTS:
                params: str or dict
                    Path to a kinect parameters file (.json) or a python dict
                    with position or intrinsic_parameters.
                device_index: int
                    Use to interface with a specific device if more than one is
                    connected. 
                headless: bool
                    If true, will allow the kinect to collect and process data
                    without a display. Usefull if the kinect is plugged into a 
                    server. 
                pipeline: PacketPipeline
                    Optionally pass a pylibfreenect2 pipeline that will be used
                    with the kinect. Possible types are as follows:
                        OpenGLPacketPipeline
                        CpuPacketPipeline
                        OpenCLPacketPipeline
                        CudaPacketPipeline
                        OpenCLKdePacketPipeline
                    Note that this will override the headless argument - 
                    Headless requires CUDA or OpenCL pipeline. 
        '''

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if (num_devices == 0):
            raise RuntimeError('No device connected!')
        if (device_index >= num_devices):
            raise RuntimeError('Device {} not available!'.format(device_index))
        self._device_index = device_index

        # Import pipeline depending on headless state
        self._headless = headless
        if (pipeline is None):
            pipeline = self._import_pipeline(self._headless)
        print("Packet pipeline:", type(pipeline).__name__)
        self.device = self.fn.openDevice(self.fn.getDeviceSerialNumber(self._device_index), 
            pipeline=pipeline)

        # We want all the types
        types = (FrameType.Color | FrameType.Depth | FrameType.Ir)
        self.listener = SyncMultiFrameListener(types)

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        # We'll use this to generate registered depth images
        self.registration = Registration(self.device.getIrCameraParams(),
                                         self.device.getColorCameraParams())
   
        self._camera_position = dotdict({
            "x": 0.,
            "y": 0.,
            "z": 0.,
            "roll": 0.,
            "azimuth": 0.,
            "elevation": 0.
        })
        self._camera_params = dotdict({
            "cx":self.device.getIrCameraParams().cx,
            "cy":self.device.getIrCameraParams().cy,
            "fx":self.device.getIrCameraParams().fx,
            "fy":self.device.getIrCameraParams().fy
        })
        
        # Try and load parameters
        pos, param = self._load_camera_params(params)
        if (pos is not None):
            self._camera_position.update(pos)
        if (pos is not None):
            self._camera_params.update(param)

    def __del__(self):
        self.device.stop()

    def __repr__(self):
        params = {"position": self._camera_position, 
                  "intrinsic_parameters": self._camera_params}
        # Can't really return the pipeline
        return 'Kinect(' + str(params) + ',' + str(self._device_index) + \
                ',' + str(self._headless) + ')'

    @property
    def position(self):
        ''' 
            Return the kinect's stored position. Elements can be referenced 
                using setattr or setitem dunder methods. Can also be used to 
                update the stored dictionary:
                
                    k = ktb.Kinect()
                    k.position.x 
                    >>> 2.0
                    k.position.x = 1.0
                    k.position.x 
                    >>> 1.0
        '''
        return self._camera_position
    
    @property
    def intrinsic_parameters(self):
        ''' 
            Return the kinect's stored position. Elements can be referenced 
                using setattr or setitem dunder methods. Can also be used to 
                update the stored dictionary:
                
                    k = ktb.Kinect()
                    k.intrinsic_parameters.cx 
                    >>> 2.0
                    k.intrinsic_parameters.cx = 1.0
                    k.intrinsic_parameters.cx 
                    >>> 1.0
        '''
        return self._camera_params
        
    @staticmethod
    def _import_pipeline(headless=False):
        '''
            _import_pipeline: Imports the pylibfreenect2 pipeline based on
                whether or not headless mode is enabled. Unfortunately 
                more intelligent importing cannot be implemented (contrary
                to the example scripts) because the import raises a C
                exception, not a python one. As a result the only pipelines
                this function will load are OpenGL or OpenCL.
            ARGUMENTS:
                headless: bool
                    whether or not to run kinect in headless mode.
        '''
        if headless:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        else:
            from pylibfreenect2 import OpenGLPacketPipeline
            pipeline = OpenGLPacketPipeline()
        return pipeline

    @staticmethod
    def _load_camera_params(params_file=None):
        '''
            _load_camera_params: Optionally give the kinect's position and 
                orientation in the worldspace and/or the kinect's intrinsic 
                parameters by passing a json file. An example dictionary is 
                shown below:

                    kinect_params = {
                        "position": {
                            "x": 0,
                            "y": 0,
                            "z": 0.810,
                            "roll": 0,
                            "azimuth": 0,
                            "elevation": -34
                        }
                        "intrinsic_parameters": {
                            "cx": 256.684,
                            "cy": 207.085,
                            "fx": 366.193,
                            "fy": 366.193
                        }
                    }

                Specifically, the dict may contain the "position" or 
                "intrinsic_parameters" keys, with the above fields.
            ARGUMENTS:
                params_file: str or dict
                    Path to a kinect parameters file (.json) or a python dict
                    with position or intrinsic_parameters.
        '''
        if (params_file is None):
            return None, None
        elif isinstance(params_file, str):
            try:
                with open(params_file, 'r') as infile:
                    params_dict = json.load(infile)
            except FileNotFoundError:
                print('Kienct configuration file {} not found'.format(params_file))
                raise
        else:
            params_dict = params_file

        # If the keys do not exist, return None                                    
        return params_dict.get('position', None), params_dict.get('intrinsic_parameters', None)

    def get_frame(self, frame_type=COLOR):
        '''
            get_frame: Returns singleton or list of frames corresponding to 
                input. Frames can be any of the following types:
                    COLOR     - returns a 512 x 424 color image, 
                        registered to depth image
                    DEPTH     - returns a 512 x 424 undistorted depth 
                        image (units are m)
                    IR        - returns a 512 x 424 ir image
                    IR        - returns a 512 x 424 ir image
                    RAW_COLOR - returns a 1920 x 1080 color image
                    RAW_DEPTH - returns a 512 x 424 depth image 
                        (units are mm)
            ARGUMENTS:
                frame_type: [frame type] or frame type
                    A list of frame types. Output corresponds directly to this 
                    list. The default argument will return a single registered 
                    color image. 
        '''
        
        def get_single_frame(ft):
            if (ft == COLOR):
                f, _ = self._get_registered_frame(frames)
                return f
            elif (ft == DEPTH):
                return self._get_depth_frame(frames)
            elif (ft == RAW_COLOR):
                return self._get_raw_color_frame(frames)
            elif (ft == RAW_DEPTH):
                return self._get_raw_depth_frame(frames)
            elif (ft == IR):
                return self._get_ir_frame(frames)
            else:
                raise RuntimeError('Unknown frame type {}'.format(ft))

        frames = self.listener.waitForNewFrame()
        # Multiple types were passed
        if isinstance(frame_type, int):
            return_frames = get_single_frame(frame_type)
        else:
            return_frames = [None] * len(frame_type)
            for i, ft in enumerate(frame_type):
                return_frames[i] = get_single_frame(ft)
        self.listener.release(frames)

        return return_frames

    def get_ptcld(self, roi=None, scale=1000, colorized=False):
        '''
            get_ptcld: Returns a point cloud, generated from depth image. Units 
                are mm by default.
            ARGUMENTS:
                roi: [x, y, w, h]
                    If specified, will crop the point cloud according to the 
                    input roi. Does not accelerate runtime. 
                scale: int
                    Scales the point cloud such that ptcl = ptcl (m) / scale. 
                    ie scale = 1000 returns point cloud in mm. 
                colorized: bool
                    If True, returns color matrix along with point cloud such 
                    that if pt = ptcld[x,y,:], the color of that point is color 
                    = color[x,y,:]
        '''
        # Get undistorted (and registered) frames
        frames = self.listener.waitForNewFrame()
        if (colorized):
            registered, undistorted = self._get_registered_frame(frames)
        else:
            undistorted = self._get_depth_frame(frames)
        self.listener.release(frames)

        # Get point cloud 
        ptcld = self._depthMatrixToPointCloudPos(undistorted, self._camera_params, scale=scale)

        # Adjust point cloud based on real world coordinates
        if (self._camera_position is not None):
            ptcld = self._applyCameraMatrixOrientation(ptcld, self._camera_position)

        # Reshape to correct size
        ptcld = ptcld.reshape(DEPTH_SHAPE[1], DEPTH_SHAPE[0], 3)
        
        # If roi, extract
        if (roi is not None):
            if (isinstance(roi, tuple) or isinstance(roi, list)):
                [y, x, h, w] = roi
                xmin, xmax = int(x), int(x+w)
                ymin, ymax = int(y), int(y+h)
                ptcld = ptcld[xmin:xmax, ymin:ymax, :]
                if (colorized):
                    registered = registered[xmin:xmax, ymin:ymax, :]
            else:
                roi = np.clip(roi, 0, 1)
                for c in range(3):
                    ptcld[:,:,c] = np.multiply(ptcld[:,:,c], roi) 

        if (colorized):
            # Format the color registration map - To become the "color" input for the scatterplot's setData() function.
            colors = np.divide(registered, 255) # values must be between 0.0 - 1.0
            colors = colors.reshape(colors.shape[0] * colors.shape[1], 4) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
            colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
            colors = colors[...,::-1] #BGR to RGB
            return ptcld, colors
        
        return ptcld 

    @staticmethod
    def _depthMatrixToPointCloudPos(z, camera_params, scale=1):
        '''
            Credit to: Logic1
            https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed
        '''
        # bacically this is a vectorized version of depthToPointCloudPos()
        # calculate the real-world xyz vertex coordinates from the raw depth data matrix.
        C, R = np.indices(z.shape)

        R = np.subtract(R, camera_params['cx'])
        R = np.multiply(R, z)
        R = np.divide(R, camera_params['fx'] * scale)

        C = np.subtract(C, camera_params['cy'])
        C = np.multiply(C, z)
        C = np.divide(C, camera_params['fy'] * scale)

        return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))

    @staticmethod
    def _applyCameraMatrixOrientation(pt, camera_position=None):
        '''
            Credit to: Logic1
            https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed
        '''
        # Kinect Sensor Orientation Compensation
        # bacically this is a vectorized version of applyCameraOrientation()
        # uses same trig to rotate a vertex around a gimbal.
        def rotatePoints(ax1, ax2, deg):
            # math to rotate vertexes around a center point on a plane.
            hyp = np.sqrt(pt[:, ax1] ** 2 + pt[:, ax2] ** 2) # Get the length of the hypotenuse of the real-world coordinate from center of rotation, this is the radius!
            d_tan = np.arctan2(pt[:, ax2], pt[:, ax1]) # Calculate the vertexes current angle (returns radians that go from -180 to 180)

            cur_angle = np.degrees(d_tan) % 360 # Convert radians to degrees and use modulo to adjust range from 0 to 360.
            new_angle = np.radians((cur_angle + deg) % 360) # The new angle (in radians) of the vertexes after being rotated by the value of deg.

            pt[:, ax1] = hyp * np.cos(new_angle) # Calculate the rotated coordinate for this axis.
            pt[:, ax2] = hyp * np.sin(new_angle) # Calculate the rotated coordinate for this axis.

        if (camera_position is not None):
            rotatePoints(1, 2, camera_position['roll']) #rotate on the Y&Z plane # Usually disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
            rotatePoints(0, 2, camera_position['elevation']) #rotate on the X&Z plane
            rotatePoints(0, 1, camera_position['azimuth']) #rotate on the X&Y

            # Apply offsets for height and linear position of the sensor (from viewport's center)
            pt[:] += np.float_([camera_position['x'], camera_position['y'], camera_position['z']])

        return pt

    def record(self, filename, frame_type=COLOR, video_codec='XVID'):
        '''
            record: Records a video of the type specified. If no filename is 
                given, records as a .avi. Do not call this in conjunction with
                a typical cv2 display loop. 
            ARGUMENTS:
                filename: (str)
                    Name to save the video with. 
                frame_type: frame type
                    What channel to record (only one type).
                video_codec: (str)
                    cv2 video codec.
        '''

        # Create the video writer
        if (frame_type == RAW_COLOR):
            shape = COLOR_SHAPE
        else:
            shape = DEPTH_SHAPE[:2]
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        out = cv2.VideoWriter(filename, fourcc, 25, shape)

        # Record. On keyboard interrupt close and save.
        try:
            while True:
                frame = self.get_frame(frame_type=frame_type)
                if (frame_type == RAW_COLOR):
                    frame = frame[:,:,:3]                
                out.write(frame)
                if (not self._headless):
                    cv2.imshow("KinectVideo", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            out.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
        finally:
            out.release()
            cv2.destroyAllWindows()

    @staticmethod   
    def _new_frame(shape):
        '''
            _new_frame: Return a pylibfreenect2 frame with the dimensions
                specified. Note that Frames are pointers, and as such returning 
                using numpy images created from them directly results in some
                strange behavior. After using the frame, it is highly 
                recommended to copy the resulting image using np.copy() rather 
                than returning the Frame referencing array directly. 
        '''
        return Frame(shape[0], shape[1], shape[2])

    @staticmethod
    def _get_raw_color_frame(frames):
        ''' _get_raw_color_frame: Return the current rgb image as a cv2 image. '''
        return cv2.resize(frames["color"].asarray(), COLOR_SHAPE)

    @staticmethod
    def _get_raw_depth_frame(frames):
        ''' _get_raw_depth_frame: Return the current depth image as a cv2 image. '''
        return cv2.resize(frames["depth"].asarray(), DEPTH_SHAPE[:2])

    @staticmethod
    def _get_ir_frame(frames):
        ''' get_ir_depth_frame: Return the current ir image as a cv2 image. '''
        return cv2.resize(frames["ir"].asarray(), DEPTH_SHAPE[:2])
    
    def _get_depth_frame(self, frames):
        ''' _get_depth_frame: Return the current undistorted depth image. '''
        undistorted = self._new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        undistorted = np.copy(undistorted.asarray(np.float32))
        return undistorted

    def _get_registered_frame(self, frames):
        ''' get_registered: Return registered color and undistorted depth image. '''
        registered = self._new_frame(DEPTH_SHAPE)
        undistorted = self._new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        self.registration.apply(frames["color"], frames["depth"], undistorted, registered)
        undistorted = np.copy(undistorted.asarray(np.float32))
        registered = np.copy(registered.asarray(np.uint8))[...,:3]
        return registered, undistorted

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