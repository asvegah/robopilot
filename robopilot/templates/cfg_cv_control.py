""" 
CAR CONFIG 

This file is read by your car application's manage.py script to change the car
performance. 

EXMAPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

#CAMERA
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono

#9865, over rides only if needed, ie. TX2..
PCA9685_I2C_ADDR = 0x40
PCA9685_I2C_BUSNUM = None

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 460
STEERING_RIGHT_PWM = 290

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 500
THROTTLE_STOPPED_PWM = 370
THROTTLE_REVERSE_PWM = 220


#RobopilotGym
#Only on Ubuntu linux, you can use the simulator as a virtual robopilot and
#issue the same python manage.py drive command as usual, but have them control a virtual car.
#This enables that, and sets the path to the simualator and the environment.
#You will want to download the simulator binary from: https://github.com/tawnkramer/robopilot_gym/releases/download/v18.9/RobopilotSimLinux.zip
#then extract that and modify ROBOPILOT_SIM_PATH.
ROBOPILOT_GYM = False
ROBOPILOT_SIM_PATH = "remote" #"/home/tkramer/projects/sdsandbox/sdsim/build/RobopilotSimLinux/robopilot_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
ROBOPILOT_GYM_ENV_NAME = "robopilot-generated-track-v0" # ("robopilot-generated-track-v0"|"robopilot-generated-roads-v0"|"robopilot-warehouse-v0"|"robopilot-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "robopilot", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(robopilot|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"              # when racing on virtual-race-league use host "trainmyrobopilot.com"
SIM_ARTIFICIAL_LATENCY = 0          # this is the millisecond latency in controls. Can use useful in emulating the delay when useing a remote server. values of 100 to 400 probably reasonable.
