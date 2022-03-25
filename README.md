# Robopilot

![robopilot](https://user-images.githubusercontent.com/37585803/160148736-538a405f-c7e7-4ad2-bbc2-75aedd193f35.svg)

>Robotic systems for an unstructured world — Live Dense 3D Mapping — A system designed for real time 3D dense reconstruction using multiple depth sensors simultaneously at real time speed — A Generic Framework for Distributed Deep Neural Networks over the Cloud, the Edge, and End Devices for Computer Vision Applications.

Robopilot is minimalist and modular autonomous computer vision library for Python. It is developed with a focus on allowing fast experimentation with Distributed Deep Neural Networks. It is based on existing opensource work, associated machine vision, communications and motor-control libraries and the CUDA and Tensor Flow deep-learning framework.

## Use Robopilot if you want to

* Make a robot pilot its self.
* Experiment with autopilots, mapping computer vision and neural networks.
* Log sensor data. (images, user inputs, sensor readings)
* Control your robot via a web or game controller.
* Leverage distributed driving data.
* Capturing an object’s 3D structure from multiple viewpoints simultaneously,
* Capturing a “panoramic” 3D structure of a scene (extending the field of view of one sensor by using many)
* Streaming the reconstructed point cloud to a remote location,
* Increasing the density of a point cloud captured by a single sensor, by having multiple sensors capture the same scene.

## Test Platform

* Nvidia TX1 (x2)
* RedCat Crawler 1/5 (x1)
* Xbox Kinect for PC (x2)
* Raspberry Pi Cluster (x19)
* Intel RTF Drone (x1)
* Intel Neural Compute Stick (x5)

### Get Piloting

After building a Robopilot you can turn on your device and go to http://localhost:8887 to pilot.

### Modify your device behavior

The robopilot device is controlled by running a sequence of events

```python
#Define a vehicle to take and record pictures 10 times per second.

import time
from robopilot import Vehicle
from robopilot.parts.cv import CvCam
from robopilot.parts.tub_v2 import TubWriter
V = Vehicle()

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

#Add a camera part
cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
V.add(cam, outputs=['image'], threaded=True)

#warmup camera
while cam.run() is None:
    time.sleep(1)

#add tub part to record images
tub = TubWriter(path='./dat', inputs=['image'], types=['image_array'])
V.add(tub, inputs=['image'], outputs=['num_records'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```
