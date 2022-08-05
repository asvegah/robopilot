import cv2
import robopilot.parts.kinectv2 as kv2

k = kv2.Kinect()

while True:
    color_frame = k.get_frame()
    
    cv2.imshow('frame', color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break