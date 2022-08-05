import cv2
import robopilot.parts.kinectv2 as kv2
import numpy as np
from matplotlib import cm

k = kv2.Kinect(device_index=1)

while True:
    color_frame, depth_frame, ir_frame = k.get_frame([kv2.COLOR, kv2.DEPTH, kv2.IR])
    
    #cv2.imshow('color', color_frame)
    #cv2.imshow('depth', depth_frame)
    cv2.imshow('ir', ir_frame / 65535.)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

""" def colorize_image(img, cmap="gray", vmin=None, vmax=None):
    assert len(img.shape) < 3

    # Get the min and max
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()

    # Clip and rescale
    img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    if cmap is None or cmap == "None":
        return (np.dstack((img, img, img, np.ones((img.shape)))) * 255).astype(np.uint8)

    if cmap == "fire":
        cmap = fire
    else:
        cmap = cm.get_cmap(cmap)

    # Apply the colormap
    return (cmap(img) * 255).astype(np.uint8) """