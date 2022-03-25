"""
Script to view a robopilot camera remotely (when published using TcpServer)

Usage:
    remote_cam_view_tcp.py (--ip=<ip_address>) [--record=<path>]

Options:
    -h --help     Show this screen.
    --record=<path>      If data should be recorded (locally) specify the path
    
"""
from docopt import docopt
import robopilot as dk
from robopilot.parts.cv import CvImageView, ImgBGR2RGB, ImageScale
from robopilot.parts.network import TCPClientValue
from robopilot.parts.image import JpgToImgArr

args = docopt(__doc__)
print(args)

V = dk.vehicle.Vehicle()
V.add(TCPClientValue("camera", args["--ip"]), outputs=["jpg"])
V.add(JpgToImgArr(), inputs=["jpg"], outputs=["img_arr"])
V.add(ImgBGR2RGB(), inputs=["img_arr"], outputs=["rgb"])
V.add(ImageScale(4.0), inputs=["rgb"], outputs=["lg_img"])
V.add(CvImageView(), inputs=["lg_img"])

# Local saving of images?
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
