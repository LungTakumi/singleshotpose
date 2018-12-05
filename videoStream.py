import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import scipy.misc

from darknet import Darknet
import videoDataset as dataset
from utils import *
from MeshPly import MeshPly
import cv2

class videoStream(object):
    def __init__(self, datacfg, cfgfile, weightfile):
        self.isShow = True
        self.cap = cv2.VideoCapture(0)
        # Parse configuration files
        options      = read_data_cfg(datacfg)
        meshname     = options['mesh']

        # Parameters
        seed         = int(time.time())
        gpus         = '0'     # Specify which gpus to use
        self.test_width   = 544
        self.test_height  = 544
        torch.manual_seed(seed)
        self.use_cuda = True
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)
        self.num_classes     = 1
        self.conf_thresh     = 0.1
        self.nms_thresh      = 0.4
        self.edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

        # Read object model information, get 3D bounding box corners
        mesh          = MeshPly(meshname)
        vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        self.corners3D     = get_3D_corners(vertices)

        # Read intrinsic camera parameters
        self.internal_calibration = get_camera_intrinsic()

        self.model = Darknet(cfgfile)
        self.model.print_network()
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()

    def process(self, image):

        # For each image, get all the predictions
        img = cv2.resize(image,(self.test_width, self.test_height))
        boxes   = do_detect(self.model,img,self.conf_thresh,self.nms_thresh)     

        if(len(boxes) == 0):
            return boxes

        box_pr = boxes[0]
        # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
        for i in range(len(boxes)):
            if (boxes[i][18] > box_pr[18]):
                box_pr        = boxes[i]

        # Denormalize the corner predictions 
        corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')            
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.internal_calibration, dtype='float32'))

        if(self.isShow):
            Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
            proj_corners_pr = np.transpose(compute_projection(self.corners3D, Rt_pr, self.internal_calibration))

            return proj_corners_pr
        else:
            return R_pr, t_pr

    def run(self):
        t = time.time()
        f = 0
        isProcess = True
        isShow = self.isShow
        while True:
            f += 1
            success, image = self.cap.read()
            
            if(isProcess):
                if(isShow):
                    proj_corners_pr = self.process(image)
                    img = cv2.resize(image, (640,480))
                    if(len(proj_corners_pr) > 0):
                        for edge in self.edges_corners:
                            px = proj_corners_pr[edge, 0]
                            py = proj_corners_pr[edge, 1]
                            cv2.line(img,(px[0], py[0]), (px[1], py[1]), (255,0,0), 3)
                else:
                    R_pr, t_pr = self.process(image)
            
            fps = f//(time.time() - t)
            if(isShow):
                img = cv2.putText(img, "FPS: "+str(fps), (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                cv2.imshow('object detection', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                print("FPS: "+str(fps))
                print("R_pr: ")
                print(R_pr)
                print("t_pr: ")
                print(t_pr)
        self.cap.release()
		
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        vid = videoStream(datacfg, cfgfile, weightfile)
        vid.run()
    else:
        print('Usage:')
        print(' python videoStream.py datacfg cfgfile weightfile')


