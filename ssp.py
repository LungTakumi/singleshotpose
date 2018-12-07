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
from threading import Thread
from videoStream import VideoStream
from queue import Queue

class SSP(Thread):
    def __init__(self, datacfg, cfgfile, weightfile, queueIn, queueOut):
        Thread.__init__(self) 
        self.queueIn = queueIn
        self.queueOut = queueOut
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

        best_conf_est = -1
        # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
        for i in range(len(boxes)):
            if (boxes[i][18] > best_conf_est):
                box_pr        = boxes[i]
                best_conf_est = boxes[i][18]

        # Denormalize the corner predictions 
        corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')            
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.internal_calibration, dtype='float32'))

        Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
        proj_corners_pr = np.transpose(compute_projection(self.corners3D, Rt_pr, self.internal_calibration))

        return R_pr, t_pr, proj_corners_pr, best_conf_est

    def run(self):
        t = time.time()
        f = 0
        while True:
            f += 1
            image = self.queueIn.get()
            R_pr, t_pr, proj_corners_pr, best_conf_est = self.process(image)
            
            fps = f//(time.time() - t + 0.000001)
            if(time.time() - t >= 1):
                f = 0
                t = time.time()
            self.queueOut.put((proj_corners_pr, fps, best_conf_est))

            print("FPS: "+str(fps))
            print("best_conf_est: "+str(best_conf_est))
            print("R_pr: ")
            print(R_pr)
            print("t_pr: ")
            print(t_pr)
		
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]

        queueIn = Queue(1)
        queueOut = Queue()

        ssp = SSP(datacfg, cfgfile, weightfile, queueIn, queueOut)
        ssp.start()
        vid = VideoStream(queueIn, queueOut)
        vid.start()
    else:
        print('Usage:')
        print(' python videoStream.py datacfg cfgfile weightfile')


