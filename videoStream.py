import time
import warnings
warnings.filterwarnings("ignore")
import cv2
from threading import Thread
import queue

class VideoStream(Thread):
    def __init__(self, queueIn, queueOut):
        Thread.__init__(self) 
        self.queueIn = queueIn
        self.queueOut = queueOut
        self.edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    def run(self):
        cap = cv2.VideoCapture(0)
        t = time.time()
        f = 0
        while True:
            f += 1
            success, image = cap.read()
            self.queueIn.put(image)
            
            image = cv2.resize(image, (640,480))
            try:
                proj_corners_pr = self.queueOut.get(False)
            except queue.Empty:
                pass
            else:
                if(len(proj_corners_pr) > 0):
                    for edge in self.edges_corners:
                        px = proj_corners_pr[edge, 0]
                        py = proj_corners_pr[edge, 1]
                        cv2.line(image,(px[0], py[0]), (px[1], py[1]), (255,0,0), 3)


            fps = f//(time.time() - t + 0.000001)
            image = cv2.putText(image, "FPS: "+str(fps), (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
            cv2.imshow('object detection', image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.queueIn.task_done()
                break
        cap.release()




