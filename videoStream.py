import time
import warnings
warnings.filterwarnings("ignore")
import cv2
from threading import Thread

class VideoStream(Thread):
    def __init__(self, queue):
        Thread.__init__(self) 
        self.queue = queue
    def run(self):
        cap = cv2.VideoCapture(0)
        t = time.time()
        f = 0
        while True:
            f += 1
            success, image = cap.read()
            self.queue.put(image)
            
            fps = f//(time.time() - t + 0.000001)
            image = cv2.putText(image, "FPS: "+str(fps), (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
            cv2.imshow('object detection', image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.queue.task_done()
                break
        cap.release()




