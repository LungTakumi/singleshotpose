import cv2

def run():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (640,480))
        cv2.imshow('object detection', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()

if __name__ == '__main__':
    run()