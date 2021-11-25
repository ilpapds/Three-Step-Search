import cv2
import numpy as np

# Opens the Video file
cap = cv2.VideoCapture('sample.mp4')
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    w,h,_ = np.shape(frame)
    frame = cv2.resize(frame,(int(h/3),int(w/3)))
    if ret == False:
        break
    if i%50 == 0:
        cv2.imwrite(filename='frame_50.jpg',img=frame)
    if i%51 == 0:
        cv2.imwrite(filename='frame_51.jpg',img=frame)
        break
    i+=1

cap.release()
cv2.destroyAllWindows()