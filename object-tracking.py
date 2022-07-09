import cv2
from cv2 import boundingRect
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("/Users/harshareddy/Desktop/iymw/videoplayback.mp4")


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)


while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    RegINT = frame[140:320, 0:440]
    
    mask = object_detector.apply(RegINT)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(RegINT, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(RegINT, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(RegINT, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(RegINT, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("RegINT", RegINT)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cap.destroyAllWindows()

