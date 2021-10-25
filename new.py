from cvlearn import FaceMesh as fms

import cv2
cap = cv2.VideoCapture("http://192.168.29.232:4747/video")
detector = fms.FaceMeshDetector()
while True:
    ret, img = cap.read()
    img, face = detector.findFaceMesh(img)

    cv2.imshow("Result", img)
    cv2.waitKey(1)

