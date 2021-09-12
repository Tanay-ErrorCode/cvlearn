# cvlearn

An easy to use package which helps to do hand tracking, face detection, etc. with use of opencv module.

# Installation
   - Use Python 3.x
   - Open cmd/terminal and type:

   ```bash
pip install cvlearn
   ```

# Dependencies
- python 3.x
- opencv-python
- numpy
- mediapipe

# Examples
**Hand Tracking:**

```bash
from cvlearn import HandTrackingModule as handTracker

import cv2
cap = cv2.VideoCapture(0)
detector = handTracker.handDetector()
while True:
    ret, img = cap.read()
    img = detector.findHands(img)

    cv2.imshow("Result", img)
    cv2.waitKey(1)

```

**Face Detection:**
```bash
from cvlearn import FaceDetector as faceDetector

import cv2
cap = cv2.VideoCapture(0)
detector = faceDetector.FaceDetector()
while True:
    ret, img = cap.read()
    img = detector.findFaces(img)

    cv2.imshow("Result", img)
    cv2.waitKey(1)
```
**Drawing face mesh:**
```bash
from cvlearn import FaceMesh as fms

import cv2
cap = cv2.VideoCapture(0)
detector = fms.FaceMeshDector()
while True:
    ret, img = cap.read()
    img = detector.findFaceMesh(img)

    cv2.imshow("Result", img)
    cv2.waitKey(1)

```
**Counting fingers:**
```bash
from cvlearn import FingerCounter as fc
import cvlearn.HandTrackingModule as handTracker
import cv2

cap = cv2.VideoCapture(0)

detector = handTracker.handDetector(maxHands=1)

counter = fc.FingerCounter()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 180)

    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)

    if lmList:
        frame1 = counter.drawCountedFingers(frame, lmList, bbox)

    cv2.imshow("res", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

```



