# cvlearn

An easy-to-use package that helps with hand tracking, face detection, and more using OpenCV.

---

## Installation

* Use **Python 3.x**
* Open your terminal or command prompt and run:

```bash
pip install cvlearn
```

---

## Dependencies

* python 3.x
* opencv-python
* numpy
* mediapipe

---

## Examples

### Hand Tracking

```python
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

**Result:**

![Hand Tracking](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/handTracking.jpg)

---

### Face Detection

```python
from cvlearn import FaceDetection as faceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = faceDetector.FaceDetector()

while True:
    ret, img = cap.read()
    img = detector.findFaces(img)
    
    cv2.imshow("Result", img)
    cv2.waitKey(1)
```

**Result:**

![Face Detection](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/faceDetection.jpg)

**Side View:**

![Face Side View](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/faceDetection2.jpg)

---

### Face Mesh

```python
from cvlearn import FaceMesh as fms
import cv2

cap = cv2.VideoCapture(0)
detector = fms.FaceMeshDetector()

while True:
    ret, img = cap.read()
    img, face = detector.findFaceMesh(img)

    cv2.imshow("Result", img)
    cv2.waitKey(1)
```

**Result:**

![Face Mesh](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/faceMesh.jpg)

---

### Finger Counting

```python
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
        frame = counter.drawCountedFingers(frame, lmList, bbox)

    cv2.imshow("res", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
```

**Result:**

![Finger Counter](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/fingerCounter.jpg)

---

### Two Hands Finger Counting

```python
from cvlearn import TwoHandsFingerCounter as fc
import cv2

cap = cv2.VideoCapture(0)
counter = fc.FingerCounter()

while True:
    ret, frame = cap.read()
    frame = counter.drawCountedFingers(frame)

    cv2.imshow("res", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
```

**Result:**

![Two Hands Counter](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/MultipleHandsFingerCounter.jpg)

---

### Pose Detection

```python
import cv2
import cvlearn
from cvlearn import PoseDetector, Utils

cap = cv2.VideoCapture(0)
detector = PoseDetector.PoseDetector(detectionCon=0.5, trackCon=0.5)

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findLandmarks(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Result:**

![Pose Detection](https://raw.githubusercontent.com/Tanay-ErrorCode/cvlearn/main/images/poseDetection.jpg)
