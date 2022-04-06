import cv2
import mediapipe as mp
import math


class handDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=-1,
                                                                       circle_radius=5)
                                               )

                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (0, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 255), 2)
        return allHands, img

    def findPosition(self, img, handNo=0, draw=True):
        """

        :param img: Image to find the hand's position.
        :param handNo: Index number of hand to find position
        :param draw: Flag to draw the output on the image.
        :return: list of fingers and bbox.
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([px, py])

                if draw:
                    cv2.circle(img, (px, py), 5, (0, 255, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + (bbox[3] // 2)

        return self.lmList, bbox

    def findDistance(self, p1, p2, img, draw=True):

        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        if self.results.multi_hand_landmarks:
            # print(self.lmList[p1])
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [cx, cy]


tipIds = [4, 8, 12, 16, 20]


class FingerCounter:
    """A class to count Fingers of two hands."""

    def __init__(self):
        self.detector = handDetector(maxHands=2)

    def drawCountedFingers(self, img):

        """

        :param img: Image of hand to count fingers in.
        """

        allHands, frame = self.detector.findHands(img)

        fingers1 = []
        fingers2 = []
        fingers = []

        if len(allHands) == 2:

            lmList1, bbox1 = allHands[1]['lmList'], allHands[1]['bbox']
            if lmList1:
                if lmList1[5][0] < lmList1[17][0]:
                    if lmList1[tipIds[0]][0] < lmList1[tipIds[0] - 1][0]:
                        fingers1.append(1)
                    else:
                        fingers1.append(0)
                    for id in range(1, 5):
                        if lmList1[tipIds[id]][1] < lmList1[tipIds[id] - 1][1]:
                            fingers1.append(1)
                        else:
                            fingers1.append(0)
                    img = cv2.putText(img, f"{fingers1.count(1)}", (500, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0),
                                      5)
            lmList2, bbox2 = allHands[0]['lmList'], allHands[0]['bbox']
            if lmList2:
                if lmList2[5][0] > lmList2[17][0]:
                    if lmList2[tipIds[0]][0] > lmList2[tipIds[0] - 1][0]:
                        fingers2.append(1)
                    else:
                        fingers2.append(0)
                    for id in range(1, 5):
                        if lmList2[tipIds[id]][1] < lmList2[tipIds[id] - 1][1]:
                            fingers2.append(1)
                        else:
                            fingers2.append(0)
                cv2.putText(img, f"{fingers2.count(1)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)

        if len(allHands) == 1:
            lmList, bbox = allHands[0]['lmList'], allHands[0]['bbox']

            if lmList:
                if lmList[5][0] > lmList[17][0]:

                    if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    for id in range(1, 5):
                        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    img = cv2.putText(img, f"{fingers.count(1)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0),
                                      5)

                if lmList[5][0] < lmList[17][0]:

                    if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    for id in range(1, 5):
                        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    img = cv2.putText(img, f"{fingers.count(1)}", (500, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0),
                                      5)
        if len(allHands) > 2:
            pass
# mediapipe : 0.8.9.1