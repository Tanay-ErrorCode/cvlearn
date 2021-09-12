import cv2
import cvlearn.HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]


class FingerCounter:
    """A class to count Fingers"""
    def __init__(self):
        self.hi = ""

    def drawCountedFingers(self, img, lmList, bbox):
        """

        :param img: Image of hand to count fingers in.
        :param lmList: list returned by find position in HandTrackingModule
        :param bbox: bbox returned by find position in HandTrackingModule
        """
        fingers = []

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
                img = cv2.putText(img, f"{fingers.count(1)}", (500, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)

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
                img = cv2.putText(img, f"{fingers.count(1)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)

            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                          (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                          (255, 0, 255), 2)

    def countFingers(self, lmList):
        """

        :param lmList: list returned by find position in HandTrackingModule
        :return: List of fingers up or down
        """
        fingers = []

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
            totalFingers = fingers.count(1)
            return totalFingers
