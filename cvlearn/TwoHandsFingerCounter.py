"""
Two Hands Finger Counter Module
By: CVLearn
"""

import cv2

from cvlearn.HandTrackingModule import handDetector as BaseHandDetector


tipIds = [4, 8, 12, 16, 20]


class FingerCounter:
    """A class to count fingers on two hands."""

    def __init__(self):
        self.detector = BaseHandDetector(maxHands=2)

    def drawCountedFingers(self, img):
        """Count fingers and draw the total on the frame."""
        allHands, frame = self.detector.findHands(img)
        fingers = []

        if len(allHands) == 2:
            for hand_index, x_pos in ((1, 500), (0, 10)):
                lmList = allHands[hand_index]["lmList"]
                if lmList:
                    hand_fingers = self._count_fingers(lmList)
                    fingers.append(hand_fingers)
                    cv2.putText(img, f"{hand_fingers.count(1)}", (x_pos, 100),
                                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)

        if len(allHands) == 1:
            lmList = allHands[0]["lmList"]
            if lmList:
                hand_fingers = self._count_fingers(lmList)
                fingers.append(hand_fingers)
                cv2.putText(img, f"{hand_fingers.count(1)}", (10, 100),
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 5)

        return img

    @staticmethod
    def _count_fingers(lmList):
        fingers = []
        if lmList[5][0] > lmList[17][0]:
            fingers.append(1 if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0] else 0)
        else:
            fingers.append(1 if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0] else 0)

        for idx in range(1, 5):
            fingers.append(1 if lmList[tipIds[idx]][1] < lmList[tipIds[idx] - 1][1] else 0)
        return fingers
