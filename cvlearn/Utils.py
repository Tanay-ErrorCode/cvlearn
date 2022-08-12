import math

import cv2

rightEye = {'eyeUp': 257, 'eyeDown': 253, 'eyeRight': 359, 'eyeLeft': 463}
leftEye = {'eyeUp': 27, 'eyeDown': 23, 'eyeRight': 243, 'eyeLeft': 130}
mouth = {'mouthUp': 0, 'mouthDown': 17, 'mouthRight': 291, 'mouthLeft': 61}
face = {'faceUp': 10, 'faceDown': 152, 'faceRight': 454, 'faceLeft': 234}


def find_rotation(p1, p2):
    Rad2Deg = 180.0 / math.pi
    rotation = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * Rad2Deg

    return rotation


def findDistance(p1, p2, img, draw=False):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    return length, [cx, cy]
