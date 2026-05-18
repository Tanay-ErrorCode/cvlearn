import math
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

DEFAULT_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

class handDetector:

    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.5, modelPath=None):
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

        self.mpHands = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.BaseOptions = mp.tasks.BaseOptions

        self.modelPath = self._resolve_model_path(modelPath)
        self.running_mode = self.VisionRunningMode.IMAGE if self.mode else self.VisionRunningMode.VIDEO
        self._timestamp_ms = 0

        self.hands = self.mpHands.create_from_options(
            self.HandLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.modelPath),
                running_mode=self.running_mode,
                num_hands=self.maxHands,
                min_hand_detection_confidence=self.detectionCon,
                min_hand_presence_confidence=self.detectionCon,
                min_tracking_confidence=self.minTrackCon,
            )
        )
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    @staticmethod
    def _resolve_model_path(modelPath):
        if modelPath:
            return modelPath

        cache_dir = Path(tempfile.gettempdir()) / "cvlearn_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "hand_landmarker.task"

        if not model_file.exists():
            try:
                urllib.request.urlretrieve(DEFAULT_HAND_MODEL_URL, model_file)
            except Exception as exc:
                raise RuntimeError(
                    "MediaPipe Tasks requires a hand landmarker model file. "
                    "Pass modelPath=... or install a compatible cached model."
                ) from exc

        return str(model_file)

    def findHands(self, img, draw=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        if self.running_mode == self.VisionRunningMode.IMAGE:
            self.results = self.hands.detect(mp_image)
        else:
            self._timestamp_ms += 1
            self.results = self.hands.detect_for_video(mp_image, self._timestamp_ms)

        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_hand_landmarks(img, hand_landmarks)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """

        :param img: Image to find the hand's position.
        :param handNo: Index number of hand to find position
        :param draw: Flag to draw the output on the image.
        :return: list of fingers and bbox.
        """
        xList = []
        yList = []
        zList = []
        bbox = []
        self.lmList = []
        if getattr(self.results, "hand_landmarks", None):
            if handNo >= len(self.results.hand_landmarks):
                return self.lmList, bbox

            myHand = self.results.hand_landmarks[handNo]
            for idx, lm in enumerate(myHand):
                h, w, c = img.shape
                px, py, pz = int(lm.x * w), int(lm.y * h), lm.z
                xList.append(px)
                yList.append(py)
                zList.append(pz)
                self.lmList.append([px, py, pz])

                if draw:
                    cv2.circle(img, (px, py), 5, (0, 255, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH

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

        if getattr(self.results, "hand_landmarks", None):
            x1, y1 = self.lmList[p1][0], self.lmList[p1][1]
            x2, y2 = self.lmList[p2][0], self.lmList[p2][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [cx, cy]

        return 0, img, []

    def _draw_hand_landmarks(self, img, hand_landmarks):
        h, w, _ = img.shape
        points = []

        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))
            cv2.circle(img, (px, py), 5, (0, 255, 0), cv2.FILLED)

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 2)
