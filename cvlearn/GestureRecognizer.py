"""
Gesture Recognizer Module
By: CVLearn
"""

import cv2
import math
import tempfile
import time
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class GestureRecognizer:
    """
    Advanced hand gesture recognition with velocity and combo detection.
    """

    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5, modelPath=None):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelPath = self._resolve_model_path(modelPath)

        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        self.hands = self.HandLandmarker.create_from_options(
            self.HandLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.modelPath),
                running_mode=self.RunningMode.VIDEO,
                num_hands=self.maxHands,
                min_hand_detection_confidence=self.detectionCon,
                min_hand_presence_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon,
            )
        )

        self.tipIds = [4, 8, 12, 16, 20]
        self.gesture_history = []
        self.max_gesture_history = 15
        self.prev_lm = None
        self._timestamp_ms = 0
        self.results = None

    @staticmethod
    def _resolve_model_path(modelPath):
        if modelPath:
            return modelPath

        cache_dir = Path(tempfile.gettempdir()) / "cvlearn_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "hand_landmarker.task"
        if not model_file.exists():
            urllib.request.urlretrieve(HAND_MODEL_URL, model_file)
        return str(model_file)

    def findHands(self, img, draw=True):
        """
        Find hands in an image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self._timestamp_ms += 1
        self.results = self.hands.detect_for_video(mp_image, self._timestamp_ms)

        if getattr(self.results, "hand_landmarks", None) and draw:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_hand_landmarks(img, hand_landmarks)

        return img

    def getHandLandmarks(self, img, handNo=0):
        """
        Get landmarks for a specific hand.
        """
        lmList = []
        if getattr(self.results, "hand_landmarks", None) and handNo < len(self.results.hand_landmarks):
            myHand = self.results.hand_landmarks[handNo]
            h, w, _ = img.shape
            for lm in myHand:
                lmList.append([int(lm.x * w), int(lm.y * h), lm.z])
        return lmList

    def recognizeGesture(self, img, handNo=0):
        lmList = self.getHandLandmarks(img, handNo)
        if len(lmList) < 21:
            return "none", 0.0

        fingers = self._getFingerStates(lmList)
        gesture, confidence = self._classifyGesture(lmList, fingers)

        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.max_gesture_history:
            self.gesture_history.pop(0)

        return gesture, confidence

    def detectGestureVelocity(self, lmList):
        if self.prev_lm is None or len(lmList) < 21 or len(self.prev_lm) < 21:
            self.prev_lm = lmList
            return 0, "none"

        curr_center = np.mean(np.array(lmList)[:, :2], axis=0)
        prev_center = np.mean(np.array(self.prev_lm)[:, :2], axis=0)
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        velocity = math.sqrt(dx**2 + dy**2)

        if velocity > 0:
            angle = math.degrees(math.atan2(dy, dx))
            if -45 <= angle <= 45:
                direction = "right"
            elif 45 < angle <= 135:
                direction = "down"
            elif -135 <= angle < -45:
                direction = "up"
            else:
                direction = "left"
        else:
            direction = "none"

        self.prev_lm = lmList
        return velocity, direction

    def detectGestureCombo(self):
        if len(self.gesture_history) < 3:
            return None

        recent = self.gesture_history[-3:]
        combos = {
            ("thumbs_up", "open_hand", "thumbs_up"): "celebrate",
            ("peace", "peace", "peace"): "triple_peace",
            ("rock", "rock", "rock"): "rock_n_roll",
            ("ok", "ok", "ok"): "approval",
        }
        return combos.get(tuple(recent), None)

    def getHandDistance(self, img):
        if not getattr(self.results, "hand_landmarks", None) or len(self.results.hand_landmarks) < 2:
            return None, None

        lm1 = self.getHandLandmarks(img, 0)
        lm2 = self.getHandLandmarks(img, 1)
        if len(lm1) < 21 or len(lm2) < 21:
            return None, None

        center1 = np.mean(np.array(lm1)[:, :2], axis=0)
        center2 = np.mean(np.array(lm2)[:, :2], axis=0)
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        midpoint = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)
        return distance, midpoint

    def drawGestureInfo(self, img, gesture, confidence, handNo=0):
        x_pos = 50 + (handNo * 300)
        cv2.rectangle(img, (x_pos, 50), (x_pos + 250, 150), (0, 255, 0), 2)
        cv2.putText(img, f"Hand {handNo + 1}", (x_pos + 10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Gesture: {gesture}", (x_pos + 10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(img, f"Confidence: {confidence:.2f}", (x_pos + 10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return img

    def getHandOrientation(self, lmList):
        if len(lmList) < 21:
            return "unknown"

        hand_width = abs(max(p[0] for p in lmList) - min(p[0] for p in lmList))
        hand_height = abs(max(p[1] for p in lmList) - min(p[1] for p in lmList))
        return "side_view" if hand_width > hand_height else "palm_facing"

    def _getFingerStates(self, lmList):
        fingers = []
        fingers.append(1 if lmList[4][0] < lmList[3][0] else 0)
        for tip_id in self.tipIds[1:]:
            fingers.append(1 if lmList[tip_id][1] < lmList[tip_id - 2][1] else 0)
        return fingers

    def _classifyGesture(self, lmList, fingers):
        fingers_up = sum(fingers)
        if fingers_up == 5:
            return "open_hand", 0.9
        if fingers_up == 0:
            return "fist", 0.9
        if fingers_up == 1 and fingers[0] == 1:
            return "thumbs_up", 0.8
        if fingers_up == 2 and fingers[1] == 1 and fingers[2] == 1:
            return "peace", 0.8
        if fingers_up == 2 and fingers[3] == 1 and fingers[4] == 1:
            return "rock", 0.8
        if fingers_up == 1 and fingers[1] == 1:
            thumb_index_dist = math.sqrt((lmList[4][0] - lmList[8][0])**2 + (lmList[4][1] - lmList[8][1])**2)
            if thumb_index_dist < 50:
                return "ok", 0.8
            return "pointing", 0.8
        return "unknown", 0.5

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

    def main(self):
        pass
