"""
Pose Detector Module
By: CVLearn
"""

import cv2
import math
import tempfile
import urllib.request
from pathlib import Path

import mediapipe as mp


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker/full/latest/pose_landmarker.task"
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
]


class PoseDetector:
    def __init__(self, staticMode=False, modelComplexity=1, smoothLandmarks=True,
                 enableSegmentation=False, smoothSegmentation=True,
                 detectionCon=0.5, trackCon=0.5, modelPath=None):
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        self.modelPath = self._resolve_model_path(modelPath)
        self.pose = self.PoseLandmarker.create_from_options(
            self.PoseLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.modelPath),
                running_mode=self.RunningMode.IMAGE if self.staticMode else self.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.detectionCon,
                min_pose_presence_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon,
                output_segmentation_masks=self.enableSegmentation,
            )
        )
        self._timestamp_ms = 0
        self.results = None

    @staticmethod
    def _resolve_model_path(modelPath):
        if modelPath:
            return modelPath

        cache_dir = Path(tempfile.gettempdir()) / "cvlearn_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "pose_landmarker.task"
        if not model_file.exists():
            urllib.request.urlretrieve(POSE_MODEL_URL, model_file)
        return str(model_file)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        if self.staticMode:
            self.results = self.pose.detect(mpImage)
        else:
            self._timestamp_ms += 1
            self.results = self.pose.detect_for_video(mpImage, self._timestamp_ms)

        if getattr(self.results, "pose_landmarks", None) and draw:
            self._draw_pose(img, self.results.pose_landmarks[0])
        return img

    def findLandmarks(self, img, draw=True, returnZ=False):
        lmList = []
        if getattr(self.results, "pose_landmarks", None):
            h, w, _ = img.shape
            for lm in self.results.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if returnZ:
                    lmList.append([cx, cy, lm.z])
                else:
                    lmList.append([cx, cy])
            if draw:
                self._draw_pose(img, self.results.pose_landmarks[0])
        return lmList

    def _draw_pose(self, img, landmarks):
        h, w, _ = img.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(img, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(img, point, 3, (0, 255, 0), cv2.FILLED)


PoseDetector = PoseDetector
