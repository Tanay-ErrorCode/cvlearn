"""
Face Mesh Module
By: CVLearn
"""

import cv2
import mediapipe as mp
import tempfile
import urllib.request
from pathlib import Path


FACE_MESH_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

FACE_CONNECTIONS = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
]


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5, RefineLandmarks=False, modelPath=None):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refine_landmarks = RefineLandmarks

        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        self.modelPath = self._resolve_model_path(modelPath)
        self.faceMesh = self.FaceLandmarker.create_from_options(
            self.FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.modelPath),
                running_mode=self.RunningMode.IMAGE if self.staticMode else self.RunningMode.VIDEO,
                num_faces=self.maxFaces,
                min_face_detection_confidence=self.minDetectionCon,
                min_face_presence_confidence=self.minDetectionCon,
                min_tracking_confidence=self.minTrackCon,
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
        model_file = cache_dir / "face_landmarker.task"
        if not model_file.exists():
            urllib.request.urlretrieve(FACE_MESH_MODEL_URL, model_file)
        return str(model_file)

    def findFaceMesh(self, img, draw=True):
        """
        Find face landmarks in an image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        if self.staticMode:
            self.results = self.faceMesh.detect(mpImage)
        else:
            self._timestamp_ms += 1
            self.results = self.faceMesh.detect_for_video(mpImage, self._timestamp_ms)

        faces = []
        if getattr(self.results, "face_landmarks", None):
            for faceLms in self.results.face_landmarks:
                face = []
                for lm in faceLms:
                    ih, iw, _ = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
                    face.append([x, y, z])
                    if draw:
                        cv2.circle(img, (x, y), 1, (0, 255, 0), cv2.FILLED)
                faces.append(face)
                if draw:
                    self._draw_connections(img, face)
        return img, faces

    def _draw_connections(self, img, face):
        for start, end in FACE_CONNECTIONS:
            if start < len(face) and end < len(face):
                cv2.line(img, (face[start][0], face[start][1]), (face[end][0], face[end][1]), (0, 255, 0), 1)


FaceMesh = FaceMeshDetector
