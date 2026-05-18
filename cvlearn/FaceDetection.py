"""
Face Detection Module
By: CVLearn
Website: https://github.com/Tanay-ErrorCode/cvlearn
"""

import cv2
import mediapipe as mp
import tempfile
import time
import urllib.request
from pathlib import Path


SHORT_RANGE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/face_detector/blaze_face_short_range/float16/latest/face_detector.task"
)

FULL_RANGE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/face_detector/blaze_face_full_range/float16/latest/face_detector.task"
)


class FaceDetector:
    """
    Find faces in realtime using the lightweight MediaPipe Tasks face detector.
    """

    def __init__(self, minDetectionCon=0.5, modelSelection=0, modelPath=None):
        """
        :param minDetectionCon: Minimum confidence score for a face to count.
        :param modelSelection: 0 for short range, 1 for full range.
        :param modelPath: Optional path to a local .task model file.
        """
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.modelPath = self._resolve_model_path(modelPath, modelSelection)

        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetectorTask = mp.tasks.vision.FaceDetector
        self.FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        self.faceDetection = self.FaceDetectorTask.create_from_options(
            self.FaceDetectorOptions(
                base_options=self.BaseOptions(model_asset_path=self.modelPath),
                running_mode=self.RunningMode.IMAGE,
                min_detection_confidence=self.minDetectionCon,
                min_suppression_threshold=0.3,
            )
        )
        self.faces = []
        self.results = None

    @staticmethod
    def _resolve_model_path(modelPath, modelSelection):
        if modelPath:
            return modelPath

        cache_dir = Path(tempfile.gettempdir()) / "cvlearn_models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_file = cache_dir / (
            "face_detector_full_range.task" if modelSelection == 1 else "face_detector_short_range.task"
        )
        if model_file.exists():
            return str(model_file)

        model_url = FULL_RANGE_MODEL_URL if modelSelection == 1 else SHORT_RANGE_MODEL_URL
        try:
            urllib.request.urlretrieve(model_url, model_file)
            return str(model_file)
        except Exception as exc:
            raise RuntimeError(
                "Could not download a default face detector model. "
                "Pass modelPath=... with a local .task file."
            ) from exc

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return bounding-box info.
        :param img: Image to find faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings, bounding box list.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.faceDetection.detect(mpImage)
        bboxs = []
        self.faces = []

        if self.results and self.results.detections:
            ih, iw, _ = img.shape

            for face_id, detection in enumerate(self.results.detections):
                confidence = detection.categories[0].score if detection.categories else 0.0
                if confidence < self.minDetectionCon:
                    continue

                bboxC = detection.bounding_box
                bbox = (
                    int(bboxC.origin_x),
                    int(bboxC.origin_y),
                    int(bboxC.width),
                    int(bboxC.height),
                )
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                bboxInfo = {
                    "id": face_id,
                    "bbox": bbox,
                    "score": [confidence],
                    "center": (cx, cy),
                }
                bboxs.append(bboxInfo)
                self.faces.append(bboxInfo)

                if draw:
                    self._draw_face_box(img, bbox, confidence)

        return img, bboxs

    def CalculateNumberOfFaces(self):
        """
        :return: Calculates number of detected faces.
        """
        return len(self.faces)

    @staticmethod
    def _draw_face_box(img, bbox, confidence):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{int(confidence * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


def main():
    cap = cv2.VideoCapture(0)

    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img, draw=False)

        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox["bbox"]
            score = int(bbox["score"][0] * 100)

            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'{score}%', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
