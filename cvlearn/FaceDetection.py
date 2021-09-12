import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, minDetectionCon=0.5):
        self.face_i = []
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.face_num = 0
        self.faces = []

    def findFaces(self, img, draw=True):
        """

        :param img: Image to find faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        self.faces = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                self.face_i.append((cx, cy))
                if draw:
                    img = cv2.rectangle(img, bbox, (0, 255, 0), 2)
                self.faces.append(self.face_i)

        return img

    def CalculateNumberOfFaces(self):
        """
        :return: Calculates Number of faces.
        """
        return len(self.faces)
