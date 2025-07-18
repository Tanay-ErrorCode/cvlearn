import cv2
import math
import mediapipe as mp


class PoseDetector:

    def __init__(self, staticMode=False, modelComplexity=1, smoothLandmarks=True,
                 enableSegmentation=False, smoothSegmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=staticMode,
            model_complexity=modelComplexity,
            smooth_landmarks=smoothLandmarks,
            enable_segmentation=enableSegmentation,
            smooth_segmentation=smoothSegmentation,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=3)

    def findPose(self, img, draw=True):
        """
        This Python function finds and draws pose landmarks on an image if they are detected.
        
        :param img: The `img` parameter is the input image that you want to process and detect poses on.
        It is expected to be in BGR format
        :param draw: The `draw` parameter in the `findPose` method is a boolean parameter that
        determines whether the pose landmarks should be drawn on the image or not. If `draw` is set to
        `True`, the method will draw the pose landmarks on the image using the `mpDraw.draw_landmarks`,
        defaults to True (optional)
        :return: The function `findPose` returns the image with the pose landmarks drawn on it if `draw`
        is set to `True`. If `draw` is set to `False`, it returns the original image without any
        modifications.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findLandmarks(self, img, draw=True, returnZ=False):
        """
        The function `findLandmarks` extracts landmarks from a given image and optionally draws them on
        the image while also providing the option to return the relative depth information.
        
        :param img: The `img` parameter in the `findLandmarks` function is the image on which you want
        to detect and draw landmarks. It is the input image where the pose landmarks will be identified
        and potentially drawn on if the `draw` parameter is set to `True`
        :param draw: The `draw` parameter in the `findLandmarks` function is a boolean parameter that
        determines whether the landmarks should be drawn on the image or not. If `draw` is set to
        `True`, the landmarks will be visualized on the image using the `mpDraw.draw_landmarks` method,
        defaults to True (optional)
        :param returnZ: The `returnZ` parameter in the `findLandmarks` function determines whether the
        function should include the relative depth information (z-coordinate) of the landmarks in the
        output list `lmList`, defaults to False (optional)
        :return: The function `findLandmarks` returns a list of landmarks detected in the input image.
        The landmarks are represented as either [cx, cy] (if `returnZ` is False) or [cx, cy, cz] (if
        `returnZ` is True), where cx and cy are the coordinates of the landmark in the image and cz is
        the relative depth (usually negative if in front
        """

        lmList = []
        if self.results.pose_landmarks:
            h, w, _ = img.shape
            for lm in self.results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if returnZ:
                    cz = lm.z
                    lmList.append([cx, cy, cz])
                else:
                    lmList.append([cx, cy])
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return lmList
