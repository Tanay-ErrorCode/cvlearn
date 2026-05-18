from . import HandTrackingModule
from . import FingerCounter
from . import FaceDetection
from . import FaceMesh
from . import PoseDetector
from .GestureRecognizer import GestureRecognizer
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package],
    stderr=subprocess.DEVNULL)
try:
    install('msvc-runtime')
except Exception as e:
    pass
__all__ = [
    "HandTrackingModule",
    "FingerCounter",
    "FaceDetection",
    "FaceMesh",
    "PoseDetector",
    "GestureRecognizer",
]
 