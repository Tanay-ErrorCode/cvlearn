import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package],
    stderr=subprocess.DEVNULL)
try:
    install('msvc-runtime')
except Exception as e:
    pass

from cvlearn import HandTrackingModule
from cvlearn import FingerCounter
from cvlearn import FaceDetection
from cvlearn import FaceMesh
from cvlearn import PoseDetector
