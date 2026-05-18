from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent

long_description = (here / "README.md").read_text()

VERSION = '0.5.3'
DESCRIPTION = 'Coumputer Vision helping Library'
setup(
    name="cvlearn",
    version=VERSION,
    author="ErrorCode (Tanay)",
    author_email="codingkaku@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'opencv-python', 'mediapipe'],
    keywords=[
        'python', 'opencv', 'computer vision', 'cv2', 'image processing', 
        'mediapipe', 'pose detection', 'face mesh', 'hand tracking', 'cvlearn'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.x",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video",
    ]
)