from setuptools import setup, find_packages
import pathlib
import os
here = pathlib.Path(__file__).parent

long_description = (here / "README.md").read_text()

VERSION = '0.4.3'
DESCRIPTION = 'Coumputer Vision helping Library'
print(long_description)
setup(
    name="cvlearn",
    version=VERSION,
    author="ErrorCode (Tanay)",
    author_email="codingkaku@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy','opencv-python','opencv-contrib-python','mediapipe'],
    keywords=[
        'python', 'opencv', 'computer vision', 'cv2', 'image processing', 
        'mediapipe', 'pose detection', 'face mesh', 'hand tracking', 'cvlearn'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video",
    ]
)