from setuptools import setup, find_packages
import pathlib
import os
here = pathlib.Path(__file__).parent

long_description = (here / "README.md").read_text()

VERSION = '0.3.3'
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
    keywords=['python', 'video', 'stream', 'ErrorCode', 'Tanay', 'opencv', 'opencv-python', 'cv', 'cv2', 'cvlearn'],
    classifiers=[
        "Development Status :: 4 - Beta",  
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)