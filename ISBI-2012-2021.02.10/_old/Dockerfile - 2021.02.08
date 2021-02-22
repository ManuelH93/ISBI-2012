FROM pytorch/pytorch:latest

RUN python3 -m pip install Pillow
RUN python3 -m pip install tifffile
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install opencv-python
RUN python3 -m pip install matplotlib
RUN apt-get install unzip