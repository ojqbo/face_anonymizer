FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y libffmpeg-nvenc-dev ffmpeg
RUN apt-get install -y curl

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10
RUN apt-get install -y python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

COPY requirements.txt ./
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt  && rm requirements.txt

WORKDIR /app
EXPOSE 80
