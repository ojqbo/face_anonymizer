# Face anonymizer
WebApp for anonymizing faces in video files  

https://user-images.githubusercontent.com/61695389/201243782-3b5490a1-bc7c-428a-95b3-63c89182a3fe.mp4

## About
This app helps you to anonymize faces present video file. The app lets you to seek the video while previewing the results, configure the anonymization details, and export the video (with sound) to the same container as the file you provided (mp4, webm, etc.). This app can process video containers not supported by browsers (e.x. mkv), in such case, the preview capability will not work.
In preview mode, only a few next frames are processed, starting from current pointer, to save compute resources. If you decide that the previewed result is acceptable, you can export video with your anonymization settings by clicking `Apply and Download` button. The app will then trigger download, label all frames, and encode the result on the fly into the downloaded file. 

## Some technical details
The result will always be a video of constant frame rate (CFR). 
Initially the app allowed you to seek and preview the video without uploading the video in the first place, the required file segments were pulled from you as needed. Unfortunately nearly all videos I encountered while developing the app were of variable frame rate, and for long videos it led to unacceptable drifts and other problems.
Now the app uses [dmlc/decord](https://github.com/dmlc/decord) which analyzes the video in the first seek pass before actual reading of frames. As a benefit, the reading process is faster. If this initial seek pass could be done client side (in browser) then the initial download would no longer be necessary - this is possible but difficult.

# Running the app locally
The app was tested only on python3.10 and Ubuntu 22.04.
## Environment setup
Install python3.10 and ffmpeg packages (Ubuntu 22.04):  
```bash
$ sudo apt-get install python3 python3-venv python3-pip ffmpeg
$ pip3 install poetry
```  
create venv and install requirements:  
`$ make .venv`
### [optional] GPU acceleration
For GPU inference only Nvidia GPUs are supported, CUDA 11.x should come with default nvidia drivers like `nvidia-driver-510`. Manual cuDNN installation would be necessary:  
`$ sudo apt-get install nvidia-cudnn`  
This step is not strictly necessary. If GPU acceleration is not possible, the `onnxruntime` should fallback to CPU inference.  
For ffmpeg acceleration install:  
`$ sudo apt-get install libffmpeg-nvenc-dev`  
### Running the app [CPU and GPU compatible]
to run the app on `http://localhost:8080` enter:  
`$ make`  
or equivalently (opens the app in default browser):  
`$ make run_and_visit`  
# Building a docker image with docker-compose
In order to run the image with GPU support, you need nvidia driver `470.57.02` or newer as the container use CUDA 11.4 (see [compatibility chart](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package)). Additional install of `nvidia-container-toolkit` is also needed.
Nvidia's [instructions](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html#enabling-the-docker-repository-and-installing-the-nvidia-container-toolkit) to install `nvidia-container-toolkit`:
```bash
$ sudo apt-get install -y docker docker-compose  # presumably already done

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```
building and running the container will expose the service under `http://172.25.0.2`
```bash
$ docker-compose -f docker-compose.yaml build
$ docker-compose -f docker-compose.yaml run anonymizer
```
# Credits
- Idea for the app builds on [ORB-HD/deface](https://github.com/ORB-HD/deface). Compared to `deface`, this app contribution is the seek capability and ability to export video with sound. This app lacks support for image processing, which `deface` supports (MIT license),
- `centerface` model is based on [Star-Clouds/centerface](https://github.com/Star-Clouds/centerface) (MIT license),
- HTML template and website design credits go to [HTML5 UP](https://html5up.net/fractal) (CCA 3.0 license),
- [dmlc/decord](https://github.com/dmlc/decord) - video reading (Apache-2.0 license),
- [OpenCV](https://opencv.org/) - image manipulation server and client side (Apache-2.0 license),
- [ONNX](https://onnx.ai/) - inference (Apache-2.0 license),
- [PyTorch](https://pytorch.org/) - model import, edits, and export back to `.onnx` format (modified BSD license),
- [numpy](https://numpy.org) - label post processing (BSD-3-Clause license),
- [aiohttp](https://docs.aiohttp.org/en/stable/web.html) - asyncio HTTP server (Apache-2.0 license).
- [docker](https://www.docker.com/) - build and runtime [moby/moby](https://github.com/moby/moby), [docker/cli](https://github.com/docker/cli) (Apache-2.0 license).
