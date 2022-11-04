# Face anonymizer
web app for anonymizing faces in video files

# running the app
the app was tested on Ubuntu 22.04
## environment setup
install python3.10 environment (Ubuntu 22.04):  
`$ sudo apt-get install python3 python3-venv ffmpeg`  
  install venv and its requirements:  
`$ make`
### [optional] GPU acceleration
For GPU inference only Nvidia GPUs are supported, CUDA 11.x should come with default nvidia drivers like `nvidia-driver-510`. Manual cuDNN installation would be necessary:  
`$ sudo apt-get install nvidia-cudnn`  
this step is not strictly necessary. If GPU acceleration is not possible, the `onnxruntime` should fallback to CPU inference.  
For ffmpeg acceleration install:  
`$ sudo apt-get install libffmpeg-nvenc-dev`  
### running the app [CPU and GPU compatible]
to run the app on `http://localhost:8080` enter:  
`$ make run`  
or equivalently (opens the app in default browser):  
`$ make run_and_visit`  
## building a docker image with docker-compose
In order to run the image with GPU support, you need nvidia driver `470.57.02` or newer. Additional install of `nvidia-container-toolkit` is also needed.
Nvidia's [instructions](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html#enabling-the-docker-repository-and-installing-the-nvidia-container-toolkit) to install `nvidia-container-toolkit`:
```bash
$ sudo apt-get install -y docker docker-compose
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```
building and running the container will expose the service under `http://172.25.0.11`
```bash
$ docker-compose -f docker-compose.yaml build
$ docker-compose -f docker-compose.yaml run anonymizer
```
