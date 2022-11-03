# Face anonymizer
web app for anonymizing faces in video files

# running the app
the app was tested on Ubuntu 22.04
## environment setup
install python environment:  
`$ sudo apt install python3 python3-venv`  
  install venv and its requirements:  
`$ make`
### [optional] GPU acceleration
For GPU inference only Nvidia GPUs are supported, CUDA 11.x should come with default nvidia drivers like `nvidia-driver-510`. Manual cuDNN installation would be necessary:  
`$ sudo apt install nvidia-cudnn`  
this step is not strictly necessary. If GPU acceleration is not possible, the `onnxruntime` should fallback to CPU inference.  
For ffmpeg acceleration install:  
`$ sudo apt install libffmpeg-nvenc-dev`  
## running the app
to run the app on `http://localhost:8080` enter:  
`$ make run`  
or equivalently (opens the app in default browser):  
`$ make run_and_visit`  