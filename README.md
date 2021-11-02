![Title](Panorama.png)

# A Walk in the Black-Box: Deep Convolutional Neural Network Visualization in Virtual Reality - Server

This repository visualizes deep convolutional neural networks (CNNs) in 3D. Take a walk in your CNN and interact with it to get a more intuitive understanding of Deep Learning or to analyze your models. The software provides visualization algorithms like feature visualization to reveal what visual concepts the models have learned. Virtual Reality and desktop mode are available. Currently, CNNs for image classification and segmentation are supported. 

The repository consists of the client and the server part. The client contains the Unity project to display and to interact with the networks. The server is a Python based implementation, handles the networks and provides the client with data. The server can be run in Docker and an appropriate Dockerfile is included in the repository. 

This software is a part of the [DeepVisionVR](https://github.com/Criscraft/DeepVisionVR) project.

## Requirements

Docker and NVIDIA-docker
Use the Dockerfile to build your Docker image.

Get the [Client](https://github.com/Criscraft/DeepVisionVRClient) to use the visualization software.


## Get started with the demo

For the server you will need a Python environment that fulfills the specifications in the provided Dockerfile. We recommend to use Docker. Install [Docker](https://docs.docker.com/engine/install/) and the [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker) extension on your machine. It is advised to run server and client on different machines due to GPU memory limitations. On Windows I had problems to get GPU access in Docker. Therefore, I recommend to use Linux (WSL). In Linux, build the Docker image by executing 
    bash build.sh

Download the data needed for the demo from [HERE] (http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and extract it at DeepVisionVR/DeepLearningServer/Datasets/Caltech 
This folder should contain the 101 folders with images of different classes. Delete the BACKGROUND_Google folder, because it will lead to wrong class names.

All scripts for the demo are located at DeepVisionVR/DeepLearningServer/Projects/Demo/ Run the file start_falcon_server.sh to start the Docker container and the server. Before this, you will have to modify start_falcon_server.sh so that the Docker container. Simply change the left side of the double colon to the absolute path to the DeepLearningServer directory.
    -v /your-absolute-path/DeepLearningServer:/DeepLearningServer

Then execute 
    bash start_falcon_server.sh

Start the [Client](https://github.com/Criscraft/DeepVisionVRClient)