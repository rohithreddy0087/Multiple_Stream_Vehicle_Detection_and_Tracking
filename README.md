# Multiple streams vehicle detection and tracking on an edge device
Vehicle detection and tracking of multiple video streams on a single edge device like jetson xavier.

## Description
This project uses YOLOv5 based object detection algorithm to detect vehicles in the live video feed or recorded videos. Based on the edge device used different object tracking algorithms such as DEEPSORT, SORT and CONVENTIONAL trackers are used. This software is configuration based and acts as a server, that emits a json whenever a unique vehicle is detected in the zone. 


https://user-images.githubusercontent.com/51110057/210043009-204fbea6-12fd-4318-b27e-ab5a259b3056.mp4


## Architecture

### Batch-wise inference
This project propses the following model to accomodate detection on more than 2 different streams, since more RAM space and cores are needed. To avoid overloading GPU RAM space, we load the YOLOv5 model onto GPU only once. Following is the flowchart of the architecture used.

![Batch-wise Inference Architecture](/models/batch-arch.png)

Each video stream has a process running and a object detector is also running as a seperate on a GPU. Each frame from the stream is sent to detector process and all the frames are batched together. This batched image is then passed through the detector and the output is then segregated into respective streams and sent back to its respective process using queues. Then the video stream process continues to perform tracking and zone assignment. This process is kept on running in a loop. Although this model has small latency, it can be used for real time applications

## Object-Detection Model
We have a total of four YOLOv5 models.
1. YOLOv5n - Faster, low accuracy
2. YOLOv5s - little slower than YOLOv5n, better accuracy compared to n model
3. YOLOv5m - Good classification accuracy
4. YOLOv5m - 800x800, Can give more than 95% accuracy<br>
Reference: [YOLOv5](https://github.com/ultralytics/yolov5)


## Object Tracking
Three different object tracking algorithms are used in this project.
### 1. DEEPSORT
Reference: [Link](https://github.com/nwojke/deep_sort)<br>
slower, accurate <br>
Features for the deepsort are given using a siamese network<br>
### 2. SORT
Reference: [Link](https://github.com/abewley/sort)<br>
fast, reasonably accurate<br>
### 3. Conventional
faster, not accurate<br>
Implemented using features like contours, color, IoU etc<br>

## Server
A socket server code is written on different thread, which emits data to connected clients whenever there is a unique vehicle 
detected in the zone. Data between threads is shared using a queue. The server can be connected to multiple clients.

## Zone Assignment
A zone is defined as a polygon and we use the center of the bounding box to identify whether a vehicle is inside this ploygon or not.
Multiple zones of any polygon shape can be defined in the configuration.

## Configuring Software
1. Default session should include all the STREAM_IDS.
2. Use Model as REAL, when used on live streams
3. Zone dimensions should be named correctly and should follow the format.
