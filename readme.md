# Multiple streams vehicle detection and tracking on an edge device
Vehicle detection and tracking of multiple video streams on a single edge device like jetson xavier.

## Description
This project uses YOLOv5 based object detection algorithm to detect vehicles in the live video feed or recorded videos. Based on the edge device used different object tracking algorithms such as DEEPSORT, SORT and CONVENTIONAL trackers are used. This software is configuration based and 

## Architecture

### Batch-wise inference
This project propses the following model to accomodate detection on more than 2 different streams, since more RAM space and cores are needed. To avoid overloading GPU RAM space, we load the YOLOv5 model onto GPU only once. Following is the flowchart of the architecture used.

![Batch-wise Inference Architecture](/models/batch-arch.png)

Each video stream has a process running and a object detector is also running as a seperate on a GPU. Each frame from the stream is sent to detector process and all the frames are batched together. This batched image is then passed through the detector and the output is then segregated into respective streams and sent back to its respective process using queues. Then the video stream process continues to perform tracking and zone assignment. This process is kept on running in a loop. Although this model has small latency, it can be used for real time applications

## Object-Detection Model
We have a total of four trained YOLOv5 models.
1. YOLOv5n - Faster, low accuracy, Used in ATCS applications
2. YOLOv5s - little slower than YOLOv5n, better accuracy compared to n model, can be used in ATCS
3. YOLOv5m - Good classification accuracy, used in VIDS
4. YOLOv5m - 800x800, Can give more than 95% accuracy

## Object Tracking
Three different object tracking algorithms are present.
1. DEEPSORT - slower, accurate 
2. SORT - fast, reasonably accurate
3. Conventional - faster, not accurate

## Configuring Software
1. Default session should include all the STREAM_IDS.
2. Use Model as REAL, when used on live streams
3. Zone dimensions should be named correctly and should follow the format.
