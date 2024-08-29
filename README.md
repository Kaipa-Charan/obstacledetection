# obstacledetection

# Vision-based-Obstacle-detection

This repository is for the testing obstacle detection 

First clone the ultralytics github

$ git clone https://github.com/roboflow/ultralytics-roboflow 

then enter into the ultralytics folder $ pip install requirements.txt

 then $ pip install ultralytics

**Obstacle Detection Codes Testing**

This Python script leverages several technologies to perform real-time object detection and tracking using a depth camera. It employs the YOLO (You Only Look Once) model for detecting objects, utilizing a RealSense camera for both color and depth imaging. The script incorporates a trapezoidal region of interest (ROI) to determine whether detected objects fall inside or outside this defined area. For tracking purposes, it uses a Kalman filter to predict object movement based on their center coordinates. Background subtraction and Gaussian blur are applied to enhance the detection process. The script draws bounding boxes and labels on detected objects, calculates their distance from the camera using depth data, and visualizes results by displaying color and depth images. It also prints flags indicating the detection status and whether objects are inside or outside the ROI, and it continuously updates these results in real-time until the user presses 'q' to exit.

To test the code (In Jetson Board)

$ conda activate labelImg

$ python intelrealsense.py

and established the connection between Raspberry Pi board and Jetson Board by running codes simultaneously in both jetson and raspberry pi. Before this connect the raspberry pi and jetson with ethernet router cable then follow down steps

In Raspberry pi $ cd /home/raillabs/Desktop/jetsonboardobstacle/

then $ python obstacledata.py

In Jetson Board run  $ python sharing_data_to_masterboard.py

**Running Bash Script for Communication between Raspberrypi and Jetson**

In raspberry pi run the bashscript

$ cd Desktop

$ ./start.sh

Here the communication between the raspberry pi and Jetson board has been established so obstacle data from jetson is sending to the raspberry pi.
