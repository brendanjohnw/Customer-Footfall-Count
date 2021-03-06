# OpenCV Footfall Counter 
## Object Tracking and Detection (Brendan Lee and Seak Jian De)
The following project is a computer vision customer footfall counter that utilizes OpenCV for object tracking and detection. 
The detectiion mechanism was supplemented by pre-trained YoloV4 datasets, which were adapted from Common Objects in Common (COCO).

The main application of the project was to count the in-rate and out-rate of people enterring a particular premise such as a shopping mall or retail store.


## Features
- Detect people from a live camera feed or a video in MPEG-4.
- Tracking people and assigning with random IDs.
- Count the number of people entering or exiting by analyzing the people moving up or down

## Tech
***Object Detection*** <br />

The initial estimation of the total patrons in the area were counted using the Histogram of Gradients (HOG) detection algorithm for accurate detection of people in the first 
frame of the video. We initially planned to use HOG and explored the use of the Mask-RCNN detection for application throughout the video, but its reliance on GPU lead to poor performance on our devices. Hence, the in-built DNN in OpenCV with YoloV4 weights for detection showed much better performance.

***Object Tracking*** <br />

For object trackng we implement centroid tracking algorithms.
For this algorithms is that we passing in a list of bounding box (x,y)-cordinates for each object detected with our object detection model.
Then we compute euclidean distance between new bounding boxes and existing objects
After that we assume the object will move in between subsequent frames. Thus, we choose to assign the centroid with minimum distance between objects

***Count People*** <br />
We set a line around 3/4 heights then we store the object detected in a dictionary then if the object passed the line. We load the initial Y-Centroid to compare with current frame and determine the person is moving up or down

<img width="1267" alt="Screenshot 2021-12-08 at 2 44 17 PM" src="https://user-images.githubusercontent.com/79955754/145162429-ecadab6f-9314-410f-9fb7-aabac23405d5.png">

Screenshot of the counter in action

## Data Analysis 

**Initial plan:** Use Tableau to visualize data set generated from OpenCV in a dashboard

**Problems encountered:** Video is too short, little can be analyzed based on raw counts of people per second.

**Alternative:** Use Pedestrian Footfall Index in Dublin City Centre (sample data) as an example on how to visualize and analyze long term footfall count data

Dashboard for OpenCV data set:
https://public.tableau.com/views/OpenCVSecondsandCounts/CumulativeCountbySeconds?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

Visualizations created:
- Cumulative Count by Seconds

Dashboard for Pedestrian Footfall Index in Dublin City Centre (sample data):
https://public.tableau.com/views/PedestrianFootfallIndexinDublinCityCentre/Dashboard1?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

Visualizations created:
- Hourly footfall trend (grouped by days of the week)
- Hourly net difference trend (grouped by days of the week)
- Hourly net difference (beginning vs. end of week)
- Daily trend

Findings:
- Day with highest number of entries and exits: Friday
- Weekdays have higher entries and exits than weekends
- Common enter and exit times are 8am - 6pm on Weekdays and 10am - 6pm on Weekends
- Influx of visitor entries around 12pm - 3pm

**Examples of Application in Sunway Group Malls Management:**

**Visitor Footfall Analytics**
- Identify average time visitors spend in the mall
- Identify sales conversion rates relative to total number of visitors
- Identify _specific days of the week and times_ with higher visitor footfall to plan allocation of mall staff

**Zone Analytics**
- Identify which areas' entries and exits have higher footfall, and adjust tenants' rate accordingly
- Visualize common routes taken by visitors to optimize advertisement and outlet placement

## Known Issues
- Object tracking is not efficient
- When the detection algorithm fails to detect a person, the person will be assigned a new ID and the counting would be affected

## Future Directions
If more time and computing power is available, more sophisticated deep learning techniques can be applied to identify additional features of the environment captured by the camera.

An example use case would be retail development. Building a new shopping complex is an expensive investment and understanding the demographics in the area would be a key component to ensure maximum business success. Footfall Traffic analysis with AI computer vision can be used to determine the income level of the people walking pass the particular area by looking at the types of clothing, apparel worn by potential patrons to ascertain what kind of shops to have in the complex. 


## Dependencies

OpenCV: https://docs.opencv.org/4.x/d4/db1/tutorial_documentation.html

Numpy: https://numpy.org/doc/stable/

Matplotlib: https://matplotlib.org/stable/api/index

YoloV4: https://github.com/pjreddie/darknet

Install dependencies with 

```
  pip install -r requirements.txt

```

in your shell.

## Installation
1. clone github repository
2. Install dependencies with pip install -r requirements.txt
3. open ObjTrack.py and change the path to your video
4. To install Yolov4.weights, copy and paste the following into the command line:

```!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights```

5. Run ObjTrack.py. 

# References

Pedestrian Footfall Index in Dublin City Centre (sample data): https://data.world/ie-dublin-city-bus/952a55ab-f222-4249-8086-7bf9fdeba723

PyImageSearch People tracker and counter: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

Non-Maxima suppression: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/

Histogram of Gradients: https://learnopencv.com/histogram-of-oriented-gradients/

Mask R-CNN Object Detection: https://github.com/matterport/Mask_RCNN
