# OpenCV People Tracker and Counter 
## A Customer Footfall Count Project. (Brendan Lee and Seak Jian De)
The following project is a computer vision customer footfall counter that utilizes OpenCV for object tracking and detection. 
The detectiion mechanism was supplemented by pre-trained YoloV4 datasets, which were adapted from Common Objects in Common (COCO).

The main application of the project was to count the in-rate and out-rate of people enterring a particular premise such as a train station
shopping mall or retail store.


## Data Visualization using Tableau (Kong Yuki)
**Initial plan:** Use Tableau to visualize data set generated from OpenCV in a dashboard

**Problem encountered:** Video is too short, little can be predicted based on raw coordinates of all people per frame.

**Alternative:** Use Toronto Station Bikeshare Data (sample data) as an example on how to visualize long term footfall count data

Dashboard for mock data set (OpenCV):
https://public.tableau.com/views/E1_16387797643930/CoordinatesbyFrame?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

Dashboard for OpenCV data set:
https://public.tableau.com/views/OpenCVData/CumulativeCountbySeconds?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

Visualizations created:
- Cumulative Count by Seconds

Dashboard for Toronto Station Bikeshare Data (sample data):
https://public.tableau.com/views/TorontoStationBikeshareData/HourlyTrend?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

Visualizations created:
- Hourly footfall trend (grouped by days of the week)
- Daily footfall trend (grouped by week number, 1- 13 (Q1 of 2018))
- Weekly footfall trend
- Monthly footfall trend

## Limitations

## Future Directions
If more time and computing power is available, more sophisticated deep learning techniques can be implemented to identify additional features captured by the camera.

An example use case would be retail development. Building a new shopping complex is an expensive investment and understanding the demographics in the area would be a key component to ensure maximum business success. Footfall Traffic analysis with AI computer vision can be used to determine the income level of the people walking pass the particular area by looking at the types of clothing, apparel worn by potential patrons to ascertain what kind of shops to have in the complex. 


## Dependencies

OpenCV:https://docs.opencv.org/4.x/d4/db1/tutorial_documentation.html

Numpy: https://numpy.org/doc/stable/

Matplotlib: https://matplotlib.org/stable/api/index

YoloV4: https://github.com/pjreddie/darknet

# References

Toronto Station Bikeshare Data (sample data): https://www.kaggle.com/apexinsideapex/station-footfall-analysis/data
PyImageSearch People tracker and counter: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
Non-Maxima suppression: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
