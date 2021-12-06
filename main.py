import cv2
import cv2 as cv
import math
import numpy as np
import imutils
from tracker import *

# Tracker takes the bounding boxes of the objects
tracker = EuclideanDistTracker()

""" Gathering initial counts from the first frame using the Histogram of Oriented Gradients Algorithm"""
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
capture = cv.VideoCapture("FootfallVideo.mp4")
# Constants

# Returns the direction of the bounding box/person
def magnitude(two_dimensional_array):
    return math.sqrt(sum(number**2 for number in two_dimensional_array))


# Estimated, visible initial count
def initial_count():
    # capturing the first frame
    _, first_frame = capture.read()
    frame_height = first_frame.shape[0]
    frame_width = first_frame.shape[1]
    (first_frame_region, _) = hog.detectMultiScale(first_frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    return len(first_frame_region)


def main():
    capture = cv.VideoCapture("FootfallVideo.mp4")
    count = initial_count()
    bounding_box_list = []

    # Extracts the moving objects from a stable camera
    # the history and varThreshold arguments are in place to ensure minor movement in the still camera
    # does not lead to false positives
    # note: varThreshold argument, higher the value, less false positives
    # background reduction convolution layers
    detector = cv.createBackgroundSubtractorMOG2(history = 100, varThreshold= 50)
    frame_counter = 0
    print(f"Initial number of people: {initial_count()}")
    while True:
        frame_counter = frame_counter+1
        first_id_dict = {}
        ret, frame = capture.read()
        # cropping the video to only show area where customers are enterring and exiting
        window_height, window_width, colour_channels = frame.shape
        cropped_frame = frame[round(window_height/2):window_height, 0:window_width]


        mask = detector.apply(cropped_frame)
        # Cleaning the binary mask such that the shadows are not detected as noise.
        _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
        # Detects the contours found on the binary masked video
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # drawing the contours based on the binary mask image
        # Stores each coordinate of a person
        # Each column represents the a person on the screen
        """Approach: 
            1. draw a boundary line (region of interest ROI) at the lower portion of the video screen
            2. Since the coordinates represent the top left corner of the bounding box, check whether the y is equal to
                to the region of interest (ROI).
            3. To determine whether person is leaving or entering the mall, calculate the magnitude by which each person 
                is moving at
                use the magnitude() function defined above
                - To prevent noise, only if the bounding box of the person intersects with the roi, then only it will be counted as an increment or decrement
            4. Increase counter if person is moving in the -ve direction of y (By convention top left is the origin)
            5. Decrease counter if person is moving in the +ve direction of y"""
        people_detections = []

        for contour in contours:
            contour_area = cv.contourArea((contour))
            if contour_area > 100:
                #cv.drawContours(cropped_frame, [contour], -1, [0, 255 ,0], 2)
                x, y, w, h = cv.boundingRect(contour)
                people_detections.append([x, y, w, h])


        line_height = (cropped_frame.shape[0] - round(cropped_frame.shape[0]/3))-50
        cv.line(cropped_frame, (0, line_height),
                      (cropped_frame.shape[1], line_height), (0, 255, 0), 2)

        # Tracking objects
        # coordinate_dict contains a dictionary of bounding box ids and the coordinates of the bounding boxes
        bounding_box_ids, coordinate_dict = tracker.update(people_detections)
        bounding_box_list.append(coordinate_dict)


        for bounding_box_id in bounding_box_ids:
            x, y, w, h, id = bounding_box_id

            # Placing each detected contour in a bounding box
            cv.putText(cropped_frame, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            rectangle_location = (x,y)
            cv.rectangle(cropped_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            centroid_location = (x+round(w/2),y+round(h/2))

            cv.circle(cropped_frame, centroid_location, 5, 3)
            if id not in first_id_dict.keys():
                first_id_dict[id] = (x, y)
            elif (centroid_location[1]==line_height) and (line_height-first_id_dict[id][1] < 0):
                count = count + 1
            elif (centroid_location[1]==line_height) and (line_height-first_id_dict[id][1] > 0):
                count = count - 1
        cv.putText(cropped_frame, str(count), (30,30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        #cv.imshow("Video", frame)
        #cv.imshow("Mask", mask)

        cv.imshow("Cropped Video", cropped_frame)

        exit_key = cv.waitKey(30)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

main()