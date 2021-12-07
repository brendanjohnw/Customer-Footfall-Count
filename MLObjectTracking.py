import cv2 as cv
import numpy as np
import tracker
import matplotlib.pyplot as plt

capture = cv.VideoCapture('crowd.mp4')
people_detections = []
first_id_dict = {}

tracker = tracker.EuclideanDistTracker()

count = 0
exit_count = 0
enter_count = 0
invalid = 0

class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_file = 'yolov4.cfg'
weights_file = 'yolov4.weights'

net = cv.dnn_DetectionModel(weights_file, config_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((130, 130, 130))
net.setInputSwapRB(True)

while True:
    conf_threshold = 0.45
    # Lower the more sensitive the nms
    nms_threshold = 1
    ret, frame = capture.read()
    window_height, window_width, colour_channels = frame.shape
    frame = frame[round(window_height / 2):window_height, :]
    class_ids, confidence, bounding_box = net.detect(frame, conf_threshold)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    # converting to list from numpy array
    # format of the list is: [x, y, w, h]
    bounding_box = list(bounding_box)

    # converting from numpy array to list
    confidence = list(np.array(confidence).reshape(1, -1)[0])

    # removing numpy wrapper class and converting to primitive type float
    confidence = list(map(float, confidence))
    # To remove overlapping detections
    # note that bounding_box is a list of bounding boxes
    # Based on index values, it will eliminate those that shouldn't be included
    indices = cv.dnn.NMSBoxes(bounding_box, confidence, conf_threshold, nms_threshold)
    print(indices)

    for index in indices:
        index = index[0]  # removes bracket from the list since the list is of length 1
        box = bounding_box[index]
        # print(bounding_box)
        x, y, w, h = box[0], box[1], box[2], box[3]

        people_detections.append([x, y, w, h])
    bounding_box_ids, coordinate_dict = tracker.update(people_detections)

    line_height = (frame.shape[0] - round(frame.shape[0] / 3)) - 50
    cv.line(frame, (0, line_height), (frame.shape[1], line_height), (0, 255, 0), 6)

    for bounding_box_id in bounding_box_ids:
        x, y, w, h, id = bounding_box_id
        centroid_location = (x + round(w / 2), y + round(h / 2))

        color = colors[id % len(colors)]
        color = [i * 255 for i in color]
        id = color[1]

        if id not in first_id_dict.keys():
            first_id_dict[id] = [x, y, False]  # Third parameter tells if the centroid intersected before
        if (line_height - 10 <= centroid_location[1] and centroid_location[1] <= line_height + 10) \
                and (first_id_dict[id][2] == False):
            if (centroid_location[1] < first_id_dict[id][1]):
                count = count + 1
                enter_count = enter_count + 1
                first_id_dict[id][2] = True
            elif (centroid_location[1] > first_id_dict[id][1]):
                count = count + 1
                exit_count = exit_count + 1
                first_id_dict[id][2] = True

            # if(line_height - first_id_dict[id][1] > 0):
            #    count = count - 1
            #    exit_count = exit_count+1
            #    first_id_dict[id][2] = True
            # if (line_height - first_id_dict[id][1] < 0):
            #    count = count + 1
            #    enter_count = enter_count + 1
            #    first_id_dict[id][2] = True

        cv.circle(frame, centroid_location, 5, color, 10)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        bounding_box_ids.remove(bounding_box_id)
        people_detections.remove([x, y, w, h])

    cv.putText(frame, "Net difference: " + str(count), (30, 30), cv.FONT_HERSHEY_PLAIN, 1,
               (255, 0, 0), 2)
    cv.putText(frame, "Number of exits: " + str(exit_count), (60, 60), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv.putText(frame, "Number of entrances: " + str(enter_count), (90, 90), cv.FONT_HERSHEY_PLAIN, 1,
               (255, 0, 0), 2)
    cv.putText(frame, "Errors: " + str(invalid), (120, 120), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    # if len(class_ids) != 0:
    #    for class_id, confidence, bounding_box in zip(class_ids.flatten(), confidence.flatten(), bounding_box):
    #        cv.rectangle(frame, bounding_box, (0, 255, 0), 2)
    #        cv.putText(frame, class_names[class_id].upper(), (bounding_box[0]+10, bounding_box[1]+30),
    #                   cv.FONT_HERSHEY_PLAIN,1,(0, 255, 0), 2)
    cv.imshow("Video", frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        exit()


