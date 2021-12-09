import cv2
import math

import datetime
import numpy as np
import matplotlib.pyplot as plt
import tracker
import time



#Load Video Image
video = cv2.VideoCapture('./video/crowd.mp4')
def HOG_initial_count():
    # capturing the first frame
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    _, first_frame = video.read()
    (first_frame_region, _) = hog.detectMultiScale(first_frame, winStride=(3, 3), padding=(8, 8), scale=1.05)
    return len(first_frame_region)
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(video.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./video/results.avi', codec, vid_fps, (vid_width, vid_height))

#Model file
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_file = 'yolov4.cfg'
weights_file = 'yolov4.weights'

#Setup model
net = cv2.dnn_DetectionModel(weights_file, config_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((130, 130, 130))
net.setInputSwapRB(True)

conf_threshold = 0.40
nms_threshold = 0.40
seconds = 0
exact_seconds = 0
init_time = 0
total_frames = 0
total_fps = 0

seconds_person_counts = {}
seconds_person_enter = {}
seconds_person_exit = {}


# Instantiating the tracker
tracker = tracker.EuclideanDistTracker()

# For generating approximated data
def time_estimator(average):
    runtime_seconds = time.time() - init_time
    return math.floor(int(runtime_seconds) / average)

pre_obj = {}
down = int(0)
up = int(0)
total = int(0)
init_time = time.time()
total_counts = HOG_initial_count()
try:
    while True:
        total_frames += 1
        _, img = video.read()

        if img is None:
            print("Completed")
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_time = time.time()


        # Model
        class_ids, confidence, bounding_box = net.detect(img_in, conf_threshold, nms_threshold)

        people_detections = []
        for (classId, score, box) in zip(class_ids, confidence, bounding_box):
            if int(classId) == 0:
                x, y, w, h = box[0], box[1], box[2], box[3]
                people_detections.append([x, y, w, h])

        bounding_box_ids, coordinate_dict = tracker.update(people_detections)

        height, width, _ = img.shape
        cv2.line(img, (0, int(4*height/6+height/20)), (width, int(4*height/6+height/20)), (0, 255, 0), thickness=2)
        cv2.line(img, (0, int(4*height/6-height/20)), (width, int(4*height/6-height/20)), (0, 255, 0), thickness=2)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

        for bounding_box_id in bounding_box_ids:
            x, y, w, h, id = bounding_box_id
            centroid_location = (x + round(w / 2), y + round(h / 2))
            centerY = y + round(h / 2)

            color = colors[int(id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.circle(img, centroid_location, 3, (0,255,0), 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=2)

            text = f"classID:{id}"

            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1,
                        color=color, thickness=2)



            if centerY <= (4*height/6+height/20) and centerY >= int(4*height/6-height/20):
                if (len(pre_obj) == 0):
                    pre_obj[str(id)] = {
                        "center_y": centerY,
                        "Flags": False
                    }

                try:
                    pre_obj[str(id)]

                    if pre_obj[str(id)]["center_y"] < centerY and pre_obj[str(id)]["Flags"] == False:
                            down += 1
                            pre_obj[str(id)]["Flags"] = True
                            total_counts -= 1
                    elif pre_obj[str(id)]["center_y"] > centerY and pre_obj[str(id)]["Flags"] == False:
                            up += 1
                            pre_obj[str(id)]["Flags"] = True
                            total_counts += 1



                except KeyError:
                    pre_obj[str(id)] = {
                        "center_y": centerY,
                        "Flags": False
                    }

        fps = 1. / (time.time() - start_time)
        total_fps +=fps
        average_frames = total_fps/total_frames
        exact_seconds = time_estimator(average_frames)

        seconds_person_counts[exact_seconds] = total_counts
        print(exact_seconds)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
        cv2.putText(img, "People Entering: " + str(up), (0, 60), 0, 0.5, (0, 0, 255), 1)
        cv2.putText(img, "People Exiting: " + str(down), (0,90), 0, 0.5, (0,0,255), 1)
        cv2.putText(img, "Total People in premise: " + str(total_counts),(0, 120), 0, 0.5, (0,0,255), 1)

        cv2.imshow("Video", img)
        out.write(img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit()

    import csv

    data = seconds_person_counts.items()
    data_1 = seconds_person_enter.items()
    data_2 = seconds_person_exit.items()

    with open('seconds_and_counts.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Second', 'Cumulative Count'])
        for row in data:
            csv_out.writerow(row)
except:




    video.release()
    cv2.destroyAllWindows()

