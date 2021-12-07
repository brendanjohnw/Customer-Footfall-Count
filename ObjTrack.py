import cv2
import numpy as np
import matplotlib.pyplot as plt
import tracker
import time

#Load Video Image
video = cv2.VideoCapture('./video/crowd.mp4')

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

tracker = tracker.EuclideanDistTracker()

pre_obj = {}
down = int(0)
up = int(0)

while True:
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

        cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=2)

        text = f"classID:{id}"

        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
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
                elif pre_obj[str(id)]["center_y"] > centerY and pre_obj[str(id)]["Flags"] == False:
                        up += 1
                        pre_obj[str(id)]["Flags"] = True

            except KeyError:
                pre_obj[str(id)] = {
                    "center_y": centerY,
                    "Flags": False
                }

    fps = 1./(time.time()-start_time)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.putText(img, "Up: " + str(up), (0, 80), 0, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Down: " + str(down), (0,130), 0, 0.5, (0,0,255), 1)

    cv2.imshow("Video", img)
    out.write(img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        exit()

video.release()
out.release()
cv2.destroyAllWindows()

