import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from shapely.geometry import box
from shapely.geometry import Polygon as shapely_poly

# https://towardsdatascience.com/parking-spot-detection-using-mask-rcnn-cb2db74a0ff5
# https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [1]:
            car_boxes.append(box)
        # car_boxes.append(box)
    return np.array(car_boxes)

# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "people1.png")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
# parked_car_boxes = [([129, 286, 248, 481]), ([129, 158, 248, 329]), ([129, 2, 248, 156])]
# parked_car_boxes = np.array(parked_car_boxes)
# free_space_frames = 0
# print(parked_car_boxes)
parked_car_boxes = []
# Load the image file we want to run detection on
image = cv2.imread(IMAGE_DIR)
newImage = image.copy()

rgb_image = image[:, :, ::-1]
results = model.detect([rgb_image], verbose=0)
r = results[0]

car_boxes = get_car_boxes(r['rois'], r['class_ids'])
print("Cars found in frame of video:")
# Draw each box on the frame
for box in car_boxes:
    print("People: ", box)

    y1, x1, y2, x2 = box

    # Draw the box
    cv2.rectangle(newImage, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # if parked_car_boxes is None:
    #     # This is the first frame of video - assume all the cars detected are in parking spaces.
    #     # Save the location of each car as a parking space box and go to the next frame of video.
    #     parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    # else:
    #     # We already know where the parking spaces are. Check if any are currently unoccupied.

    #     # Get where cars are currently located in the frame
    #     car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    #     # See how much those cars overlap with the known parking spaces
    #     overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

    #     # Assume no spaces are free until we find one that is free
    #     free_space = False

    #     # # Loop through each known parking space box
    #     # for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

    #     #     # For this parking space, find the max amount it was covered by any
    #     #     # car that was detected in our image (doesn't really matter which car)
    #     #     max_IoU_overlap = np.max(overlap_areas)

    #     #     # Get the top-left and bottom-right coordinates of the parking area
    #     #     y1, x1, y2, x2 = parking_area

    #     #     # Check if the parking space is occupied by seeing if any car overlaps
    #     #     # it by more than 0.15 using IoU
    #     #     if max_IoU_overlap < 0.15:
    #     #         # Parking space not occupied! Draw a green box around it
    #     #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     #         # Flag that we have seen at least one open space
    #     #         free_space = True
    #     #     else:
    #     #         # Parking space is still occupied - draw a red box around it
    #     #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    #     #     # Write the IoU measurement inside the box
    #     #     font = cv2.FONT_HERSHEY_DUPLEX
    #     #     cv2.putText(image, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

    #     # # If at least one space was free, start counting frames
    #     # # This is so we don't alert based on one frame of a spot being open.
    #     # # This helps prevent the script triggered on one bad detection.
    #     # if free_space:
    #     #     free_space_frames += 1
    #     # else:
    #     #     # If no spots are free, reset the count
    #     #     free_space_frames = 0

cv2.imshow('People Detected', newImage)
cv2.waitKey(0)
# cv2.imshow('Output', image)
# cv2.waitKey(0)