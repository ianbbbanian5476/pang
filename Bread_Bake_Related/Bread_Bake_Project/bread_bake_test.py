from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from math import dist
import matplotlib
matplotlib.use('agg')

def draw_saturation_histogram_map(image):
    histogram, bin_edges = saturation_histogram(image)

    # Make a random plot...
    fig = plt.figure()
    fig.add_subplot(111)
    plt.stairs(histogram, bin_edges)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    saturation_histogram_map = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Clear all figure
    plt.cla()
    plt.close('all')

    return saturation_histogram_map

def draw_saturation_heat_map(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the Saturation channel
    saturation_channel = hsv_image[:, :, 1]
    saturation_channel = saturation_channel / 255

    # Make a random plot...
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(saturation_channel, cmap='jet', vmin=0, vmax=0.8)
    ax.axis('off')
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    saturation_heat_map = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    saturation_heat_map = cv2.cvtColor(saturation_heat_map, cv2.COLOR_RGB2BGR)

    # Clear all figure
    plt.cla()
    plt.close('all')

    return saturation_heat_map

def saturation_histogram(image):
    # Convert image to HSV (Hue, Saturation, Value)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat_image = hsv_image[:, :, 1]  # Extract the Saturation channel

    # Use the mask to get the saturation of non-transparent pixels
    non_transparent_pixels = sat_image[np.where(gray_image != 0)]
    non_transparent_pixels = non_transparent_pixels / 255

    # Calculate and return the histogram
    histogram, bin_edges = np.histogram(non_transparent_pixels, bins=256, range=(0, 1))
    return histogram, bin_edges

video_name = '20240409_163914.mp4'
video_path = os.path.join('test_videos', video_name)

# Load a model
model = YOLO('segment_models/bread-seg.pt')  # load an official model

output_dir = 'outputs'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_shape = (1280, 720)
fps = 1
output_path = os.path.join(output_dir, f'{video_name[:-4]}_output.mp4')
out = cv2.VideoWriter(output_path, fourcc, fps, video_shape)

cap = cv2.VideoCapture(video_path)
# self.cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start_time = time.time()
last_detect_time = time.time()
is_first_segment = True
last_bounding_box_center = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if time.time() - last_detect_time > 1 / fps:
        # Predict with the model
        results = model(frame)  # predict on an image
        result = results[0]
        # Process result
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Probs object for classification outputs

        objects_len = len(result)
        if objects_len == 0:
            continue

        if is_first_segment:
            # Instance Segmentation
            target_list = []
            for i in range(objects_len):
                bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                mask_contour = np.array(masks.xy[i], dtype=int)
                contour_area = cv2.contourArea(mask_contour)
                target_list.append((bounding_box, mask_contour, contour_area))

            # sort target list by contour area
            for i in range(objects_len):
                for j in range(i + 1, objects_len):
                    if target_list[i][2] < target_list[j][2]:
                        temp = target_list[j]
                        target_list[j] = target_list[i]
                        target_list[i] = temp

            # Get the largest three target
            target_list = target_list[:3]

            # Find the most centered target
            target_bounding_box = None
            target_contour = None
            target_distance = 999999
            frame_center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
            for target_data in target_list:
                bounding_box = target_data[0]
                mask_contour = target_data[1]
                contour_area = target_data[2]
                bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                center_distance = dist(bounding_box_center, frame_center)
                if center_distance < target_distance:
                    target_bounding_box = bounding_box
                    target_contour = mask_contour
                    target_distance = center_distance
            
            last_bounding_box_center = (int((target_bounding_box[0] + target_bounding_box[2]) / 2), int((target_bounding_box[1] + target_bounding_box[3]) / 2))
            is_first_segment = False
        else:
             # Instance Segmentation
            target_bounding_box = None
            target_contour = None
            target_distance = 999999
            target_bounding_box_center = None
            for i in range(objects_len):
                bounding_box = np.array(boxes.xyxy[i].cpu(), dtype=int)
                mask_contour = np.array(masks.xy[i], dtype=int)
                contour_area = cv2.contourArea(mask_contour)
                bounding_box_center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
                center_distance = dist(bounding_box_center, last_bounding_box_center)
                if center_distance < target_distance:
                    target_bounding_box = bounding_box
                    target_contour = mask_contour
                    target_distance = center_distance
                    target_bounding_box_center = bounding_box_center
            last_bounding_box_center = target_bounding_box_center


        black_canvas = np.zeros_like(frame)
        cv2.drawContours(black_canvas, [target_contour], -1, (255, 255, 255), cv2.FILLED) # this gives a binary mask
        black_canvas[np.where(black_canvas != 0)] = frame[np.where(black_canvas != 0)]
        processed_image = black_canvas[target_bounding_box[1]:target_bounding_box[3], target_bounding_box[0]: target_bounding_box[2]]
        bread_image = frame[target_bounding_box[1]:target_bounding_box[3], target_bounding_box[0]: target_bounding_box[2]]

        saturation_heat_map = draw_saturation_heat_map(processed_image)
        saturation_histogram_map = draw_saturation_histogram_map(processed_image)

        saturation_histogram_width = 500
        result_height = processed_image.shape[0] * 2
        result_width = processed_image.shape[1] + saturation_histogram_width
        result_canvas = np.zeros((result_height, result_width, 3))

        saturation_heat_map = cv2.resize(saturation_heat_map, (processed_image.shape[1], processed_image.shape[0]), interpolation=cv2.INTER_AREA) 
        saturation_histogram_map = cv2.resize(saturation_histogram_map, (saturation_histogram_width, result_height), interpolation=cv2.INTER_AREA)    

        result_canvas[:processed_image.shape[0], :processed_image.shape[1], :] = bread_image
        result_canvas[processed_image.shape[0]:, :processed_image.shape[1], :] = saturation_heat_map
        result_canvas[:, processed_image.shape[1]:, :] = saturation_histogram_map

        result_canvas = np.uint8(result_canvas)
        result_canvas = cv2.resize(result_canvas, video_shape, interpolation=cv2.INTER_AREA)
        out.write(result_canvas)
        last_detect_time = time.time()


# Release everything if job is finished
cap.release()
out.release()
print(f'Output Complete! Spend Time: {time.time() - start_time}')
