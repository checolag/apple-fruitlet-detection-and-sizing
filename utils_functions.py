"""
Author: Giorgio Checola
Date: 2025-02-11
"""

import os
import shutil
import pyrealsense2 as rs
import numpy as np
import torch
from ultralytics import YOLO
import random
import time
import colorsys
from pathlib import Path
import cv2
from bagpy import bagreader
import glob
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance as scip_dist
from scipy.stats import zscore

# anomaly detection
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import boto3

class DepthCamera():
    """
    Handles operations for Intel RealSense depth cameras and .bag files.
    
    Methods:
        __init__: Initializes the camera or loads a .bag file.
        get_frame: Retrieves aligned color and depth frames, applying filters to depth data.
        get_depth_scale: Returns the depth scaling factor.
        release: Stops the camera pipeline.
    """

    def __init__(self, resolution_width=640, resolution_height=480, bag_file=None):
        """
        Initializes the DepthCamera instance.

        Parameters:
        - resolution_width (int): Width of the video stream (default: 640).
        - resolution_height (int): Height of the video stream (default: 480).
        - bag_file (str): Path to the .bag file to process (default: None).
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        if bag_file:
            config.enable_device_from_file(bag_file)
            profile = self.pipeline.start(config)
            profile.get_device().as_playback().set_real_time(False)
            print(f"Loaded bag file: {bag_file}")
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale =  depth_sensor.get_depth_scale()
        else:
            print("Loading Intel Realsense Camera")
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale =  depth_sensor.get_depth_scale()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            print(f"Camera product: {device_product_line}")

            config.enable_stream(rs.stream.depth, resolution_width,  resolution_height, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, resolution_width,  resolution_height, rs.format.bgr8, 30)
            self.pipeline.start(config)
        
        self.align = rs.align(rs.stream.color)
        self.color_frame = None
        self.depth = None

    def get_frame(self):
        """
        Captures a frame from the camera or .bag file.

        Returns:
        - success (bool): True if the frame was captured successfully.
        - color_image (np.array): Color image frame as a NumPy array.
        - color_frame (rs.frame): Raw color frame.
        - depth_image (np.array): Depth image as a NumPy array.
        - filled_depth (rs.frame): Filtered depth frame.
        - depth_colormap (np.array): Colorized depth map.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        if not depth_frame or not color_frame:
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None, None, None, None
        
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        depth_frame = spatial.process(depth_frame)
        filled_depth = depth_frame.as_depth_frame()

        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, color_image, color_frame, depth_image, filled_depth, depth_colormap
    
    def get_depth_scale(self):
        """
        Retrieves the depth scaling factor.
        "scaling factor" refers to the relation between depth map units and meters; 
        it has nothing to do with the focal length of the camera.
        Depth maps are typically stored in 16-bit unsigned integers at millimeter scale, thus to obtain Z value in meters, the depth map pixels need to be divided by 1000.
        Returns:
        - float: Scaling factor to convert depth values to meters.
        """
        return self.depth_scale

    def release(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()

def extract_specific_frame_from_bag(bag_file_path, target_frame_number):
    """
    Extracts a specific frame from a .bag file.

    Parameters:
    - bag_file_path (str): Path to the .bag file.
    - target_frame_number (int): Frame number to extract.

    Returns:
    - color_image (np.array): Extracted color image.
    - color_raw_frame (rs.frame): Raw color frame.
    - depth_image (np.array): Extracted depth image.
    - depth_raw_frame (rs.frame): Raw depth frame.
    - depth_map (np.array): Colorized depth map.
    """
    start_time = time.time()
    camera = DepthCamera(bag_file=bag_file_path)

    while True:
        ret, color_image, color_raw_frame, depth_image, depth_raw_frame, depth_map = camera.get_frame()
        if not ret:
            break
        if color_raw_frame.get_frame_number() == target_frame_number:
            print(f"Frame {target_frame_number} extracted in {round(time.time()-start_time,2)} sec")
            camera.release()
            return color_image, color_raw_frame, depth_image, depth_raw_frame, depth_map
        
    camera.release()
    print(f"Frame {target_frame_number} not found in the bag file.")
    return None, None, None, None, None
    
def get_distance_point(depth_frame, x, y):
    """
    Retrieves the depth value at a specific pixel.

    Parameters:
    - depth_frame (rs.frame): Depth frame.
    - x (int): Horizontal pixel coordinate.
    - y (int): Vertical pixel coordinate.

    Returns:
    - float: Depth value in meters.
    """
    distance = depth_frame.get_distance(x, y)
    return distance

def depth2PointCloud(depth, rgb, depth_scale, clip_distance_max):
    """
    Converts depth and RGB frames into a point cloud.

    Parameters:
    - depth (rs.frame): Depth frame.
    - rgb (rs.frame): RGB frame.
    - depth_scale (float): Scaling factor to convert depth values to meters.
    - clip_distance_max (float): Maximum distance threshold.

    Returns:
    - np.array: Point cloud with XYZRGB data.
    """
    intrinsics = depth.profile.as_video_stream_profile().intrinsics
    depth = np.asanyarray(depth.get_data()) * depth_scale # 1000 mm => 0.001 meters
    rgb = np.asanyarray(rgb.get_data())
    rows,cols  = depth.shape

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)
    
    valid = (depth > 0) & (depth < clip_distance_max) #remove from the depth image all values above a given value (meters).
    valid = np.ravel(valid)
    z = depth 
    x =  z * (c - intrinsics.ppx) / intrinsics.fx
    y =  z * (r - intrinsics.ppy) / intrinsics.fy

    z = np.ravel(z)[valid]
    x = np.ravel(x)[valid]
    y = np.ravel(y)[valid]
    
    r = np.ravel(rgb[:,:,0])[valid]
    g = np.ravel(rgb[:,:,1])[valid]
    b = np.ravel(rgb[:,:,2])[valid]
    
    pointsxyzrgb = np.dstack((x, y, z, r, g, b))
    pointsxyzrgb = pointsxyzrgb.reshape(-1,6)

    return pointsxyzrgb

# save the pointclouds in ply format
def create_point_cloud_file2(vertices, filename):
    """
    Saves a point cloud to a PLY file.

    Parameters:
    - vertices (np.array): Point cloud data (XYZRGB format).
    - filename (str): Output PLY file path.
    """
    ply_header = '''ply
  format ascii 1.0
  element vertex %(vert_num)d
  property float x
  property float y
  property float z
  property uchar red
  property uchar green
  property uchar blue
  end_header
  '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')


# data handling
def divide_files_into_subfolders(input_dir):
    """
    Divide JPG, TXT, and XML files into three separate subfolders.

    Parameters:
    - input_dir (str): Path to the directory containing the files to be organized.

    The function creates three subfolders within the `input_dir`:
    - jpg_files: Contains all .jpg files.
    - txt_files: Contains all .txt files.
    - xml_files: Contains all .xml files.

    Example usage:
    divide_files_into_subfolders("/home/checolag/Downloads/fem/07-05-2024-093542")
    """
    # Define subfolder names
    subfolders = {
        'jpg': 'jpg_files',
        'txt': 'txt_files',
        'xml': 'xml_files'
    }

    # Create subfolders if they don't exist
    for subfolder in subfolders.values():
        os.makedirs(os.path.join(input_dir, subfolder), exist_ok=True)

    # Iterate through files in the input directory
    for filename in os.listdir(input_dir):
        # Skip directories
        if os.path.isdir(os.path.join(input_dir, filename)):
            continue
        
        # Get the file extension
        file_ext = filename.split('.')[-1].lower()
        
        # Move the file to the corresponding subfolder
        if file_ext in subfolders:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(input_dir, subfolders[file_ext], filename)
            shutil.move(src_path, dst_path)
            print(f"Moved: {filename} -> {subfolders[file_ext]}")

def frame_predict(model, frame, device, classes=None, imgsz=640, conf=0.3, obb=False, depth_filter=False):
    """
    Performs object detection on a single frame.

    Parameters:
    - model (YOLO): YOLO object detection model.
    - frame (np.array): Input image frame.
    - device (str): Device to use for inference ('cpu' or 'cuda').
    - classes (list[int], optional): Filter specific class IDs (default: None).
    - imgsz (int): Image size for inference (default: 640).
    - conf (float): Confidence threshold for detection (default: 0.3).
    - depth_filter (bool): Whether to filter detections based on depth (default: False).

    Returns:
    - bboxes (np.array): Bounding boxes of detected objects.
    - class_ids (np.array): Class IDs of detected objects.
    - scores (np.array): Confidence scores for detections.
    """
    
    if depth_filter:
        pass
    results = model.predict(source=frame, save=False, save_txt=False,
                            imgsz=imgsz,
                            conf=conf,
                            classes=classes,
                            device=device)
    
    result = results[0]
    if obb:
        bboxes = np.array(result.obb.xyxyxyxy.cpu(), dtype="int")
        class_ids = np.array(result.obb.cls.cpu(), dtype="int")
        scores = np.array(result.obb.conf.cpu(), dtype="float").round(2)
    else:
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

    return bboxes, class_ids, scores

def get_id_by_class_name(class_name, model_classes):
    for i, name in enumerate(model_classes.values()):
        if name.lower() == class_name.lower():
            return i
    return -1

def random_colors(N, bright=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]


class ObjectDetection:
    def __init__(self, weights_path):
        # Load Network
        self.weights_path = weights_path
        self.colors = self.random_colors(800)

        self.model = YOLO(self.weights_path)
        self.classes = self.model.names
        # print(self.classes) 
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")
        print(self.device)

    def get_id_by_class_name(self, class_name):
        for i, name in enumerate(self.classes.values()):
            if name.lower() == class_name.lower():
                return i
        return -1
    

    def random_colors(self, N, bright=False):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    
    def detect(self, frame, imgsz=640, conf=0.25, nms=True, classes=None, device=None):
     
        filter_classes = classes if classes else None
        device = device if device else self.device
       
        results = self.model.predict(source=frame, save=False, save_txt=False,
                                     imgsz=imgsz,
                                     conf=conf,
                                     nms=nms,
                                     classes=filter_classes,
                                     half=False,
                                     device=device)  # save predictions as labels

        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, scores

def create_video(bag_path):
    output_folder = bag_path.parent
    color_files = list(Path(os.path.join(output_folder, "color")).glob("*.png"))
    color_files.sort()
    depth_files = list(Path(os.path.join(output_folder, "depth_aligned_colorized")).glob("*.png"))
    depth_files.sort()
    
    first_color_image = cv2.imread(str(color_files[0]))
    height, width, _ = first_color_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video (other options: 'XVID', 'MJPG', 'DIVX', etc.)
    video_out = cv2.VideoWriter(os.path.join(output_folder, bag_path.stem + ".mp4"), fourcc, 30.0, (2*width, height))
    
    # Loop through the image files and write frames to the video
    for color_file, depth_file in zip(color_files, depth_files):
        color_frame = cv2.imread(str(color_file))
        depth_frame = cv2.imread(str(depth_file))
        stacked_frame = cv2.hconcat([color_frame, depth_frame])
        video_out.write(stacked_frame)
    
    # Release the VideoWriter
    video_out.release()
    
def bag_frames(bag, delete=True):
  b = bagreader(bag)
  data = b.topic_table
  print("Color Image Information:")
  print(data[data.Types == "sensor_msgs/Image"][["Topics", "Message Count", "Frequency"]].iloc[1])
  print("Depth Image Information:")
  print(data[data.Types == "sensor_msgs/Image"][["Topics", "Message Count", "Frequency"]].iloc[0])
  bag_path = Path(bag)

  if delete:
    shutil.rmtree(bag_path.parent / bag_path.stem)
    
  return data[data.Types == "sensor_msgs/Image"]

def pipeline(config, bag_file, output_dir):
    output_csv = os.path.join(output_dir, Path(bag_file).stem + '.csv')
    model = YOLO(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = DepthCamera(bag_file=bag_file)
    bag_data = bag_frames(bag_file)

    cur_frame_number=-1
    df_columns = ['frame_number', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'conf_score', 'idx', 'x_c', 'y_c', 'z_c', 'centroid_flag', 'dim1_mm', 'dim2_mm']
    df = pd.DataFrame(columns=df_columns)

    while True:
        ret, color_image, color_raw_frame, depth_image_mm, depth_raw_frame, _ = camera.get_frame()
        depth_image_m = depth_image_mm * camera.depth_scale
        if not ret:
            break

        if color_raw_frame.get_frame_number() == cur_frame_number:
            continue
        if cur_frame_number < color_raw_frame.get_frame_number():
            cur_frame_number = color_raw_frame.get_frame_number()

            bboxes, class_ids, scores = frame_predict(model, color_image, device=device, classes=[0], imgsz=config["image_size"], conf=config["conf_threshold"], obb=config["obb"])
            depth_intrinsics = depth_raw_frame.profile.as_video_stream_profile().intrinsics   # color_raw_frame is the same
            
            center_3d_points = [] 
            frames_data = []
            centroid_flags = np.zeros(len(bboxes))

            distances = []
            if len(bboxes) > 0:
                for idx, (bbox, score) in enumerate(zip(bboxes, scores)):
                    if config["obb"]:
                        x1, y1, x2, y2, x3, y3, x4, y4 = bbox.flatten()
                        cx, cy = (x1 + x2 + x3 + x4) // 4, (y1 + y2 + y3 + y4) // 4
                        mask = np.zeros_like(depth_image_m, dtype=np.uint8)
                        cv2.fillPoly(mask, [bbox], 1)
                        points_inside_obb = depth_image_m[mask == 1]
                        filtered_data = points_inside_obb[
                            (points_inside_obb > 0) &
                            (points_inside_obb < config["background_threshold"]) &
                            (~np.isnan(points_inside_obb))
                        ]
                    else:
                        x1, y1, x2, y2 = bbox
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cropped_depth_data = depth_image_m[y1:y2, x1:x2]
                        filtered_data = cropped_depth_data[(cropped_depth_data > 0) &
                                                           (cropped_depth_data < config["background_threshold"]) &
                                                           (~np.isnan(cropped_depth_data))] # remove 0,0,0 points and detections over 1 m
                    
                    if filtered_data.size == 0:
                        distance = np.nan
                    else:
                        distance = np.median(filtered_data)
                    distances.append(distance)
                
                    center_3d_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], distance)
                    center_3d_point_cm = [coord * 100 for coord in center_3d_point]
                    center_3d_points.append(center_3d_point_cm)

                    frame_data = {key: value for key, value in {
                        "frame_number": cur_frame_number,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x3": x3 if 'x3' in locals() else None,
                        "y3": y3 if 'y3' in locals() else None,
                        "x4": x4 if 'x4' in locals() else None,
                        "y4": y4 if 'y4' in locals() else None,
                        "conf_score": score,
                        "idx": idx,
                        "x_c": center_3d_point_cm[0],
                        "y_c": center_3d_point_cm[1],
                        "z_c": center_3d_point_cm[2],
                        "centroid_flag": 0,
                        "dim1_mm": np.nan,
                        "dim2_mm": np.nan,
                    }.items() if value is not None}

                    frames_data.append(frame_data)
                    
                center_3d_points_array = np.array(center_3d_points)
                non_nan_indices = ~np.isnan(center_3d_points_array).any(axis=1)
                non_nan_3d_points_array = center_3d_points_array[non_nan_indices]

                non_nan_original_indices = np.where(non_nan_indices)[0] # indices numbers (not True or False)
                if config["zscore"]:
                    if np.all(np.isnan(non_nan_3d_points_array)):
                        non_error_indices = []
                        center_3d_points_filtered = non_nan_3d_points_array[non_error_indices]
                    else:
                        median_z = np.median(non_nan_3d_points_array[:, 2])
                        mad = np.median(np.abs(non_nan_3d_points_array[:, 2] - median_z))
                        if mad == 0:
                            non_error_indices = np.ones(non_nan_3d_points_array.shape[0], dtype=bool)
                        else:
                            modified_z_scores = 0.6745 * (non_nan_3d_points_array[:, 2] - median_z) / mad
                            non_error_indices = np.abs(modified_z_scores) <= config["zscore_threshold"]
                        center_3d_points_filtered = non_nan_3d_points_array[non_error_indices]
                        original_indices = non_nan_original_indices[non_error_indices] #non_zero_original_indices[non_error_indices]
                        original_indices_for_closest_corymb = []
                else:
                    center_3d_points_filtered = non_nan_3d_points_array
                    original_indices = non_nan_original_indices
                
                if len(center_3d_points_filtered) == 0:
                    print("Not enough points to perform clustering. Go to the next frame")
                    original_indices_for_closest_corymb = []
                elif len(center_3d_points_filtered) == 1:
                    print("Only one fruitlet detected --> main cluster")
                    closest_corymb_centroid = center_3d_points_filtered[0]
                    original_indices_for_closest_corymb = original_indices  # Only one index
                    centroid_flags[original_indices_for_closest_corymb[0]] = 1
                else:
                    pairwise_distances = scip_dist.pdist(center_3d_points_filtered, metric='euclidean') # other metrics?
                    Z = linkage(pairwise_distances, method=config["linkage_type"]) # ward # you can try "complete", "single" ...
                    labels = fcluster(Z, t=config["cluster_threshold"], criterion='distance')
                    
                    centroids = {}
                    unique_labels = np.unique(labels)
                    for label in unique_labels:
                        cluster_points = center_3d_points_filtered[labels == label]
                        centroid = np.mean(cluster_points, axis=0)
                        centroids[label] = centroid
            
                    closest_corymb_centroid = None
                    min_distance = float('inf') 
            
                    for label, centroid in centroids.items():
                        distance_centroid = np.linalg.norm(centroid)
                        if distance_centroid < min_distance:
                            min_distance = distance_centroid
                            closest_corymb_centroid = (label, centroid)

                    filtered_indices = np.where(labels == closest_corymb_centroid[0])[0]
                    original_indices_for_closest_corymb = original_indices[filtered_indices]
                    for idx in original_indices_for_closest_corymb:
                        centroid_flags[idx] = 1

                dim1_mm_list = []
                dim2_mm_list = []
            
                for bbox, class_id, score, distance in zip(bboxes[original_indices_for_closest_corymb],
                                                        class_ids[original_indices_for_closest_corymb], 
                                                        scores[original_indices_for_closest_corymb],
                                                        [distances[idx] for idx in original_indices_for_closest_corymb]):     
                    if config['obb']:
                        x1, y1, x2, y2, x3, y3, x4, y4 = bbox.flatten()
                        cx, cy = (x1 + x2 + x3 + x4) // 4, (y1 + y2 + y3 + y4) // 4

                        mid1_x, mid1_y = (x1 + x4) // 2, (y1 + y4) // 2  # Midpoint of edge (x1, y1) <-> (x4, y4)
                        mid2_x, mid2_y = (x2 + x3) // 2, (y2 + y3) // 2  # Midpoint of edge (x2, y2) <-> (x3, y3)
                        mid3_x, mid3_y = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint of edge (x1, y1) <-> (x2, y2)
                        mid4_x, mid4_y = (x3 + x4) // 2, (y3 + y4) // 2  # Midpoint of edge (x3, y3) <-> (x4, y4)

                        # Deproject the midpoints into 3D points
                        point1 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mid1_x, mid1_y], distance)
                        point2 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mid2_x, mid2_y], distance)
                        point3 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mid3_x, mid3_y], distance)
                        point4 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mid4_x, mid4_y], distance)

                        # Compute the width and height in millimeters
                        dim1_mm = np.linalg.norm(np.array(point2) - np.array(point1)) * 1000
                        dim2_mm = np.linalg.norm(np.array(point4) - np.array(point3)) * 1000

                    else:
                        x1, y1, x2, y2 = bbox
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        point1x = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, cy], distance)
                        point2x = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, cy], distance)
                        point1y = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y1, cx], distance)
                        point2y = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y2, cx], distance)

                        dim1_mm = np.linalg.norm(np.array(point2x) - np.array(point1x)) * 1000
                        dim2_mm = np.linalg.norm(np.array(point2y) - np.array(point1y)) * 1000
                    
                    dim1_mm_list.append(dim1_mm)
                    dim2_mm_list.append(dim2_mm)
                
                k=0
                for i, frame_data in enumerate(frames_data):
                    if centroid_flags[i] == 1:
                        frame_data['centroid_flag'] = 1
                        frame_data['dim1_mm'] = dim1_mm_list[k]
                        frame_data['dim2_mm'] = dim2_mm_list[k]
                        k+=1
                frame_df = pd.DataFrame(frames_data, columns=df_columns)
                df = pd.concat([df, frame_df], ignore_index=True)

        else:
            break

    df.to_csv(output_csv, index=False)
    print("CSV create!")
    print(f'Number of color and depth frame: {bag_data[["Topics", "Message Count", "Frequency"]].iloc[1]["Message Count"]}, {bag_data[["Topics", "Message Count", "Frequency"]].iloc[0]["Message Count"]}')
  


def anomaly_detection(config, csv_file):
    outlier_csv = os.path.splitext(csv_file)[0] + "_outliers.csv" 
    loss_image = os.path.splitext(csv_file)[0] + "_outliers.png"

    def min_measure(row):
        return min(row['dim1_mm'], row['dim2_mm'])
    
    df_predicted = pd.read_csv(csv_file, index_col=None)
    df_predicted_min = df_predicted.copy()
    df_predicted_min['min_measure'] = df_predicted_min.apply(min_measure, axis=1)
    df_predicted_min = df_predicted_min[df_predicted_min["centroid_flag"] == 1]
    df_centroid = df_predicted_min.copy()
    df_centroid['count'] = df_centroid.groupby('frame_number')['frame_number'].transform('count')
    df_centroid = df_centroid[['x_c', 'y_c', 'z_c', 'conf_score', 'min_measure', 'frame_number', 'count']]

    x_train = df_centroid.values
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = torch.FloatTensor(x_train)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    class EnhancedVAE(nn.Module):
        def __init__(self, z):
            super(EnhancedVAE, self).__init__()
            self.z = z
            self.fc1 = nn.Linear(len(df_centroid.columns), 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc21 = nn.Linear(64, z)  # Mean of latent space
            self.fc22 = nn.Linear(64, z)  # Log variance of latent space

            self.fc3 = nn.Linear(z, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, 128)
            self.bn4 = nn.BatchNorm1d(128)
            self.fc5 = nn.Linear(128, len(df_centroid.columns))

            # self.dropout = nn.Dropout(0.3)  # Dropout layer to reduce overfitting

        def encode(self, x):
            h1 = torch.relu(self.bn1(self.fc1(x)))
            h2 = torch.relu(self.bn2(self.fc2(h1)))
            return self.fc21(h2), self.fc22(h2)  # Return mean and log variance

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h3 = torch.relu(self.bn3(self.fc3(z)))
            h4 = torch.relu(self.bn4(self.fc4(h3)))
            return torch.sigmoid(self.fc5(h4))  # Output must be between 0 and 1

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

        def loss_function(self, recon_x, x, mu, logvar, beta):
            BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + beta * KLD
        
    Z = config["z"]
    enhanced_vae = EnhancedVAE(z=Z)
    optimizer = optim.Adam(enhanced_vae.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = config["vae_epochs"]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    loss_values = []
    start_time = time.time()
    for epoch in range(num_epochs):
        enhanced_vae.train()
        optimizer.zero_grad()
        recon_batch, mu, logvar = enhanced_vae(x_train)
        beta = 1
        loss = enhanced_vae.loss_function(recon_batch, x_train, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_values.append(loss.item())

        # Print the learning rate to track changes
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {current_lr}')

    with torch.no_grad():
        enhanced_vae.eval()
        reconstructed, mu, logvar = enhanced_vae(x_train)
        mse = torch.mean((x_train - reconstructed) ** 2, dim=1).numpy()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total Training Time: {total_time:.2f} seconds')

    threshold = np.percentile(mse, config["percentile_threshold"])
    df_centroid['outlier'] = (mse > threshold).astype(int)
    df_centroid[df_centroid["outlier"]==1].to_csv(outlier_csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(loss_image, bbox_inches='tight', dpi=500)  # Save the loss plot
    plt.close()

def bag_images(bag, csv_file, output_image):
    os.makedirs(output_image, exist_ok=True)
    df = pd.read_csv(csv_file, index_col=None)
    frame_numbers = df["frame_number"].unique()

    for target_frame_number in frame_numbers:
        color_image, color_raw_frame, depth_image, depth_raw_frame, depth_map = extract_specific_frame_from_bag(
            bag, target_frame_number
        )
        if color_image is None:
            continue
        df_frame = df[df["frame_number"] == target_frame_number]
        modified_image = color_image.copy()
        for _, row in df_frame.iterrows():
            x1, y1, x2, y2, x3, y3, x4, y4 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']), int(row['x3']), int(row['y3']), int(row['x4']), int(row['y4'])
            cx, cy = (x1 + x2 + x3 + x4) // 4, (y1 + y2 + y3 + y4) // 4
            if row['centroid_flag'] == 1:
                color = (194, 91, 91)  # Highlighted color for centroid flag
                
                mid1_x, mid1_y = (x1 + x4) // 2, (y1 + y4) // 2  # Midpoint of edge (x1, y1) <-> (x4, y4)
                mid2_x, mid2_y = (x2 + x3) // 2, (y2 + y3) // 2  # Midpoint of edge (x2, y2) <-> (x3, y3)
                mid3_x, mid3_y = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint of edge (x1, y1) <-> (x2, y2)
                mid4_x, mid4_y = (x3 + x4) // 2, (y3 + y4) // 2  # Midpoint of edge (x3, y3) <-> (x4, y4)

                # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(modified_image, (mid1_x, mid1_y), (mid2_x, mid2_y), (255, 255, 255), 2)
                cv2.line(modified_image, (mid3_x, mid3_y), (mid4_x, mid4_y), (255, 255, 255), 2)
                # cv2.line(modified_image, (x1, cy), (x2, cy), (255, 255, 255), 2)
                # cv2.line(modified_image, (cx, y1), (cx, y2), (255, 255, 255), 2)
                cv2.putText(modified_image, f"x={row['dim1_mm']:.1f} mm", (cx - 40, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(modified_image, f"y={row['dim2_mm']:.1f} mm", (cx - 40, cy - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                color = (247, 247, 128)
            
            # cv2.rectangle(modified_image, (x1, y1), (x2, y2), color, 2)
            obb_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
            cv2.polylines(modified_image, [obb_points], isClosed=True, color=color, thickness=2)
            cv2.putText(modified_image, f"{int(row['idx'])}", (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(modified_image, f"{row['conf_score']:.2f}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
            
            # cv2.putText(modified_image, f"{int(row['idx'])}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # cv2.putText(modified_image, f"{row['conf_score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
            
        output_image_path = os.path.join(output_image, f'{target_frame_number}.jpg')
        cv2.imwrite(output_image_path, cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)) #cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
