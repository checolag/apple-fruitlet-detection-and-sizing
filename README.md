# *Apple Phenotyping using deep learning and 3D depth analysis: An experimental study on fruitlet sizing during early development*

This GitHub repository contains the code for processing bag files recorded by Intel® RealSense™ Depth Cameras. 
The pipeline allows you to analyze RGB and depth frame sequences to extract the most informative data for estimating fruitlet count and diameters, including validation against caliper measurements.

### Step 0: Clone the GitHub repository
```
git clone https://github.com/checolag/apple-fruitlet-detection-and-sizing.git
cd apple-fruitlet-detection-and-sizing
```
### Step 1: Create a Python Local Environment
Start by creating a Python virtual environment and installing the necessary dependencies from the `requirements.txt` file. It might take a while.

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

### Step 2: Download the complete [dataset](https://doi.org/10.5281/zenodo.14844598).
To execute the RGB-D workflow, you need `bag_videos.zip` (12.7 GB) and `ground_truth_caliper_measurements.csv`.
Once extracted the zip file:
- rename it as `data` and move it to the clone directory
- use `config.yaml` and `config_post.yaml` to modify your settings based on your preferences

### Step 3: Run the Executing Scripts for Video Analysis and Anomaly Detection
```bash
./run_pipeline.sh
```
This bash script executes two Python files `main.py` and `main_postprocessing.py`.
`main.py` takes a configuration file and processes bag files frame by frame. It includes 3 tasks:
- `video_processing`: executes detection and depth analysis, including size estimation
- `outlier_detection`: takes as input the csv created in the previous task and perform anomaly detection to remove false positives bounding boxes
- `image_creation`: creates all the images using RGB frames and bounding box annotations from the final csv file

 


