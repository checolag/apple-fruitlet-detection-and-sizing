# *Apple Phenotyping using deep learning and 3D depth analysis: An experimental study on fruitlet sizing during early development*

This GitHub repository contains the code for processing bag files recorded by Intel® RealSense™ Depth Cameras. 
The pipeline allows you to analyze RGB and depth frame sequences to extract the most informative data for estimating fruitlet count and diameters, including validation against caliper measurements.

### Step 0: Clone the GitHub repository
```
git clone https://github.com/checolag/apple-fruitlet-detection-and-sizing.git
cd apple-fruitlet-detection-and-sizing
```
### Step 1: Set Up a Python Virtual Environment
- Ensure you are using **Python 3.8.10** for compatibility: `python3 --version`
Create and activate a Python virtual environment by installing the necessary dependencies from the `requirements.txt` file. It might take a while.

```bash
python3 -m venv my_env
source my_env/bin/activate # On Windows, use `my_env\Scripts\activate`
pip install -r requirements.txt
```

### Step 2: Download and Set Up the [Dataset](https://doi.org/10.5281/zenodo.14844598)
- To execute the RGB-D workflow, download the necessary files from Zenodo (`bag_videos.zip` (12.7 GB), `ground_truth_caliper_measurements.csv`)
- Extract `bag_videos.zip`
- Move the `bag_file` folder into `data` the cloned repository directory
- Modify `config.yaml` and `config_post.yaml` to adjust settings as needed.

### Step 3: Run the Executing Scripts for Video Analysis and Anomaly Detection
```bash
./run_pipeline.sh
```
This bash script executes two Python files `main.py` and `main_postprocessing.py`.
1. `main.py` takes as input a configuration file and processes bag files frame by frame. It includes 3 tasks:
- `video_processing`: performs object detection, depth analysis, and size estimation of fruitlets for all videos
- `outlier_detection`: filters the generated CSV file to remove false-positive bounding boxes using anomaly detection
- `image_creation`: generates RGB images with bounding boxes based on the final CSV file

2. `main_postprocessing.py` takes as input the same configuration file and validate the results. It includes 2 tasks:
- `final_dataset_creation`: concatenates all CSV files into a single large DataFrame.
- `validation`: extracts the most informative frames for each video in the final dataset and performs a performance evaluation against the ground truth.


 


