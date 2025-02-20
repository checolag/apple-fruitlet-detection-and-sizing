# *Apple Phenotyping using deep learning and 3D depth analysis: An experimental study on fruitlet sizing during early development*

This GitHub repository contains the code for processing bag files recorded by Intel® RealSense™ Depth Cameras. 
The pipeline allows you to analyze RGB and depth frame sequences to extract the most informative data for estimating fruitlet count and diameters, including validation against caliper measurements.

### Step 0: Clone the GitHub repository
```
git clone https://github.com/checolag/apple-fruitlet-detection-and-sizing.git
cd apple-fruitlet-detection-and-sizing
```
### Step 1: Set Up a Python Virtual Environment
Create and activate a Python virtual environment by installing the necessary dependencies from the `requirements.txt` file. It might take a while.

```bash
python3 -m venv my_env
source my_env/bin/activate # On Windows, use `my_env\Scripts\activate`
pip install -r requirements.txt
```

### Step 2: Download and Set Up the [Dataset](https://doi.org/10.5281/zenodo.14844598)
- To execute the RGB-D workflow, download the necessary files from Zenodo (`bag_videos.zip` (12.7 GB), `ground_truth_caliper_measurements.csv`)
- Extract `bag_videos.zip`
- Rename the extracted folder to `data`
- Move the `data` folder into the cloned repository directory
- Modify `config.yaml` and `config_post.yaml` to adjust settings as needed.

### Step 3: Run the Executing Scripts for Video Analysis and Anomaly Detection
```bash
./run_pipeline.sh
```
This bash script executes two Python files `main.py` and `main_postprocessing.py`.
`main.py` takes a configuration file and processes bag files frame by frame. It includes 3 tasks:
- `video_processing`: executes detection and depth analysis, including size estimation
- `outlier_detection`: takes as input the csv created in the previous task and perform anomaly detection to remove false positives bounding boxes
- `image_creation`: creates all the images using RGB frames and bounding box annotations from the final csv file

 


