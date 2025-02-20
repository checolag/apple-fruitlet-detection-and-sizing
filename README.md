# *Apple Phenotyping using deep learning and 3D depth analysis: An experimental study on fruitlet sizing during early development*

This GitHub repository contains the code for processing bag files recorded by Intel® RealSense™ Depth Cameras. 
The pipeline allows you to analyze RGB and depth frame sequences to extract the most informative data for estimating fruitlet count and diameters, including validation against caliper measurements.

### Step 1: Create a Python Local Environment
Start by creating a Python virtual environment and installing the necessary dependencies from the `requirements.txt` file. It might take a while.

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

### Step 2: Download the complete [https://doi.org/10.5281/zenodo.14844598](dataset) and rename the directory in `data`.

### Step 3: Run the Executing Scripts for Video Analysis and Anomaly Detection
```bash
./run_pipeline.sh
```
The code runs two .py scripts used for analyze the bag videos and post-process the detection and sizing output.
The first include: 
- video processing
- 


