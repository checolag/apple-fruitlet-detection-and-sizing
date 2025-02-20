# *Apple Phenotyping using deep learning and 3D depth analysis: An experimental study on fruitlet sizing*

This GitHub repository includes the code for processing bag files obtained from depth camera recording. 
This guide explains how to process RGB and depth frame sequences to extract most informative data and retrieve individual estimates of fruitlet count and diameters, including validation against caliper measurements.

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


