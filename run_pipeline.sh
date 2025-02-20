#!/bin/bash

ENV_NAME="my_env"

# Activate the Python environment
echo "Activating Python environment: $ENV_NAME..."
source ../apple_phenotyping/$ENV_NAME/bin/activate

# # Run the first script
echo "Running main.py with config.yaml..."
python main.py --config config.yaml

if [ $? -eq 0 ]; then
    echo "main.py completed successfully. Proceeding to main_postprocessing.py..."
else
    echo "main.py encountered an error. Exiting."
    exit 1
fi

echo "Running main_postprocessing.py with config.yaml..."
python main_postprocessing.py --config config.yaml

# Check if the second script completed successfully
if [ $? -eq 0 ]; then
    echo "main_postprocessing.py completed successfully. Pipeline finished."
else
    echo "main_postprocessing.py encountered an error. Exiting."
    exit 1
fi
