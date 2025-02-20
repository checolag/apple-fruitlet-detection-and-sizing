from pathlib import Path
import re
import argparse
import yaml
from utils_functions import *

def process_bag_files(config, tasks):
    """
    Process .bag files for apple phenotyping analysis.

    Parameters:
        config (dict): Configuration dictionary containing paths, dates, corymb ranges, and tasks.
        tasks (list): List of tasks to perform, e.g., ["video_processing", "outlier_detection", "image_creation"].
    """

    pattern = re.compile(r".*_(\d+)$")
    surveys = config["dates"]
    base_path = Path(config["base_dir"])
    results_path = Path("results") / config["experiment_name"]
    results_path.mkdir(parents=True, exist_ok=True)

    start_label = config["labels"].get("start_label", None)
    end_label = config["labels"].get("end_label", None)
    corymb_range = determine_corymb_ranges(start_label, end_label)
    if not base_path.exists():
        print(f"Data directory not found: {base_path}")
    else:
        bag_files = sorted([f for f in base_path.glob("*.bag") if f.is_file()])
        filtered_files = []
        for bag_file in bag_files:
            match = pattern.search(bag_file.stem)
            if match:
                number = int(match.group(1))
                if corymb_range[0] <= number <= corymb_range[1]:
                    filtered_files.append(bag_file)

        for bag_file in filtered_files:
            print(f"Processing bag file: {bag_file}")
            output_dir = results_path / bag_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created {output_dir}")

            if "video_processing" in tasks:
                pipeline(config, str(bag_file), output_dir)
            
            if "outlier_detection" in tasks:
                csv_file = output_dir / f"{bag_file.stem}.csv"
                anomaly_detection(config, str(csv_file))
            
            if "image_creation" in tasks:
                csv_file = output_dir / f"{bag_file.stem}.csv"
                output_image = output_dir / "images"
                bag_images(str(bag_file), str(csv_file), output_image)


def determine_corymb_ranges(start_label, end_label):
    """
    Determine the range of corymb labels to analyze.

    Parameters:
        start_label (int): Starting corymb label.
        end_label (int): Ending corymb label.
    Returns:
        tuple: A tuple containing the range of corymb labels (inclusive).
    """
    if start_label is None or end_label is None:
        return (1, 105)
    
    start_label, end_label = int(start_label), int(end_label)
    return (start_label, end_label)

def main(config_path):
    """
    Main function to load the configuration and initiate processing.

    Parameters:
        config_path (str): Path to the YAML configuration file.
    """

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    tasks = config.get("tasks")
    process_bag_files(config, tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process apple phenotyping .bag files.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
