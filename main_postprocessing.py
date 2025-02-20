import os
import subprocess
from pathlib import Path
import re
import argparse
import yaml
from utils_functions import *
from scipy.optimize import linear_sum_assignment
import pickle
from scipy.stats import gaussian_kde

def get_orientation_and_bud_type(label):
    if 1 <= label <= 15:
        orientation = "east"
    elif 16 <= label <= 30 or 101 <= label <= 105:
        orientation = "west"
    else:
        orientation = None
    if label % 3 == 1:
        bud_type = "apical annual"
    elif label % 3 == 2:
        bud_type = "lateral annual"
    elif label % 3 == 0 or 101 <= label <= 105:
        bud_type = "pluriennial"
    else:
        bud_type = None
    return orientation, bud_type

def post_process_results(config, tasks):
    """
    Process .bag files for apple phenotyping analysis.

    Parameters:
        config (dict): Configuration dictionary containing paths, dates, corymb ranges, and tasks.
        tasks (list): List of tasks to perform, e.g., ["video_processing", "outlier_detection", "image_creation"].
    """

    current_dir = Path("results")
    (current_dir / f"final_dataframes").mkdir(parents=True, exist_ok=True)
    final_data_path = current_dir / f"final_dataframes/{config['experiment_name']}.csv"
    final_outliers_path = current_dir / f"final_dataframes/{config['experiment_name']}_outliers.csv"
    if "final_dataset_creation" in tasks:
        if not final_data_path.exists() or not final_outliers_path.exists():
            pattern = re.compile(rf'.*\d+\.csv$')
            csv_files = [file for file in current_dir.glob(f"{config['experiment_name']}/2024*/*.csv") if pattern.match(str(file))]
            csv_files = sorted(csv_files)

            dfs = []
            for file in csv_files:
                df = pd.read_csv(file, index_col=None)
                df['file_name'] = file.stem
                df['date'] = pd.to_datetime(file.stem[:8]).strftime("%Y-%m-%d")
                label_number = int(file.stem.split('_')[-1])
                orientation, bud_type = get_orientation_and_bud_type(label_number)
                df['orientation'] = orientation
                df['bud_type'] = bud_type
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            column_order = ['file_name', 'date', 'orientation', 'bud_type'] + [col for col in combined_df.columns if col not in ['file_name', 'date', 'orientation', 'bud_type']]
            combined_df_ordered = combined_df[column_order]
            def min_measure(row):
                if row['centroid_flag'] == 1:
                    return min(row['dim1_mm'], row['dim2_mm'])
            combined_df_ordered = combined_df_ordered.copy()
            combined_df_ordered['min_measure'] = combined_df_ordered.apply(min_measure, axis=1)
            combined_df_ordered.to_csv(final_data_path, index=False)

            pattern = re.compile(rf'.*\d+_outliers\.csv$')
            csv_files_outliers = [file for file in current_dir.glob(f"{config['experiment_name']}/2024*/*_outliers.csv") if pattern.match(str(file))]
            csv_files_outliers = sorted(csv_files_outliers)
            dfs_outliers = []
            for file in csv_files_outliers:
                df = pd.read_csv(file)
                df['file_name'] = file.stem
                df['date'] = pd.to_datetime(file.stem[:8]).strftime("%Y-%m-%d")
                label_number = int(file.stem.split('_')[-2])
                orientation, bud_type = get_orientation_and_bud_type(label_number)
                df['orientation'] = orientation
                df['bud_type'] = bud_type
                dfs_outliers.append(df)
            combined_outliers_df = pd.concat(dfs_outliers, ignore_index=True)
            column_order = ['file_name', 'date', 'orientation', 'bud_type'] + [col for col in combined_outliers_df.columns if col not in ['file_name', 'date', 'orientation', 'bud_type']]
            combined_df_ordered = combined_outliers_df[column_order]
            combined_outliers_df.to_csv(final_outliers_path, index=False)

    if "validation" in tasks:
        if not final_data_path.exists() or not final_outliers_path.exists():
            print(f"You need to create the final dataset")
            return 
        else:
            df_gt = pd.read_csv(Path("groundtruth") / "file_allcorymbs_new_corrected.csv").dropna(subset=['Diameter'])
            df_gt = df_gt.dropna(subset=['Diameter'])
            df_size = df_gt.groupby(['Date', 'Label']).size().reset_index(name='Count')

            df = pd.read_csv(final_data_path)
            df_corymb = df[df["centroid_flag"]==1]
            df_corymb = df_corymb.copy()
            df_corymb["count"] = df_corymb.groupby(['file_name','frame_number'])['frame_number'].transform('size')
            df_corymb['label'] = df_corymb["file_name"].apply(lambda x: int(x.split('_')[-1]))

            df_outliers = pd.read_csv(final_outliers_path)
            df_filtered = df_corymb.merge(df_outliers, on=['x_c', 'y_c', 'z_c','frame_number'], how='left', indicator=True)
            df_filtered = df_filtered.rename(columns={'file_name_x': 'file_name', 'date_x': 'date', 'orientation_x':'orientation', 
                                                    'bud_type_x':'bud_type', 'conf_score_x':'conf_score', 'min_measure_x':'min_measure', 'count_x':'count'
                                                    })
            df_cleaned = df_filtered[df_filtered['_merge'] == 'left_only'].drop(columns='_merge')
            df_cleaned = df_cleaned[df_corymb.columns]
            df_cleaned["new_count"] = df_cleaned.groupby(['file_name','frame_number'])['frame_number'].transform('size')


            valid_combinations = set(zip(df_cleaned['label'], df_cleaned['date']))
            df_size = df_size[df_size.apply(lambda row: (row['Label'], row['Date']) in valid_combinations, axis=1)]

            if config["frame_extraction"] == "max_stable":
                final_counts_max_stable = []
                for file_name in df_cleaned['file_name'].unique():
                    df_current = df_cleaned[df_cleaned['file_name'] == file_name]
                    bboxes_per_frame = df_current.groupby("frame_number").size()
                    
                    bboxes_per_frame_diff = bboxes_per_frame.diff().fillna(0)
                    stable_counts = bboxes_per_frame[bboxes_per_frame_diff == 0]
                    if not stable_counts.empty:
                        stable_count = stable_counts.max()
                    else:
                        stable_count = 0
                    final_counts_max_stable.append(stable_count)

                df_size = df_size.copy()
                df_size['max_stable_count'] = final_counts_max_stable
                df_cleaned_maxstabletot = df_cleaned.merge(df_size[['Date', 'Label', 'max_stable_count']],
                                        left_on=['date', 'label', 'new_count'], 
                                        right_on=['Date', 'Label', 'max_stable_count'], how='inner').drop(columns=['Date', 'Label', 'max_stable_count'])

            summary_stats = []
            for file_name in df_cleaned_maxstabletot['file_name'].unique(): 
                df_current = df_cleaned_maxstabletot[df_cleaned_maxstabletot['file_name'] == file_name]
                
                diameters = np.array(df_current.groupby('frame_number')['min_measure'].apply(lambda x: x.tolist()).tolist())
                
                n_frames, n_clusters = diameters.shape
                tracked_clusters = [[] for _ in range(n_clusters)]
                previous_frame = diameters[0, :]
                for i in range(n_clusters):
                    tracked_clusters[i].append(previous_frame[i]) 
                for i in range(1, n_frames):
                    current_frame = diameters[i, :]
                    cost_matrix = np.zeros((n_clusters, n_clusters))

                    for j in range(n_clusters):  # Loop over rows
                        for k in range(n_clusters):  # Loop over columns
                            cost_matrix[j, k] = abs(previous_frame[j] - current_frame[k])

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Get optimal row and column indices
                    new_clusters = [None] * n_clusters

                    for j, k in zip(row_ind, col_ind):  # Loop through assigned indices
                        tracked_clusters[j].append(current_frame[k])  # Append current cluster's diameter
                        new_clusters[j] = current_frame[k] 
                    previous_frame = np.array(new_clusters)
                tracked_clusters = [np.array(cluster) for cluster in tracked_clusters]
                for i, cluster in enumerate(tracked_clusters):  # Loop through tracked clusters
                    median_value = np.median(cluster)  # Calculate median
                    summary_stats.append({
                        'file_name': file_name,  # File name
                        'diameter': median_value,  # Median value
                        'method': 'hungarian'  # Method used
                    })

            summary_df_tot = pd.DataFrame(summary_stats)
            summary_df_tot["count"] = summary_df_tot.groupby("file_name").transform('size')

            rmse_results_tot_unsupervised = {}
            def plot_diameters_post_nomatch(df, file_name, list_results):
                df_current = df[df['file_name'] == file_name]
                date = pd.to_datetime(df_current["file_name"].unique()[0].split('_')[0], format='%Y%m%d').strftime("%Y-%m-%d")
                label = int(df_current["file_name"].unique()[0].split('_')[-1]) # to modify
                
                df_gt_frame = df_gt[(df_gt['Date'] == date) & (df_gt['Label'] == label)]
                
                gt_diameters = df_gt_frame['Diameter'].values
                pred_diameters = df_current['diameter'].values
                if len(gt_diameters) == 0 or len(pred_diameters) == 0:
                    print(f"No matching frames for {file_name}. Skipping plotting.")
                    list_results[file_name] = []
                    print(file_name)
                    return

                cost_matrix = np.abs(gt_diameters[:, np.newaxis] - pred_diameters)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                selected_gt = gt_diameters[row_ind]
                selected_pred = pred_diameters[col_ind]
                unmatched_gt = np.delete(gt_diameters, row_ind)
                unmatched_pred = np.delete(pred_diameters, col_ind)
                
                rmse = np.sqrt(np.mean((selected_gt - selected_pred) ** 2))
                    
                list_results[file_name] = {
                    'rmse': round(rmse,3),
                    'matched_gt': selected_gt,
                    'matched_pred': selected_pred,
                    'unmatched_gt': unmatched_gt,
                    'unmatched_pred': unmatched_pred
                }
                
            for file_name in summary_df_tot['file_name'].unique():
                plot_diameters_post_nomatch(summary_df_tot, file_name, rmse_results_tot_unsupervised)

            csv_path = current_dir / 'pipeline_tuning_tests.csv'
            if not csv_path.exists():
                columns = ['test', 'RMSE mean', 'RMSE mode', 'videos with 0 count error', 'mean count error']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(csv_path, index=False)

            test_df = pd.read_csv(csv_path)
            # mean
            rmse_values = np.array([item['rmse'] for item in list(rmse_results_tot_unsupervised.values())])
            mean_rmse = round(rmse_values.mean(),2)
            # mode
            def find_mode_kde(data):
                kde = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 1000)
                kde_vals = kde(x_vals)
                peak_index = np.argmax(kde_vals)
                return x_vals[peak_index]
            best_rmse_mode = round(find_mode_kde(rmse_values), 2)

            # videos with 0 count error
            count = 0
            count_unmatched = 0
            for key, value in rmse_results_tot_unsupervised.items():
                if len(value['unmatched_gt']) == 0 and len(value['unmatched_pred']) == 0:
                    count += 1
                count_unmatched += len(value['unmatched_gt']) + len(value['unmatched_pred'])
            mean_count_error = round(count_unmatched / len(rmse_results_tot_unsupervised), 2)

            new_results = []
            new_results.append({
                    'test': config["experiment_name"], 
                    'RMSE mean': mean_rmse,
                    'RMSE mode': best_rmse_mode,
                    'videos with 0 count error': count,
                    'mean count error': mean_count_error
                })
            new_results_df = pd.DataFrame(new_results)
            test_df = pd.concat([test_df, new_results_df], ignore_index=True)
            test_df.to_csv(current_dir / 'pipeline_tuning_tests.csv', index=False)
            (current_dir / f'pkl_files').mkdir(parents=True, exist_ok=True)
            
            with open(current_dir / f'pkl_files/rmse_{config["experiment_name"]}.pkl', 'wb') as f:
                pickle.dump(rmse_results_tot_unsupervised, f)

def main(config_path):
    """
    Main function to load the configuration and initiate processing.

    Parameters:
        config_path (str): Path to the YAML configuration file.
    """

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    tasks = config.get("tasks")
    post_process_results(config, tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process apple phenotyping .bag files.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
    
