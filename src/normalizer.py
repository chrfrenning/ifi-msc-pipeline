# #####################################################################
#
# Normalization
#
# Based on merged dataset produced by the Correlator this step
# is removing outliers, trims recordings to sleep stages,
# calculates global normalization statistics,
# and amends the dataset with signals derived with our
# normalization strategies.
#
# Copyright 2026 Christopher Frenning
# chrifren@ifi.uio.no, christopher@frenning.com
#
# Licensed under the MIT License - see LICENSE.md for details
#
# # Please cite my thesis if you use this code in your own work.
#
# #####################################################################


import os
import shutil
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from datetime import datetime


#
# Logging
#

from src.utilities.mylogging import create_logger
from src.utilities.mytimer import Timer
from src.utilities.mytips import get_life_tip
from src.utilities.myutilities import load_metadata, save_metadata
from src.utilities.mymetrics import track_count, track_latest, track_series

logger = create_logger("normalizer")
vv_logger = create_logger("normalizer_verbose")


#
# Configuration
#



from src.utilities.myconfig import SAMPLING_FREQUENCY, Z_SCORE_THRESHOLD, SIGNALS_TO_NORMALIZE, SLEEP_SCORE_COLUMNS
from src.utilities.myconfig import PROCESSED_DIRECTORY, NORMALIZED_DIRECTORY
from src.utilities.myconfig import SAVE_CSV, SAVE_HDF5


#
# Signals to compute global normalization for
#

def compute_signal_stats(signal_values, z_score_threshold):
    # Lets use numpy for this
    all_values = np.array(signal_values)

    # Remove null values
    all_values = all_values[all_values != 0]

    # Compute raw mean/std before outlier removal (used for z-score outlier detection in the amend step)
    if len(all_values) > 1 and np.std(all_values) > 1e-10:
        raw_mean = float(np.mean(all_values))
        raw_std = float(np.std(all_values))
        z_scores = np.abs((all_values - raw_mean) / raw_std)
        filtered_values = all_values[z_scores < z_score_threshold]
    else:
        raw_mean = float(np.mean(all_values)) if len(all_values) > 0 else 0.0
        raw_std = float(np.std(all_values)) if len(all_values) > 0 else 0.0
        filtered_values = all_values

    stats_dict = {
        'raw_mean': raw_mean,
        'raw_std': raw_std,
        'min': float(filtered_values.min() if len(filtered_values) > 0 else 0),
        'max': float(filtered_values.max() if len(filtered_values) > 0 else 0),
        'mean': float(filtered_values.mean() if len(filtered_values) > 0 else 0),
        'std': float(filtered_values.std() if len(filtered_values) > 0 else 0),
        'median': float(np.median(filtered_values) if len(filtered_values) > 0 else 0),
        'n_samples': len(filtered_values),
        'n_outliers_removed': len(signal_values) - len(filtered_values)
    }

    return stats_dict

def load_and_trim_recording(recording_directory, trim_sleep_stages):
    # Read the source data, which is either HDF5 or CSV
    filename = os.path.join(recording_directory, f"merged_{SAMPLING_FREQUENCY}.h5")
    if not os.path.exists(filename):
        filename = os.path.join(recording_directory, f"merged_{SAMPLING_FREQUENCY}.csv")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Merged file not found for recording {recording_directory}")

    # Read the source data, which is either HDF5 or CSV
    df = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_hdf(filename)

    # Remove any row not scored as sleep in any of the sleep score columns
    if trim_sleep_stages:
        logger.debug(f"Rows in recording {recording_directory} before trimming: {len(df)}")

        present = [c for c in SLEEP_SCORE_COLUMNS if c in df.columns]
        if present:
            df = df[df[present].eq(1).any(axis=1)]

        logger.debug(f"Rows in recording {recording_directory} after trimming: {len(df)}")

    return df

def compute_single_recording_stats(recording_directory, destination_directory, trim_sleep_stages, z_score_threshold, dryrun):
    track_count("compute_single_recording_stats_start")

    # Load and trim the recording
    logger.info(f"Normalizing recording {recording_directory} -> {destination_directory}")
    df = load_and_trim_recording(recording_directory, trim_sleep_stages)

    # Calculate stats for each signal
    signal_stats = {}
    for signal_name in df.columns:
        if any([signal_name.startswith(signal) for signal in SIGNALS_TO_NORMALIZE]):
            signal_stats[signal_name] = compute_signal_stats(df[signal_name].dropna().values, z_score_threshold)

    # Zero out outliers in the df so global stats are computed from clean data
    for signal_name in df.columns:
        if any([signal_name.startswith(signal) for signal in SIGNALS_TO_NORMALIZE]):
            ss = signal_stats[signal_name]
            if ss['raw_std'] > 1e-10:
                z_scores = np.abs((df[signal_name] - ss['raw_mean']) / ss['raw_std'])
                df.loc[z_scores >= z_score_threshold, signal_name] = 0

    # Save the signal stats in the metadata file
    metadata = load_metadata(recording_directory)
    metadata['normalizer'] = {}
    metadata['normalizer']['trim_sleep_stages'] = trim_sleep_stages
    metadata['normalizer']['z_score_threshold'] = z_score_threshold
    metadata['normalizer']['source_directory'] = recording_directory
    metadata['normalizer']['signal_stats'] = signal_stats

    if not dryrun:
        output_directory = os.path.join(destination_directory, os.path.basename(recording_directory))
        os.makedirs(output_directory, exist_ok=True)
        save_metadata(output_directory, metadata)

    track_count("compute_single_recording_stats_success")
    return df

def amend_single_recording_with_normalization(recording_directory, destination_directory, trim_sleep_stages, z_score_threshold, global_stats):
    track_count("amend_single_recording_with_normalization_start")

    #
    # Taxonomy:
    #
    #   _rn_ -> normalized per recording
    #   _gn_ -> normalized globally
    #
    #   We also have per epoch normalization applied in the ML pipeline

    logger.info(f"Amending recording {recording_directory} with normalized signals -> {destination_directory}")
    output_directory = os.path.join(destination_directory, os.path.basename(recording_directory))

    # Load and trim the recording
    df = load_and_trim_recording(recording_directory, trim_sleep_stages)
    metadata = load_metadata(output_directory)

    # Zero out outliers before normalization using per-recording raw_mean/raw_std,
    # keeping track of outlier positions so we can also zero the normalized columns
    outlier_masks = {}
    for signal_name in df.columns:
        if any([signal_name.startswith(signal) for signal in SIGNALS_TO_NORMALIZE]):
            ss = metadata['normalizer']['signal_stats'][signal_name]
            if ss['raw_std'] > 1e-10:
                z_scores = np.abs((df[signal_name] - ss['raw_mean']) / ss['raw_std'])
                outlier_mask = z_scores >= z_score_threshold
                outlier_masks[signal_name] = outlier_mask
                df.loc[outlier_mask, signal_name] = 0
            else:
                outlier_masks[signal_name] = pd.Series(False, index=df.index)

    # Apply per recording normalization
    # Creating new columns with postfix _rn_mean and _rn_zscore
    for signal_name in df.columns:
        if any([signal_name.startswith(signal) for signal in SIGNALS_TO_NORMALIZE]):
            signal_stats = metadata['normalizer']['signal_stats'][signal_name]
            rn_range = signal_stats['max'] - signal_stats['min']
            if rn_range == 0:
                df[f"{signal_name}_rn_mean"] = 0
            else:
                df[f"{signal_name}_rn_mean"] = (df[signal_name] - signal_stats['mean']) / rn_range
            # assert signal_stats['std'] != 0, f"Standard deviation is 0 for signal {signal_name}"
            if signal_stats['std'] == 0:
                df[f"{signal_name}_rn_zscore"] = 0
            else:
                df[f"{signal_name}_rn_zscore"] = df[signal_name].apply(lambda x: (x - signal_stats['mean']) / signal_stats['std'])

            # Zero out normalized values at outlier positions
            if signal_name in outlier_masks:
                df.loc[outlier_masks[signal_name], f"{signal_name}_rn_mean"] = 0
                df.loc[outlier_masks[signal_name], f"{signal_name}_rn_zscore"] = 0

    # Apply global normalization
    for signal_name in df.columns:
        if any([signal_name.startswith(signal) and not '_rn_' in signal_name for signal in SIGNALS_TO_NORMALIZE]):
            signal_stats = global_stats[signal_name]
            gn_range = signal_stats['max'] - signal_stats['min']
            if gn_range == 0:
                df[f"{signal_name}_gn_mean"] = 0
            else:
                df[f"{signal_name}_gn_mean"] = (df[signal_name] - signal_stats['mean']) / gn_range
            assert signal_stats['std'] != 0, f"Standard deviation is 0 for signal {signal_name}"
            if signal_stats['std'] == 0:
                df[f"{signal_name}_gn_zscore"] = 0
            else:
                df[f"{signal_name}_gn_zscore"] = df[signal_name].apply(lambda x: (x - signal_stats['mean']) / signal_stats['std'])

            # Zero out normalized values at outlier positions
            if signal_name in outlier_masks:
                df.loc[outlier_masks[signal_name], f"{signal_name}_gn_mean"] = 0
                df.loc[outlier_masks[signal_name], f"{signal_name}_gn_zscore"] = 0

    # Prepare to update the meadata file
    metadata['normalizer']['output'] = []
    output_info = metadata['normalizer']['output']

    # Now write the amended dataset
    if SAVE_CSV:
        with Timer(operation_name="save_normalized_csv", writelog=True):
            output_directory = os.path.join(destination_directory, os.path.basename(recording_directory))
            os.makedirs(output_directory, exist_ok=True)
            csv_file = os.path.join(destination_directory, os.path.basename(recording_directory), f"normalized_{SAMPLING_FREQUENCY}.csv")
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            df.to_csv(csv_file, index=False)

            output_info.append({
                'type': 'csv',
                'file': os.path.basename(csv_file),
                'size': os.path.getsize(csv_file),
                'sampling_frequency': SAMPLING_FREQUENCY,
                'first_ts': df['timestamp'].min().isoformat(),
                'last_ts': df['timestamp'].max().isoformat(),
                'num_rows': len(df),
            })

    if SAVE_HDF5:
        with Timer(operation_name="save_normalized_hdf5", writelog=True):
            output_directory = os.path.join(destination_directory, os.path.basename(recording_directory))
            os.makedirs(output_directory, exist_ok=True)
            hdf5_file = os.path.join(destination_directory, os.path.basename(recording_directory), f"normalized_{SAMPLING_FREQUENCY}.h5")
            os.makedirs(os.path.dirname(hdf5_file), exist_ok=True)
            df.to_hdf(hdf5_file, key='data', mode='w')

            output_info.append({
                'type': 'h5',
                'file': os.path.basename(hdf5_file),
                'size': os.path.getsize(hdf5_file),
                'sampling_frequency': SAMPLING_FREQUENCY,
                'first_ts': df['timestamp'].min().isoformat(),
                'last_ts': df['timestamp'].max().isoformat(),
                'num_rows': len(df),
            })

    # Save the metadata file
    save_metadata(output_directory, metadata)
    track_count("amend_single_recording_with_normalization_success")


def normalize_all(source_directory, destination_directory, trim_sleep_stages, z_score_threshold, no_global, ignore, dryrun, random_order, max_count, verbose):
    logger.info("Computing statistics and normalizing all recordings")

    # Get all recordings
    all_recordings = sorted(os.listdir(source_directory))
    all_recordings = [x for x in all_recordings if os.path.isdir(os.path.join(source_directory, x))]
    all_recordings = sorted(all_recordings, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    if random_order: random.shuffle(all_recordings)
    all_recordings = [os.path.join(source_directory, recording) for recording in all_recordings]
    logger.info(f"Found {len(all_recordings)} recordings...")
    track_latest("num_recordings", len(all_recordings))

    # While testing you may want to use --max-count and --random to loop faster
    if max_count > 0: all_recordings = all_recordings[:max_count]

    # Warning memory hog, disable with --no-global flag
    global_df = pd.DataFrame()

    # Iterate over each recording and normalize, write new amended dataset to destination directory
    for recording in tqdm(all_recordings, desc="Normalizing", disable=not verbose):
        try:
            df = compute_single_recording_stats(recording, destination_directory, trim_sleep_stages, z_score_threshold, dryrun)
            if not no_global: global_df = pd.concat([global_df, df])
        except Exception as e:
            logger.error(f"Error normalizing recording {recording}: {e}")
            if not ignore:
                raise

    # Calculate the global stats and write to the destination_directory as global_stats.json
    if no_global: sys.exit(0)

    global_stats = {}
    for signal_name in global_df.columns:
        if any([signal_name.startswith(signal) for signal in SIGNALS_TO_NORMALIZE]):
            global_stats[signal_name] = compute_signal_stats(global_df[signal_name].dropna().values, z_score_threshold)

    # Save the global stats to the destination_directory as global_stats.json
    if not dryrun:
        os.makedirs(destination_directory, exist_ok=True)
        with open(os.path.join(destination_directory, 'global_stats.json'), 'w') as f:
            json.dump(global_stats, f, indent=2)
    else:
        json.dump(global_stats, sys.stdout, indent=2)
        sys.stdout.flush()

    # Now we can go back and amend each recording with global and per recording normalization
    if not dryrun:
        for recording in all_recordings:
            try:
                amend_single_recording_with_normalization(recording, destination_directory, trim_sleep_stages, z_score_threshold, global_stats)
            except Exception as e:
                logger.error(f"Error amending recording {recording}: {e}")
                if not ignore:
                    raise

        # if source and destination directories are not the same, copy the summary file
        if source_directory != destination_directory:
            summary_file = os.path.join(source_directory, 'summary.h5')
            if os.path.exists(summary_file):
                shutil.copy(summary_file, os.path.join(destination_directory, 'summary.h5'))
            summary_csv = os.path.join(source_directory, 'summary.csv')
            if os.path.exists(summary_csv):
                shutil.copy(summary_csv, os.path.join(destination_directory, 'summary.csv'))



#
# Entry point
#

def main():
    argparser = argparse.ArgumentParser(description='Normalize all recordings', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    argparser.add_argument('--source', '-s', type=str, default=PROCESSED_DIRECTORY, help='Data directory to process')
    argparser.add_argument('--destination', '-d', type=str, default=NORMALIZED_DIRECTORY, help='Destination directory to save results')
    argparser.add_argument('--trim-sleep-stages', '-t', action='store_true', help='Trim recordings to sleep periods only')
    argparser.add_argument('--z-score-threshold', '-z', type=float, default=Z_SCORE_THRESHOLD, help='Z-score threshold for outlier removal')
    argparser.add_argument('--ignore', '-i', action='store_true', help='Ignore errors and skip to next recording')
    argparser.add_argument('--dryrun', '-n', action='store_true', help='Dry run, don\'t actually write files')
    argparser.add_argument('--random', '-r', action='store_true', help='Shuffle the recordings before processing')
    argparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (debug level)')
    argparser.add_argument('--very-verbose', '-vv', action='store_true', help='Very verbose output (debug level)')
    argparser.add_argument('--max-count', '-m', type=int, default=0, help='Maximum number of recordings to process')
    argparser.add_argument('--no-global', action='store_true', help='Do not compute global normalization statistics')
    args = argparser.parse_args()

    # Set up logging
    if not args.verbose:
        logger.setLevel(logging.WARNING)

    if not args.very_verbose:
        vv_logger.setLevel(logging.CRITICAL)
        logging.getLogger('mytimer').setLevel(logging.CRITICAL)
    
    # Run the main processing
    track_latest("normalize_start_time", datetime.now().isoformat())
    track_latest("max_count", args.max_count)
    track_latest("trim_sleep_stages", args.trim_sleep_stages)
    track_latest("z_score_threshold", args.z_score_threshold)
    track_latest("ignore_errors", args.ignore)
    track_latest("no_global", args.no_global)

    # Print some info
    print(f"* Sleep Study Pipeline - Normalizer (chrifren@ifi.uio.no)")
    print(f"Analyzing all recordings and applying normalization strategies into {args.destination}...")
    print(f"This will take a while... {get_life_tip()}\n")

    with Timer(operation_name="Normalize", writelog=False) as timer:
        normalize_all(args.source, args.destination, args.trim_sleep_stages, args.z_score_threshold, args.no_global, args.ignore, args.dryrun, args.random, args.max_count, args.verbose)
        print(f"Normalization completed in {timer.get_duration():.2f}s")

    track_latest("normalize_end_time", datetime.now().isoformat())

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
