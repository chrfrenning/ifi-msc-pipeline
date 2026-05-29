# #####################################################################
#
# Correlator
#
# Based on data harvested in our first pipeline step this
# process will perform first normalization steps and 
# then merge all signals and annotations into a single dataset.
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
import sys
import json
import random
import logging
import argparse
import pyedflib
import threading
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fractions import Fraction
from scipy.signal import savgol_filter, resample_poly
from scipy import stats

from src.utilities.mylogging import printer, printer_indent
from src.utilities.mysignal import standardize_signal, normalize_signal, estimate_lag, align_short_recording_to_continuous, baseline_correction
from src.utilities.mytips import get_life_tip
from src.utilities.myutilities import load_metadata, save_metadata
from src.utilities.myutilities import postfix_filename, bn, dp
from src.utilities.mytimer import Timer
from src.utilities.mymetrics import track_count, track_latest, track_series, track_sum



# Tell pandas we know what we are doing
pd.set_option('future.no_silent_downcasting', True)



#
# Logging
#

from src.utilities.mylogging import create_logger
logger = create_logger("correlator")
vv_logger = create_logger("correlator_verbose")



#
# Configuration, make changes in your local .env file
#

from src.utilities.myconfig import SAVE_RAW_CSV, SAVE_RAW_HDF5, SAVE_DERIVED_CSV, SAVE_DERIVED_HDF5, SAVE_CSV, SAVE_HDF5
from src.utilities.myconfig import PSG_SIGNALS, SOMNOFY_COLUMNS, PSG_ALIGNMENT_SIGNAL
from src.utilities.myconfig import SAVGOL_WINDOW_SIZE, SAVGOL_POLYORDER, ALIGNMENT_SEARCH_WINDOW, ALIGNMENT_MAX_LAG, SOMNOFY_LENGTH_FOR_ALIGNMENT
from src.utilities.myconfig import SAMPLING_FREQUENCY

SCORING_EVENT_TYPES = [
    'APNEA-CENTRAL',
    'APNEA-MIXED',
    'APNEA-OBSTRUCTIVE',
    'DESAT',
    'HYPOPNEA',
    'SLEEP-REM',
    'SLEEP-S0',
    'SLEEP-S1',
    'SLEEP-S2',
    'SLEEP-S3',
    'SLEEP-UNSCORED',
]



#
# Helper functions
#

def find_edf_file(recording_dir):
    # enum all EDF files
    alternatives = []
    for file in os.listdir(recording_dir):
        if file.upper().endswith('.EDF'):
            filename = os.path.join(recording_dir, file)
            alternatives.append((filename, os.path.getsize(filename)))

    if len(alternatives) > 1:
        logger.warning(f"Multiple EDF files found for {recording_dir}: {alternatives}, using largest file")
        track_count("multiple_edf_files_found")
    
    # sort by size, smallest first
    if len(alternatives) > 0:
        alternatives.sort(key=lambda x: x[1])
        return alternatives[-1][0] # largest file

    # no EDF files found, return None
    return None

def find_scoring_file(recording_dir):
    for file in os.listdir(recording_dir):
        if file.upper().endswith('.TXT'):
            return os.path.join(recording_dir, file)
    
    raise ValueError(f"No scoring file found for {recording_dir}")

def find_somnofy_files(recording_dir, metadata):
    if 'somnofy_metadata' not in metadata:
        return []

    somnyfy_files = []
    for record in metadata.get('somnofy_metadata', []):
        somnyfy_files.append(os.path.join(recording_dir, record['file']))

    return somnyfy_files

def find_sleepstage_files(recording_dir, metadata):
    sleepstage_files = []
    for record in metadata.get('somnofy_sleepstage', []):
        sleepstage_files.append(os.path.join(recording_dir, record['file']))

    return sleepstage_files

def get_max_sampling_frequency(f):
    max_sampling_frequency = 0
    for i in range(f.signals_in_file):
        if f.getLabel(i) in PSG_SIGNALS:
            sampling_frequency = f.getSampleFrequency(i)
            vv_logger.debug(f"Sampling frequency for {f.getLabel(i)}: {sampling_frequency}")
            if sampling_frequency > max_sampling_frequency:
                max_sampling_frequency = sampling_frequency

    track_series("max_sampling_frequency", max_sampling_frequency)
    return max_sampling_frequency

def calculate_ahi(scoring_df):
    apnea_events = {'APNEA-OBSTRUCTIVE', 'APNEA-CENTRAL', 'APNEA-MIXED'}
    sleep_stages = {'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S3', 'SLEEP-REM'}

    if scoring_df is None or scoring_df.empty:
        return {'ahi': 0.0, 'apnea_count': 0, 'hypopnea_count': 0, 'total_events': 0, 'total_sleep_time_hours': 0.0}

    apnea_count = int(scoring_df['event'].isin(apnea_events).sum())
    hypopnea_count = int((scoring_df['event'] == 'HYPOPNEA').sum())
    total_events = apnea_count + hypopnea_count

    tst_seconds = scoring_df.loc[scoring_df['event'].isin(sleep_stages), 'duration'].sum()
    tst_hours = tst_seconds / 3600.0

    ahi = total_events / tst_hours if tst_hours > 0 else 0.0

    track_series("ahi", ahi)
    track_series("apnea_count", apnea_count)
    track_series("hypopnea_count", hypopnea_count)
    track_series("total_events", total_events)
    track_series("total_sleep_time_hours", tst_hours)

    return {
        'ahi': round(ahi, 2),
        'apnea_count': apnea_count,
        'hypopnea_count': hypopnea_count,
        'total_events': total_events,
        'total_sleep_time_hours': round(tst_hours, 4),
    }



#
# PSG conversion
#
# The PSG signal is our master signal - without it no other data makes sense
# The EDF file contains signals of varying sampling frequencies that we want to harmonize
#
# We use the starttime and endtime of the EDF file as the master
# The reason is that the PSG machine is on for an entire overnight recording
# while the Somnofy device turns on/off as it detects a patient
#
# Our dataset is basically a single row for every second of the night
# As we continue processing, the other formats will be tailored to extend this
# dataset with more colums with flags telling us if that row has been scored somehow
#

def convert_edf(edf_file, destination_dir):
    with Timer(operation_name="convert_edf", writelog=True) as timer:
        logger.debug(f"Converting EDF file {bn(edf_file)}")

        # Read the EDF file
        with Timer(operation_name="read_edf_file", writelog=True) as timer:
            f = pyedflib.EdfReader(edf_file)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        max_sampling_frequency = get_max_sampling_frequency(f)

        vv_logger.debug(f"Found {n} signals: {signal_labels}, max sampling frequency: {max_sampling_frequency}")

        data_dict = {}
        signal_freqs = {}
        for i in range(f.signals_in_file):
            if signal_labels[i] in PSG_SIGNALS:
                logger.debug(f"Reading signal {signal_labels[i]}")
                data_dict[signal_labels[i]] = f.readSignal(i)
                signal_freqs[signal_labels[i]] = f.getSampleFrequency(i)

        # resample all signals to the max sampling frequency
        max_duration = 0
        for signal in data_dict:
            signal_duration = len(data_dict[signal]) / signal_freqs[signal]
            if signal_duration > max_duration:
                max_duration = signal_duration
                
        target_length = int(np.ceil(max_duration * max_sampling_frequency))
        logger.debug(f"Resampling all signals to max sampling frequency of {max_sampling_frequency} Hz, target length: {target_length}")
        
        with Timer(operation_name="Upsample all signals to max sampling frequency", writelog=True):
            for signal in data_dict:
                current_freq = signal_freqs[signal]
                if current_freq != max_sampling_frequency:
                    freq_ratio = max_sampling_frequency / current_freq
                    freq_fraction = Fraction(freq_ratio).limit_denominator(10000)
                    up = freq_fraction.numerator
                    down = freq_fraction.denominator
                    logger.debug(f"Resampling {signal} from {current_freq} Hz to {max_sampling_frequency} Hz (up={up}, down={down})")
                    with Timer(operation_name=f"Resample {signal} to max sampling frequency", writelog=True):
                        data_dict[signal] = resample_poly(data_dict[signal], up, down, window=("kaiser", 5.0))
                    logger.debug(f"Signal {signal} resampled to length {len(data_dict[signal])}")
                else:
                    logger.debug(f"Signal {signal} already at max sampling frequency, length {len(data_dict[signal])}")

        # convert to pandas dataframe, create timestamps
        logger.debug(f"Converting to pandas dataframe, creating timestamps")
        df = pd.DataFrame(data_dict)
        start_ts = f.getStartdatetime()
        df.insert(0, 'timestamp', pd.date_range(
            start=start_ts, 
            periods=len(df), 
            freq=pd.Timedelta(seconds=1/max_sampling_frequency)
        ))

        # save the processed data in full resolution
        csv_file_name = os.path.join(destination_dir, f"psg.csv")
        if SAVE_RAW_CSV:
            logger.debug(f"Saving high resolution CSV in {csv_file_name}")
            with Timer(operation_name="save_highres_csv", writelog=True):
                df.to_csv(csv_file_name, index=False)
        
        h5_file_name = os.path.join(destination_dir, f"psg.h5")
        if SAVE_RAW_HDF5:
            logger.debug(f"Saving high resolution HDF5 in {h5_file_name}")
            with Timer(operation_name="save_highres_hdf5", writelog=True):
                df.to_hdf(h5_file_name, key='data', mode='w')

        # resample to 1hz
        logger.debug(f"Resampling to {SAMPLING_FREQUENCY}")
        track_latest("sampling_frequency", SAMPLING_FREQUENCY)
        with Timer(operation_name="resample_to_sampling_frequency", writelog=True):
            df.set_index('timestamp', inplace=True)
            df = df.resample(SAMPLING_FREQUENCY).mean()
            df.reset_index(inplace=True)

        if SAVE_DERIVED_CSV:
            csv_file_name = postfix_filename(csv_file_name, SAMPLING_FREQUENCY)
            logger.debug(f"Saving 1Hz data to CSV in {csv_file_name}")
            with Timer(operation_name="save_resampled_csv", writelog=True):
                df.to_csv(csv_file_name, index=False)

        if SAVE_DERIVED_HDF5:
            h5_file_name = postfix_filename(h5_file_name, SAMPLING_FREQUENCY)
            logger.debug(f"Saving 1Hz data to HDF5 in {h5_file_name}")
            with Timer(operation_name="save_resampled_hdf5", writelog=True):
                df.to_hdf(h5_file_name, key='data', mode='w')

        return df



#
# Scoring conversion
#
# The scoring file has timestamp and duration fields
# We convert this to a set of rows with boolean flags for each event
# This can then be merged into the main PSG dataframe with one row per second
# of the overnight sleep study
#
# We therefore do not store the duration per se, but a set of rows with the same
# boolean flag can be used to deduce the duration of an event
#

def read_scoring_file(scoring_file):
    with open(scoring_file, 'r') as f:
        return f.read()

def create_empty_scoring_df():
    return pd.DataFrame(columns=['timestamp','APNEA-OBSTRUCTIVE','SLEEP-S1','DESAT','HYPOPNEA','SLEEP-S2','SLEEP-S3','APNEA-MIXED','APNEA-CENTRAL','SLEEP-REM'])

def convert_scoring(scoring_file, destination_dir):
    # Create empty scoring df if we have none
    if not os.path.exists(scoring_file):
        return create_empty_scoring_df()

    #
    # Read and parse the scoring file
    #

    scoring_content = read_scoring_file(scoring_file)
    
    # Pick out everything on the line after the line that starts with Time [hh:mm:ss.xxx] Event
    pos = scoring_content.find('\nTime [hh:mm:ss.xxx]\tEvent')
    next_line_pos = scoring_content.find('\n', pos+1)
    all_events = scoring_content[next_line_pos+1:]
    
    # Parse the tab-separated data
    lines = all_events.strip().split('\n')
    data_rows = []
    
    for line in lines:
        if line.strip():  # skip empty lines
            parts = line.split('\t')
            if len(parts) >= 3:  # ensure we have timestamp, event, and duration
                timestamp_str = parts[0]
                event = parts[1]
                duration_str = parts[2]
                
                # Convert timestamp to pandas datetime
                timestamp = pd.to_datetime(timestamp_str)
                
                # Convert duration to float
                try:
                    duration = float(duration_str)
                except ValueError:
                    duration = 0.0  # fallback for invalid duration values
                
                data_rows.append({
                    'timestamp': timestamp,
                    'event': event,
                    'duration': duration
                })
    
    # If there is no data, return an empty dataframe
    if len(data_rows) == 0:
        track_count("no_scoring_data")
        return create_empty_scoring_df()
    
    df = pd.DataFrame(data_rows)

    # save the original transformed source
    if SAVE_CSV:
        logger.debug(f"Saving scoring file as CSV")
        csv_file = os.path.join(destination_dir, 'scoring.csv')
        df.to_csv(csv_file, index=False)
        
    if SAVE_HDF5:
        logger.debug(f"Saving scoring file as HDF5")
        h5_file = os.path.join(destination_dir, 'scoring.h5')
        df.to_hdf(h5_file, key='data', mode='w')

    # Create columns for each distinct event, interpolate to the duration value for each event
    start_time = df['timestamp'].min().floor(SAMPLING_FREQUENCY)
    end_time = df['timestamp'].max().floor(SAMPLING_FREQUENCY)
    unique_events = df['event'].unique()
    vv_logger.debug(f"Unique events: {unique_events}")
    track_series("unique_events", len(unique_events))

    # create a dataframe with timestamps in SAMPLING_FREQUENCY resolution between start_time and end_time
    timestamps = pd.date_range(start=start_time, end=end_time, freq=SAMPLING_FREQUENCY)
    df_timestamps = pd.DataFrame({'timestamp': timestamps})

    # insert a column for each event and initialize to 0
    for event in unique_events:
        df_timestamps[event] = 0

    # for each row in df, set the column for the event to 1 for the duration of the event in df_timestamps
    for index, row in df.iterrows():
        event = row['event']
        duration = row['duration']
        df_timestamps.loc[(df_timestamps['timestamp'] >= row['timestamp']) & (df_timestamps['timestamp'] <= row['timestamp'] + timedelta(seconds=duration)), event] = 1

    # save the result
    if SAVE_DERIVED_CSV:
        logger.debug(f"Saving result source to CSV")
        csv_file = os.path.join(destination_dir, f'scoring_{SAMPLING_FREQUENCY}.csv')
        df_timestamps.to_csv(csv_file, index=False)
    
    if SAVE_DERIVED_HDF5:
        logger.debug(f"Saving result to HDF5")
        h5_file = os.path.join(destination_dir, f'scoring_{SAMPLING_FREQUENCY}.h5')
        df_timestamps.to_hdf(h5_file, key='data', mode='w')
    
    return df_timestamps, df


#
# VitalThings Sleep Staging conversion
#

def convert_sleepstage(sleepstage_files, destination_dir):
    if len(sleepstage_files) == 0:
        track_count("no_sleepstage_files")
        return pd.DataFrame(columns=['timestamp', 'sleep_stage'])

    expanded_dfs = []

    for sleepstage_file in sleepstage_files:
        logger.debug(f"Processing sleepstage file {bn(sleepstage_file)}")
        df = pd.read_csv(sleepstage_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        # Remove timezone info and sub-second precision to match PSG timestamps
        df['timestamp'] = df['timestamp'].dt.tz_localize(None).dt.floor('s')
        df.set_index('timestamp', inplace=True)

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=SAMPLING_FREQUENCY)
        df = df.reindex(full_range)
        df.index.name = 'timestamp'
        df['sleep_stage'] = df['sleep_stage'].ffill().astype(int)
        df.reset_index(inplace=True)

        expanded_dfs.append(df)

    # Merge all expanded dataframes into a single timeline
    start_time = min(df['timestamp'].min() for df in expanded_dfs)
    end_time = max(df['timestamp'].max() for df in expanded_dfs)

    timestamps = pd.date_range(start=start_time, end=end_time, freq=SAMPLING_FREQUENCY)
    merged = pd.DataFrame({'timestamp': timestamps})
    merged.set_index('timestamp', inplace=True)

    for df in expanded_dfs:
        df.set_index('timestamp', inplace=True)
        merged = merged.combine_first(df)

    merged['sleep_stage'] = merged['sleep_stage'].ffill().bfill().astype(int)
    merged.reset_index(inplace=True)

    if SAVE_DERIVED_CSV:
        csv_file = os.path.join(destination_dir, f'sleepstage_{SAMPLING_FREQUENCY}.csv')
        logger.debug(f"Saving merged sleepstage to CSV in {csv_file}")
        merged.to_csv(csv_file, index=False)

    if SAVE_DERIVED_HDF5:
        h5_file = os.path.join(destination_dir, f'sleepstage_{SAMPLING_FREQUENCY}.h5')
        logger.debug(f"Saving merged sleepstage to HDF5 in {h5_file}")
        merged.to_hdf(h5_file, key='data', mode='w')

    return merged



#
# Somnofy conversion
#

def _build_merged_somnofy(recording_dir, dfs, file_prefix):
    start_time = dfs[0]['timestamp'].min().floor(SAMPLING_FREQUENCY)
    end_time = dfs[0]['timestamp'].max().floor(SAMPLING_FREQUENCY)
    for df in dfs:
        df_start = df['timestamp'].min().floor(SAMPLING_FREQUENCY)
        df_end = df['timestamp'].max().floor(SAMPLING_FREQUENCY)
        if df_start < start_time:
            start_time = df_start
        if df_end > end_time:
            end_time = df_end
    
    # create a dataframe with timestamps in SAMPLING_FREQUENCY resolution between start_time and end_time
    timestamps = pd.date_range(start=start_time, end=end_time, freq=SAMPLING_FREQUENCY)
    df_timestamps = pd.DataFrame({'timestamp': timestamps})
    df_timestamps.set_index('timestamp', inplace=True)

    # merge each df into df_timestamps
    for df in dfs:
        df.set_index('timestamp', inplace=True)
        df = df.resample(SAMPLING_FREQUENCY).mean()
        #df_timestamps = pd.merge(df_timestamps, df, on='timestamp', how='left')
        df_timestamps = df_timestamps.combine_first(df)

    # fill in the missing values with 0
    df_timestamps = df_timestamps.fillna(0).infer_objects(copy=False)
    df_timestamps.reset_index(inplace=True)

    # resample to SAMPLING_FREQUENCY
    #df_timestamps.set_index('timestamp', inplace=True)
    #df_timestamps = df_timestamps.resample(SAMPLING_FREQUENCY).mean()
    #df_timestamps.reset_index(inplace=True)

    # Save the result
    if SAVE_DERIVED_CSV:
        csv_file = os.path.join(recording_dir, f'{file_prefix}_{SAMPLING_FREQUENCY}.csv')
        logger.debug(f"Saving intermediate Somnofy dataset to CSV in {csv_file}")
        df_timestamps.to_csv(csv_file, index=False)

    if SAVE_DERIVED_HDF5:
        h5_file = os.path.join(recording_dir, f'{file_prefix}_{SAMPLING_FREQUENCY}.h5')
        logger.debug(f"Saving intermediate Somnofy dataset to HDF5 in {h5_file}")
        df_timestamps.to_hdf(h5_file, key='data', mode='w')

    return df_timestamps

def create_empty_somnofy_df():
    return pd.DataFrame(columns=SOMNOFY_COLUMNS)

def convert_somnofy(metadata, somnofy_files, edf_df, destination_dir):
    # If we don't have any data, return empty dataframes
    if len(somnofy_files) == 0:
        track_count("no_somnofy_files")
        return create_empty_somnofy_df(), create_empty_somnofy_df()

    # Initialize lists to store the dataframes for each somnofy file
    somnofy_dfs = []
    aligned_dfs = []
    can_align = PSG_ALIGNMENT_SIGNAL in edf_df.columns

    # We need a node in the metadata to store the alignment shifts we apply
    if not metadata.get('correlator', None):
        metadata['correlator'] = {
            'sampling_frequency': SAMPLING_FREQUENCY,
        }

    if not metadata['correlator'].get('baselineadjustment', None):
        metadata['correlator']['baselineadjustment'] = {
            'savgol_window_size': SAVGOL_WINDOW_SIZE,
            'savgol_polyorder': SAVGOL_POLYORDER,
        }

    # Iterate over each somnofy file and convert it to a dataframe
    metadata['correlator']['somnofy'] = {}
    for somnofy_file in somnofy_files:
        somnofy_df = pd.read_csv(somnofy_file)
        somnofy_df['timestamp'] = pd.to_datetime(somnofy_df['timestamp']) # convert to datetime

        metadata['correlator']['somnofy'][os.path.basename(somnofy_file)] = {}
        infodict = metadata['correlator']['somnofy'][os.path.basename(somnofy_file)]
        
        logger.debug(f"Applying baseline correction to relative_distance with window size {SAVGOL_WINDOW_SIZE} and polyorder {SAVGOL_POLYORDER}")
        with Timer(operation_name="baseline_correction", writelog=True) as baselinetimer:
            somnofy_df['rel_dist_corr'] = baseline_correction(somnofy_df, 'relative_distance') # apply baseline correction
            infodict['baseline_corr_duration_sec'] = baselinetimer.get_duration()

        # create aligned copy; timestamps will be shifted if alignment succeeds
        aligned_df = somnofy_df.copy()

        if can_align and len(somnofy_df) > 0:
            psg_start = edf_df['timestamp'].iloc[0]
            somnofy_start = somnofy_df['timestamp'].iloc[0]
            approximate_start_index = int((somnofy_start - psg_start).total_seconds())

            # resample to 1Hz for a contiguous array suitable for cross-correlation
            temp_df = somnofy_df[['timestamp', 'rel_dist_corr']].copy()
            temp_df.set_index('timestamp', inplace=True)
            logger.debug(f"Resampling to {SAMPLING_FREQUENCY}Hz")
            with Timer(operation_name="Resampling", writelog=True) as resamplingtimer:
                temp_df = temp_df.resample(SAMPLING_FREQUENCY).mean().interpolate()
                infodict['resampling_duration_sec'] = resamplingtimer.get_duration()
            somnofy_signal = temp_df['rel_dist_corr'].values
            chest_signal = edf_df[PSG_ALIGNMENT_SIGNAL].values

            # standardize both signals to handle scale differences
            if np.std(somnofy_signal) > 0 and np.std(chest_signal) > 0:
                somnofy_signal = standardize_signal(somnofy_signal)
                chest_signal = standardize_signal(chest_signal)

                logger.debug(f"Aligning Somnofy signals to PSG signal")
                with Timer(operation_name="signal_alignment", writelog=True) as alignmenttimer:
                    best_start_idx, best_correlation = align_short_recording_to_continuous(
                        chest_signal, somnofy_signal[:SOMNOFY_LENGTH_FOR_ALIGNMENT], approximate_start_index, ALIGNMENT_SEARCH_WINDOW, ALIGNMENT_MAX_LAG
                    )
                    infodict['alignment_duration_sec'] = alignmenttimer.get_duration()
                    infodict['alignment_correlation'] = best_correlation
                    infodict['alignment_start_idx'] = int(best_start_idx)
                    infodict['alignment_search_window'] = ALIGNMENT_SEARCH_WINDOW
                    infodict['alignment_max_lag'] = ALIGNMENT_MAX_LAG
                    infodict['somnofy_length_for_alignment'] = SOMNOFY_LENGTH_FOR_ALIGNMENT

                    track_series("alignment_correlation", best_correlation)

                shift_seconds = int(best_start_idx - approximate_start_index)
                logger.debug(f"Alignment for {os.path.basename(somnofy_file)}: shift={shift_seconds}s, correlation={best_correlation:.4f}")

                # apply the same temporal offset to all signals in this somnofy file
                aligned_df['timestamp'] = aligned_df['timestamp'] + timedelta(seconds=shift_seconds)
                infodict['alignment_shift_seconds'] = shift_seconds
                track_series("alignment_shift_seconds", shift_seconds)
            else:
                logger.debug(f"Alignment skipped for {os.path.basename(somnofy_file)}: constant signal detected")

        somnofy_dfs.append(somnofy_df)
        aligned_dfs.append(aligned_df)

    original_merged = _build_merged_somnofy(destination_dir, somnofy_dfs, 'somnofy')
    aligned_merged = _build_merged_somnofy(destination_dir, aligned_dfs, 'somnofy_aligned')

    return original_merged, aligned_merged



#
# Primary processing is loading and merging datasets
# 
# Must expand and/or resample to desired Hz before merging
# Then merging all datasets into a single dataset
# Output HDF5 and optionally CSV for ease of use
#

def merge_datasets(metadata, edf_df, scoring_df, sleepstage_df, somnofy_df, aligned_somnofy_df, destination_dir):
    ''' Merges all datasets using the PSG 1Hz as master, cutting off what is outside (if any) for other signals and scoring '''
    
    # rename all edf columns to psg_{column_name}
    edf_df.columns = [f'psg_{col.lower()}' if col != 'timestamp' else col for col in edf_df.columns]

    # rename all scoring columns to scoring_{column_name}
    scoring_df.columns = [f'scoring_{col.lower()}' if col != 'timestamp' else col for col in scoring_df.columns]

    # merge scoring into the psg dataframe
    new_df = pd.merge(edf_df, scoring_df, on='timestamp', how='left')
    new_df = new_df.fillna(0).infer_objects(copy=False) # fill in the missing values with 0
    scoring_cols = [col for col in new_df.columns if col.startswith('scoring_')]
    new_df[scoring_cols] = new_df[scoring_cols].astype(int)

    # rename all sleepstage columns to sleepstage_{column_name}
    sleepstage_df.columns = [f'vtss_{col.lower()}' if col != 'timestamp' else col for col in sleepstage_df.columns]

    # cut any sleepstage rows that are outside the range of the psg dataframe
    sleepstage_df = sleepstage_df[sleepstage_df['timestamp'] >= edf_df['timestamp'].min()]
    sleepstage_df = sleepstage_df[sleepstage_df['timestamp'] <= edf_df['timestamp'].max()]

    # merge sleepstage into the psg dataframe
    new_df = pd.merge(new_df, sleepstage_df, on='timestamp', how='left')
    new_df = new_df.fillna(0).infer_objects(copy=False) # fill in the missing values with 0

    # rename all somnofy columns to somnofy_{column_name}
    somnofy_df.columns = [f'somnofy_{col.lower()}' if col != 'timestamp' else col for col in somnofy_df.columns]

    # cut any somnofy rows that are outside the range of the psg dataframe
    somnofy_df = somnofy_df[somnofy_df['timestamp'] >= edf_df['timestamp'].min()]
    somnofy_df = somnofy_df[somnofy_df['timestamp'] <= edf_df['timestamp'].max()]

    # merge somnofy into the psg dataframe
    new_df = pd.merge(new_df, somnofy_df, on='timestamp', how='left')
    new_df = new_df.fillna(0).infer_objects(copy=False) # fill in the missing values with 0

    # rename all aligned somnofy columns to aligned_{column_name}
    aligned_somnofy_df.columns = [f'aligned_{col.lower()}' if col != 'timestamp' else col for col in aligned_somnofy_df.columns]

    # cut any aligned somnofy rows that are outside the range of the psg dataframe
    aligned_somnofy_df = aligned_somnofy_df[aligned_somnofy_df['timestamp'] >= edf_df['timestamp'].min()]
    aligned_somnofy_df = aligned_somnofy_df[aligned_somnofy_df['timestamp'] <= edf_df['timestamp'].max()]

    # merge aligned somnofy into the psg dataframe
    new_df = pd.merge(new_df, aligned_somnofy_df, on='timestamp', how='left')
    new_df = new_df.fillna(0).infer_objects(copy=False) # fill in the missing values with 0

    # NOTE: We had normalization as a step here, but removed and into a separate step
    # due to the number of variants we need

    # save the result
    if not SAVE_CSV and not SAVE_HDF5:
        raise RuntimeError("We must store the result of this operation, configure either SAVE_CSV or SAVE_HDF5 in your .env file.")

    metadata['correlator']['output'] = []

    if SAVE_CSV:
        csv_file = os.path.join(destination_dir, f'merged_{SAMPLING_FREQUENCY}.csv')
        logger.debug(f"Saving final merged dataset to CSV in {csv_file}")
        with Timer(operation_name="save_merged_csv", writelog=True):
            new_df.to_csv(csv_file, index=False)
        
        metadata['correlator']['output'].append({
            'type': 'csv',
            'file': os.path.basename(csv_file),
            'size': os.path.getsize(csv_file),
            'sampling_frequency': SAMPLING_FREQUENCY,
            'first_ts': new_df['timestamp'].min().isoformat(),
            'last_ts': new_df['timestamp'].max().isoformat(),
            'num_rows': len(new_df),
        })

    if SAVE_HDF5:
        h5_file = os.path.join(destination_dir, f'merged_{SAMPLING_FREQUENCY}.h5')
        logger.debug(f"Saving final merged dataset to HDF5 in {h5_file}")
        with Timer(operation_name="save_merged_hdf5", writelog=True):
            new_df.to_hdf(h5_file, key='data', mode='w')

        metadata['correlator']['output'].append({
            'type': 'h5',
            'file': os.path.basename(h5_file),
            'size': os.path.getsize(h5_file),
            'sampling_frequency': SAMPLING_FREQUENCY,
            'first_ts': new_df['timestamp'].min().isoformat(),
            'last_ts': new_df['timestamp'].max().isoformat(),
            'num_rows': len(new_df),
        })
    
    return new_df, metadata

def process_single_recording(metadata, edf_file, scoring_file, somnofy_files, sleepstage_files, destination_dir):
    with Timer(operation_name="process_single_recording", writelog=True) as timer:
        track_count("process_single_recording_start")

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)

        edf_df = convert_edf(edf_file, destination_dir)
        scoring_df, original_scoring_df = convert_scoring(scoring_file, destination_dir)
        sleepstage_df = convert_sleepstage(sleepstage_files, destination_dir)
        somnofy_df, aligned_somnofy_df = convert_somnofy(metadata, somnofy_files, edf_df, destination_dir)
        _, _ = merge_datasets(metadata, edf_df, scoring_df, sleepstage_df, somnofy_df, aligned_somnofy_df, destination_dir)

        # Calculate AHI
        metadata['ahi'] = calculate_ahi(original_scoring_df)

        # Save the metadata to the destination directory
        save_metadata(destination_dir, metadata)

        track_count("process_single_recording_success")

        # calculate disk space used by the recording
        disk_space_used = os.path.getsize(destination_dir)
        track_sum("disk_space_used", disk_space_used)


#
# Main work happens here
#

def correlate_single_recording(recording_dir, destination_dir, require_somnofy, require_sleepstage):
    track_count("correlate_single_recording_start")

    # Load the metadata file for this overnight recording
    metadata_file = os.path.join(recording_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file does not exist")
        return

    metadata = None
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Find the EDF file to use, err if there is none
    edf_file = find_edf_file(recording_dir)
    if not edf_file:
        raise ValueError(f"No EDF file found for {recording_dir}")

    # Find the scoring file to use, err if there is none
    scoring_file = find_scoring_file(recording_dir)
    if not scoring_file:
        raise ValueError(f"No scoring file found for {recording_dir}")

    somnofy_files = find_somnofy_files(recording_dir, metadata)
    if require_somnofy and len(somnofy_files) == 0:
        raise ValueError(f"No somnofy files found for {recording_dir}")

    sleepstage_files = find_sleepstage_files(recording_dir, metadata)
    if require_sleepstage and not sleepstage_files:
        raise ValueError(f"No sleepstage files found for {recording_dir}")

    # Add correlator metadata
    metadata['correlator'] = {
        'source_directory': os.path.abspath(recording_dir),
    }

    process_single_recording(metadata, edf_file, scoring_file, somnofy_files, sleepstage_files, destination_dir)
    track_count("correlate_single_recording_success")

def correlate_worker(recording_dir, destination_directory, reprocess_all, ignore, dryrun, require_somnofy, require_sleepstage, job_limit):
    recording_id = os.path.basename(recording_dir)

    if not os.path.isdir(recording_dir):
        return

    # Bail if we already reached the max job limit (for testing)
    with job_limit['lock']:
        if job_limit['remaining'] <= 0:
            return

    try:

        destination_dir = os.path.join(destination_directory, recording_id)
        if os.path.exists(os.path.join(destination_dir, f'merged_{SAMPLING_FREQUENCY}.csv')) and not reprocess_all:
            logger.debug(f"Skipping recording {os.path.basename(recording_dir)} because it already exists")
            return

        logger.debug(f"Processing recording {bn(recording_dir)} -> {destination_dir}")
        correlate_single_recording(recording_dir, destination_dir, require_somnofy, require_sleepstage)

        # Decrement max job limit counter
        with job_limit['lock']:
            job_limit['remaining'] -= 1

    except Exception as e:
        logger.error(f"Error processing recording {recording_id}: {e}")
        if not ignore:
            traceback.print_exc()
            sys.exit(1)

def correlate_all_recordings(source_directory, destination_directory, reprocess_all, ignore, dryrun, max_count, random_order, require_somnofy, require_sleepstage, workers, verbose):
    # Get all recordings from the source directory
    all_recordings = list(os.listdir(source_directory))
    all_recordings = [os.path.join(source_directory, recording) for recording in all_recordings] # make fullpaths
        
    # Optionally randomize for better testing, otherwise sort for reproducibility and log reading
    if random_order:
        random.shuffle(all_recordings)
    else:
        all_recordings.sort(key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    # Error if no recorings found
    if len(all_recordings) == 0:
        logger.error(f"No recordings found in {source_directory}")
        return

    # Ensure the destination directory exists
    if not dryrun:
        os.makedirs(destination_directory, exist_ok=True)
    else:
        logger.info(f"Dry run, will not actually copy files")

    # Thread-safe counter so workers stop after max_count successful correlations
    max_count = max_count if max_count > 0 else len(all_recordings)
    job_limit = {'remaining': max_count, 'lock': threading.Lock()}

    # executor = ThreadPoolExecutor(max_workers=workers)
    # futures = [
    #     executor.submit(correlate_worker, recording_dir, destination_directory, reprocess_all, ignore, dryrun, require_somnofy, require_sleepstage, job_limit)
    #     for recording_dir in all_recordings
    # ]
    # executor.shutdown(wait=True)

    if not verbose: print(f"Correlating {len(all_recordings)} recordings with {workers} worker threads...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                correlate_worker,
                recording_dir,
                destination_directory,
                reprocess_all,
                ignore,
                dryrun,
                require_somnofy,
                require_sleepstage,
                job_limit,
            )
            for recording_dir in all_recordings
        ]

        for future in tqdm(as_completed(futures), total=len(futures), disable=verbose):
            result = future.result()

def create_summary(destination_directory, verbose):
    if not verbose: print(f"Creating summary of all recordings in {destination_directory}/summary.(h5|csv)")

    rows = []
    for recording_id in tqdm(sorted(os.listdir(destination_directory)), disable=verbose):
        recording_dir = os.path.join(destination_directory, recording_id)
        if not os.path.isdir(recording_dir):
            continue

        try:
            metadata = load_metadata(recording_dir)
        except Exception as e:
            logger.warning(f"Skipping {recording_id}: could not load metadata: {e}")
            continue

        scoring_meta = metadata.get('scoring_metadata', {})
        events = set(scoring_meta.get('events', []))
        correlator = metadata.get('correlator', {})
        somnofy_dict = correlator.get('somnofy', {})

        # Alignment_shift_seconds (we take only the first, its a sanity check)
        first_shift = None
        if somnofy_dict:
            first_key = next(iter(somnofy_dict))
            first_shift = somnofy_dict[first_key].get('alignment_shift_seconds')

        # Recording length from first/last timestamps
        dt_first = pd.to_datetime(metadata.get('first_ts'))
        dt_last = pd.to_datetime(metadata.get('last_ts'))
        recording_length_hours = round((dt_last - dt_first).total_seconds() / 3600.0, 4)

        ahi_meta = metadata.get('ahi', {})

        row = {
            'recording_id': metadata.get('recording_id'),
            'device_id': scoring_meta.get('patient_id'),
            'num_channels': int(metadata.get('num_channels', 0)),
            'first_ts': dt_first,
            'last_ts': dt_last,
            'recording_length_hours': recording_length_hours,
            'ahi': ahi_meta.get('ahi'),
            'apnea_count': ahi_meta.get('apnea_count'),
            'hypopnea_count': ahi_meta.get('hypopnea_count'),
            'total_events': ahi_meta.get('total_events'),
            'total_sleep_time_hours': ahi_meta.get('total_sleep_time_hours'),
            'has_somnofy': bool(metadata.get('somnofy_metadata', [])),
            'has_vt_sleepstage': bool(metadata.get('somnofy_sleepstage', [])),
            'alignment_shift_seconds': first_shift,
        }

        for event_type in SCORING_EVENT_TYPES:
            col_name = 'has_' + event_type.lower().replace('-', '_')
            row[col_name] = event_type in events

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Deterministic sort by recording_id (numeric where possible)
    if not summary_df.empty:
        summary_df['_sort_key'] = pd.to_numeric(summary_df['recording_id'], errors='coerce')
        summary_df.sort_values('_sort_key', na_position='last', inplace=True)
        summary_df.drop(columns=['_sort_key'], inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    if SAVE_HDF5: summary_df.to_hdf(os.path.join(destination_directory, 'summary.h5'), key='data', mode='w')
    if SAVE_CSV: summary_df.to_csv(os.path.join(destination_directory, 'summary.csv'), index=False)

    return summary_df



#
# Entry point, this is a one-time utility and long running process
#

def main():
    from src.utilities.myconfig import HARVEST_DIRECTORY, PROCESSED_DIRECTORY, REPROCESS_ALL

    argparser = argparse.ArgumentParser(description='Correlate EDF, scoring, and Somnofy files', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    argparser.add_argument('--source', '-s', type=str, default=HARVEST_DIRECTORY, help='Data directory to process')
    argparser.add_argument('--destination', '-d', type=str, default=PROCESSED_DIRECTORY, help='Destination directory to save results')
    argparser.add_argument('--require-somnofy', action='store_true', help='Require Somnofy files to be present')
    argparser.add_argument('--require-sleepstage', action='store_true', help='Require SleepStage files to be present')
    argparser.add_argument('--force', '-f', action='store_true', default=REPROCESS_ALL, help='Force reprocessing of all recordings')
    argparser.add_argument('--ignore', '-i', action='store_true', help='Ignore errors and skip to next recording')
    argparser.add_argument('--random', '-r', action='store_true', help='Shuffle the recordings before processing')
    argparser.add_argument('--dryrun', '-n', action='store_true', help='Dry run, don\'t actually write files')
    argparser.add_argument('--max-count', '-m', type=int, default=0, help='Maximum number of recordings to process')
    argparser.add_argument('--workers', '-w', type=int, default=8, help='Number of workers to use for processing')
    argparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (debug level)')
    argparser.add_argument('--very-verbose', '-vv', action='store_true', help='Very verbose output (debug level)')
    args = argparser.parse_args()

    # Set up logging
    if not args.verbose:
        logger.setLevel(logging.WARNING)

    if not args.very_verbose:
        vv_logger.setLevel(logging.CRITICAL)
        logging.getLogger('mytimer').setLevel(logging.CRITICAL)

    # Track the start time
    track_latest("correlate_start_time", datetime.now().isoformat())
    track_latest("max_count", args.max_count)
    track_latest("workers", args.workers)
    track_latest("require_somnofy", args.require_somnofy)
    track_latest("require_sleepstage", args.require_sleepstage)
    track_latest("ignore_errors", args.ignore)

    # Print some info
    print(f"* Sleep Study Pipeline - Correlator (chrifren@ifi.uio.no)")
    print(f"Reading, analyzing, calculating, aligning overnight sleep recordings into {args.destination}...")
    print(f"This will take a while... {get_life_tip()}\n")
    
    # Run the main processing
    with Timer(operation_name="Correlate", writelog=False) as timer:
        correlate_all_recordings(args.source, args.destination, args.force, args.ignore, args.dryrun, args.max_count, args.random, args.require_somnofy, args.require_sleepstage, args.workers, args.verbose)
        print(f"Correlation completed in {timer.get_duration():.2f}s")

    # Create a summary file with recording and essential metadata as a HDF5 and CSV file
    with Timer(operation_name="create_summary", writelog=False) as summary_timer:
        create_summary(args.destination, args.verbose)
        print(f"Created summary in {summary_timer.get_duration():.2f}s")

    track_latest("correlate_end_time", datetime.now().isoformat())

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)