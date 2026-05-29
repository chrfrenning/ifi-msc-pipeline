# #####################################################################
#
# Splitter
#
# Creates individual epochs from the overnight recordings. Each
# sample is a n second long sequence with a y-label indicating
# if there was an apnea event during that sequence, using
# a theta threshold level for number of scored seconds to
# trigger y=1 as a positive sample
#
# Copyright 2026 Christopher Frenning
# chrifren@ifi.uio.no, christopher@frenning.com
#
# Licensed under the MIT License - see LICENSE.md for details
#
# Please cite my thesis if you use this code in your own work.
#
# #####################################################################


import json
import os
import random
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from datetime import datetime

from src.utilities.mytimer import Timer
from src.utilities.mymetrics import track_count, track_series, track_latest


#
# Configuration, make changes in your local .env file
#

from src.utilities.myconfig import NORMALIZED_DIRECTORY, PROCESSED_DIRECTORY, EPOCHS_DIRECTORY, SLEEP_SCORE_COLUMNS
from src.utilities.myconfig import SAMPLING_FREQUENCY, SEQUENCE_LENGTH, Y_THETA, EPOCH_SIGNALS, SCORING_EVENTS


#
# Setup logging
#

from src.utilities.mylogging import create_logger
from src.utilities.mytips import get_life_tip
from src.utilities.myutilities import load_metadata

logger = create_logger("splitter")
vv_logger = create_logger("spitter_verbose")


#
# Normalization stats - we're bringing these on to the epochs file for lineage
#

# Column order for per-signal normalization stats tables (global_stats and per-recording signal_stats)
SIGNAL_STAT_TABLE_COLUMNS = (
    'raw_mean',
    'raw_std',
    'min',
    'max',
    'mean',
    'std',
    'median',
    'n_samples',
    'n_outliers_removed',
)


def signals_dict_to_df(signal_stats):
    rows = [
        {'signal': name, **{c: stats.get(c) for c in SIGNAL_STAT_TABLE_COLUMNS}}
        for name, stats in sorted(signal_stats.items())
    ]

    return pd.DataFrame(rows)[['signal'] + list(SIGNAL_STAT_TABLE_COLUMNS)]


def write_recording_signal_stats_table(h5_path, recording_group, signal_stats_dict):
    df = signals_dict_to_df(signal_stats_dict)
    if df is None:
        return
    df.to_hdf(h5_path, key=f'{recording_group}/signal_stats', mode='a')


def add_metadata_to_h5_recording(h5_path, recording_group, metadata_dict):
    payload = json.dumps(metadata_dict, indent=2, sort_keys=True, default=str).encode('utf-8')
    with h5py.File(h5_path, 'a') as f:
        grp = f[recording_group]
        grp.attrs['metadata_json'] = payload


def add_metadata_to_h5(h5_path, recording_group, recording_directory):
    metadata = load_metadata(recording_directory)
    add_metadata_to_h5_recording(h5_path, recording_group, metadata)

    signal_stats = metadata.get('normalizer', {}).get('signal_stats')
    write_recording_signal_stats_table(h5_path, recording_group, signal_stats)



#
# Create epochs from an overnight recording
#

def create_epochs(source_directory, destination_file, sequence_length, theta, scoring_events, trim_sleep_stages, signals, max_count, shuffle, dryrun, verbose, ignore):
    recordings = os.listdir(source_directory)
    recordings = [d for d in recordings if os.path.isdir(os.path.join(source_directory, d))]
    recordings = sorted(recordings, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    if shuffle: random.shuffle(recordings)
    recordings = recordings[:max_count] if max_count > 0 else recordings
    recordings = [os.path.join(source_directory, recording) for recording in recordings]

    if not dryrun:
        os.makedirs(os.path.dirname(os.path.abspath(destination_file)), exist_ok=True)

    first_write = True
    total_recordings = 0
    total_epochs = 0
    positive_epochs = 0
    negative_epochs = 0

    for recording_directory in tqdm(recordings, desc="Recordings", disable=verbose):
        logger.debug(f"Processing recording {recording_directory}")

        track_count("recordings_count")
        total_recordings += 1

        recording_df = None
        recording_file = os.path.join(recording_directory, f'normalized_{SAMPLING_FREQUENCY}.h5')
        if os.path.exists(recording_file):
            with Timer(operation_name="ReadHDF", writelog=False):
                recording_df = pd.read_hdf(recording_file)
        else:
            recording_file = os.path.join(recording_directory, f'normalized_{SAMPLING_FREQUENCY}.csv')
            if os.path.exists(recording_file):
                with Timer(operation_name="ReadCSV", writelog=False):
                    recording_df = pd.read_csv(recording_file)
            else:
                if ignore:
                    logger.error(f"Normalized file not found for recording {recording_directory}")
                    continue
                else:
                    raise FileNotFoundError(f"Normalized file not found for recording {recording_directory}")
        
        if trim_sleep_stages:
            logger.debug(f"Rows in recording {recording_directory} before trimming: {len(recording_df)}")

            present = [c for c in SLEEP_SCORE_COLUMNS if c in recording_df.columns]
            if present:
                recording_df = recording_df[recording_df[present].eq(1).any(axis=1)]

            logger.debug(f"Rows in recording {recording_directory} after trimming: {len(recording_df)}")

        # Create epochs, padding the last one to sequence_length if needed
        epochs = []
        labels = []
        for i in tqdm(range(0, len(recording_df), sequence_length), total=len(recording_df)//sequence_length, desc=f"  Creating epochs", leave=False, disable=verbose):
            epoch = recording_df.iloc[i:i+sequence_length]
            if len(epoch) < sequence_length:
                pad_rows = sequence_length - len(epoch)
                pad_df = pd.DataFrame(0, index=range(pad_rows), columns=epoch.columns)
                last_ts = pd.to_datetime(epoch['timestamp'].iloc[-1])
                pad_df['timestamp'] = [last_ts + pd.Timedelta(seconds=s+1) for s in range(pad_rows)]
                epoch = pd.concat([epoch, pad_df], ignore_index=True)
                logger.debug(f"Padded last epoch of {recording_directory} with {pad_rows} zero rows")
            epochs.append(epoch)

        if not epochs:
            logger.debug(f"No epochs created for {recording_directory}, skipping")
            continue

        # Calculate y-label
        for epoch in tqdm(epochs, total=len(epochs), desc=f"  Labelling", leave=False, disable=verbose):
            present = [c for c in scoring_events if c in epoch.columns]
            # count the number of seconds in the sequence where any scoring column is 1
            classified_seconds = epoch[present].any(axis=1).sum()
            y = 1 if classified_seconds >= theta else 0
            labels.append(y)

            track_series("apnea_seconds", int(classified_seconds))
            if y == 1: track_count("positive_epochs"); positive_epochs += 1
            if y == 0: track_count("negative_epochs"); negative_epochs += 1
            
        # Save the epochs and labels
        if not dryrun:
            recording_id = os.path.basename(recording_directory)

            # Filter to requested signals (always keep timestamp)
            keep_cols = None
            if signals:
                keep_cols = ['timestamp'] + [s for s in signals if s in epochs[0].columns]

            for i, epoch in tqdm(enumerate(epochs), total=len(epochs), desc=f"  Saving", leave=False, disable=verbose):
                out = epoch[keep_cols] if keep_cols else epoch
                key = f'r{recording_id:>04}/e{i:06d}_{labels[i]}'
                mode = 'w' if first_write else 'a'
                out.to_hdf(destination_file, key=key, mode=mode)
                first_write = False
                total_epochs += 1

            recording_group = f'r{recording_id:>04}'
            add_metadata_to_h5(destination_file, recording_group, recording_directory)

    # Store run parameters and counts as attributes on the root group
    if not dryrun and os.path.exists(destination_file):
        # Embed the recording summary from the correlation step
        summary_h5 = os.path.join(source_directory, 'summary.h5')
        summary_csv = os.path.join(source_directory, 'summary.csv')
        if os.path.exists(summary_h5):
            pd.read_hdf(summary_h5).to_hdf(destination_file, key='summary', mode='a')
        elif os.path.exists(summary_csv):
            pd.read_csv(summary_csv).to_hdf(destination_file, key='summary', mode='a')
        else:
            logger.warning(f"No summary file found in {PROCESSED_DIRECTORY}")

        # Load the global normalization stats
        global_stats_file = os.path.join(source_directory, 'global_stats.json')
        global_stats_raw = None
        global_stats_dict = None
        if os.path.exists(global_stats_file):
            with open(global_stats_file, 'r') as gs:
                global_stats_raw = gs.read()
            global_stats_dict = json.loads(global_stats_raw)
        else:
            logger.warning(f"No global_stats.json found in {source_directory}")

        # Write attributes to the destination file
        with h5py.File(destination_file, 'a') as f:
            f.attrs['sequence_length'] = sequence_length
            f.attrs['theta'] = theta
            f.attrs['trim_sleep_stages'] = str(trim_sleep_stages) # fix pytables limitation on bool
            f.attrs['total_recordings'] = total_recordings
            f.attrs['total_epochs'] = total_epochs
            f.attrs['positive_epochs'] = positive_epochs
            f.attrs['negative_epochs'] = negative_epochs
            f.attrs['signals'] = ','.join(signals) if signals else '*'

            # Embed the global normalization stats as a JSON blob (unchanged)
            if global_stats_raw is not None:
                f.attrs['global_stats'] = global_stats_raw

        # Add a table with normalization stats
        if global_stats_dict is not None:
            rows = [
                {'signal': name, **{c: stats.get(c) for c in SIGNAL_STAT_TABLE_COLUMNS}}
                for name, stats in sorted(global_stats_dict.items())
            ]
            pd.DataFrame(rows)[['signal'] + list(SIGNAL_STAT_TABLE_COLUMNS)].to_hdf(
                destination_file, key='global_stats', mode='a',
            )

#
# Main entry point
#

def main():
    argparser = argparse.ArgumentParser(description='Split all recordings into epochs', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    argparser.add_argument('--source', '-s', type=str, default=NORMALIZED_DIRECTORY, help='Data directory to process')
    argparser.add_argument('--destination', '-d', type=str, default=os.path.join(EPOCHS_DIRECTORY, 'epochs.h5'), help='Destination HDF5 file for all epochs')
    argparser.add_argument('--sequence-length', '-l', type=int, default=SEQUENCE_LENGTH, help='Length of the sequence in seconds')
    argparser.add_argument('--theta', '-t', type=int, default=Y_THETA, help='Threshold for number of scored seconds to trigger y=1 as a positive sample')
    argparser.add_argument('--scoring-events', type=str, default=None, help='Comma-separated list of scoring events to include (default: from .env)')
    argparser.add_argument('--trim-sleep-stages', action='store_true', help='Trim recordings to sleep periods only')
    argparser.add_argument('--signals', type=str, default=EPOCH_SIGNALS, help='Comma-separated list of signal columns to include (* for all, default: from .env)')
    argparser.add_argument('--max-count', '-m', type=int, default=0, help='Maximum number of recordings to process')
    argparser.add_argument('--random', '-r', action='store_true', help='Randomize the order of the recordings')
    argparser.add_argument('--dryrun', '-n', action='store_true', help='Dry run, don\'t actually write files')
    argparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (debug level)')
    argparser.add_argument('--very-verbose', '-vv', action='store_true', help='Very verbose output (debug level)')
    argparser.add_argument('--ignore', '-i', action='store_true', help='Ignore errors and skip to next recording')
    args = argparser.parse_args()

    # Print some info
    print(f"* Sleep Study Pipeline - Splitter (chrifren@ifi.uio.no)")
    print(f"Splitting overnight recordings into epochs of {args.sequence_length} seconds...")
    print(f"This will take a while... {get_life_tip()}\n")

    # Parse signals list
    if args.signals and args.signals.strip() != '*':
        args.signals = [s.strip() for s in args.signals.split(',') if s.strip()]
    else:
        args.signals = None

    # Parse scoring events list
    if args.scoring_events:
        args.scoring_events = [s.strip() for s in args.scoring_events.split(',') if s.strip()]
    else:
        args.scoring_events = SCORING_EVENTS

    # Warning on trimming
    if args.trim_sleep_stages:
        logger.warning("Trimming to sleep stages should ideally be done in the normalization step, not here! (It will affect normalization quality.)")

    # Set up logging
    if not args.verbose:
        logger.setLevel(logging.INFO)

    if not args.very_verbose:
        vv_logger.setLevel(logging.CRITICAL)
        logging.getLogger('mytimer').setLevel(logging.CRITICAL)

    # Track configuration
    track_latest("split_start_time", datetime.now().isoformat())
    track_latest("source", os.path.abspath(args.source))
    track_latest("max_count", args.max_count)
    track_latest("random", args.random)
    track_latest("sequence_length", args.sequence_length)
    track_latest("trim_sleep_stages", args.trim_sleep_stages)
    track_latest("theta", args.theta)
    track_latest("scoring_events", ','.join(args.scoring_events) if args.scoring_events else '*')
    track_latest("signals", ','.join(args.signals) if args.signals else '*')

    # Run the main processing
    with Timer(operation_name="Plot", writelog=True):
        create_epochs(args.source, args.destination, args.sequence_length, args.theta, args.scoring_events, args.trim_sleep_stages, args.signals, args.max_count, args.random, args.dryrun, args.verbose, args.ignore)

    # Track how much of the positive class
    from src.utilities.mymetrics import _metrics
    epsilon = 10e-10
    percentage_positive = _metrics.get("positive_epochs", 0).series[0]/_metrics.get("negative_epochs", epsilon).series[0]
    track_latest("percentage_positive", _metrics["positive_epochs"].series[0]/_metrics["negative_epochs"].series[0])

    # Goodbye
    track_latest("split_end_time", datetime.now().isoformat())

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)