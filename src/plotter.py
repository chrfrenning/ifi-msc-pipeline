# #####################################################################
#
# Plotter
#
# Uses plotly (https://plotly.com/python/) to create static html 
# pages with plots of the normalized recordings, showing apnea
# events, sleep staging from PSG and VitalThings and all signals
# derived through our pipeline
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
import argparse
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.utilities.mymetrics import track_count, track_latest, track_sum
from src.utilities.mytimer import Timer


#
# Configuration, make changes in your local .env file
#

from src.utilities.myconfig import NORMALIZED_DIRECTORY, SIGNALS_TO_NORMALIZE
from src.utilities.myconfig import SAMPLING_FREQUENCY


#
# Setup logging
#

from src.utilities.mylogging import create_logger
from src.utilities.mytips import get_life_tip
from src.utilities.myutilities import load_metadata

logger = create_logger("plotter")
vv_logger = create_logger("plotter_verbose")



#
# Scoring column definitions for annotation panels
#

# PSG sleep staging
SLEEP_STAGE_COLUMNS = {
    'scoring_sleep-s0': ('Wake', 5),
    'scoring_sleep-rem': ('REM', 4),
    'scoring_sleep-s1': ('N1', 3),
    'scoring_sleep-s2': ('N2', 2),
    'scoring_sleep-s3': ('N3', 1),
}

# PSG apnea event scoring
APNEA_EVENT_COLUMNS = {
    'scoring_apnea-obstructive': ('Obstructive', 'rgba(220, 50, 50, 0.5)'),
    'scoring_apnea-central': ('Central', 'rgba(50, 50, 220, 0.5)'),
    'scoring_apnea-mixed': ('Mixed', 'rgba(220, 140, 30, 0.5)'),
    'scoring_hypopnea': ('Hypopnea', 'rgba(160, 50, 200, 0.5)'),
    'scoring_desat': ('Desaturation', 'rgba(40, 160, 160, 0.5)'),
}

# VitalThings sleep staging: 0=Unscored, 1=Deep, 2=Light, 3=REM, 4=Wake
VT_SLEEP_STAGE_COLUMN = 'vtss_sleep_stage'
VT_STAGE_TICKVALS = [1, 2, 3, 4]
VT_STAGE_TICKTEXT = ['Deep', 'Light', 'REM', 'Wake']



#
# Main plot funciton - creates jumbo html file with all signals and plots
#

def plot_single_recording(recording, destination_directory, flatten):
    track_count("plot_single_recording_count")
    
    output_filename = os.path.join(destination_directory, os.path.basename(recording), f"plot_{SAMPLING_FREQUENCY}.html")
    if flatten: output_filename = os.path.join(destination_directory, f"plot_{os.path.basename(recording)}_{SAMPLING_FREQUENCY}.html")
    logger.debug(f"Processing recording {recording} -> {output_filename}")

    dataset = None
    metadata = load_metadata(recording)
    dataset_filename = os.path.join(recording, f'normalized_{SAMPLING_FREQUENCY}.h5')
    if os.path.exists(dataset_filename):
        with Timer(operation_name="ReadHDF", writelog=False):
            dataset = pd.read_hdf(dataset_filename)
    else:
        dataset_filename = os.path.join(recording, f'normalized_{SAMPLING_FREQUENCY}.csv')
        if os.path.exists(dataset_filename):
            with Timer(operation_name="ReadCSV", writelog=False):
                dataset = pd.read_csv(dataset_filename)
        else:
            raise FileNotFoundError(f"Normalized dataset not found in {recording}")

    # Determine which annotation panels to show based on available columns
    present_apnea_events = {c: v for c, v in APNEA_EVENT_COLUMNS.items() if c in dataset.columns}
    present_sleep_staging = {c: v for c, v in SLEEP_STAGE_COLUMNS.items() if c in dataset.columns}
    has_apnea = len(present_apnea_events) > 0
    has_sleep = len(present_sleep_staging) > 0
    has_vt_sleep = VT_SLEEP_STAGE_COLUMN in dataset.columns and dataset[VT_SLEEP_STAGE_COLUMN].nunique() > 1

    # Subplot layout
    n_rows = 1
    row_heights = [0.55]
    if has_apnea:
        n_rows += 1
        row_heights.append(0.15)
    if has_sleep:
        n_rows += 1
        row_heights.append(0.15)
    if has_vt_sleep:
        n_rows += 1
        row_heights.append(0.15)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    # Signal traces
    for signal in dataset.columns:
        if any([signal.startswith(signal_name) for signal_name in SIGNALS_TO_NORMALIZE]):
            fig.add_trace(go.Scatter(x=dataset['timestamp'], y=dataset[signal], name=signal), row=1, col=1)
    fig.update_yaxes(title_text="Signals", row=1, col=1)

    # Apnea events plot
    current_row = 2
    if has_apnea:
        for idx, (col, (label, color)) in enumerate(present_apnea_events.items()):
            level = idx
            base = np.full(len(dataset), level, dtype=float)
            top = np.where(dataset[col].values == 1, level + 0.8, level)
            fig.add_trace(go.Scatter(
                x=dataset['timestamp'], y=base,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ), row=current_row, col=1)
            fig.add_trace(go.Scatter(
                x=dataset['timestamp'], y=top,
                fill='tonexty', fillcolor=color,
                mode='lines', line=dict(width=0),
                name=label,
            ), row=current_row, col=1)

        fig.update_yaxes(
            title_text="Events",
            tickvals=[i + 0.4 for i in range(len(present_apnea_events))],
            ticktext=[v[0] for v in present_apnea_events.values()],
            row=current_row, col=1,
        )
        current_row += 1

    # PSG sleep stage
    if has_sleep:
        hypno = np.zeros(len(dataset))
        for col, (label, val) in present_sleep_staging.items():
            hypno = np.where(dataset[col].values == 1, val, hypno)

        fig.add_trace(go.Scatter(
            x=dataset['timestamp'], y=hypno,
            mode='lines', line=dict(shape='hv', width=1.5, color='rgb(70, 70, 160)'),
            name='PSG Stage',
            fill='tozeroy', fillcolor='rgba(100, 100, 200, 0.1)',
        ), row=current_row, col=1)

        fig.update_yaxes(
            title_text="PSG",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['N3', 'N2', 'N1', 'REM', 'Wake'],
            range=[0.5, 5.5],
            row=current_row, col=1,
        )
        current_row += 1

    # VitalThings sleep stage hypnogram
    if has_vt_sleep:
        vt_values = dataset[VT_SLEEP_STAGE_COLUMN].values.astype(float)
        vt_values[vt_values == 0] = np.nan

        fig.add_trace(go.Scatter(
            x=dataset['timestamp'], y=vt_values,
            mode='lines', line=dict(shape='hv', width=1.5, color='rgb(160, 70, 70)'),
            name='VT Stage',
            fill='tozeroy', fillcolor='rgba(200, 100, 100, 0.1)',
            connectgaps=False,
        ), row=current_row, col=1)

        fig.update_yaxes(
            title_text="VT",
            tickvals=VT_STAGE_TICKVALS,
            ticktext=VT_STAGE_TICKTEXT,
            range=[0.5, 4.5],
            row=current_row, col=1,
        )

    recording_id = metadata.get('recording_id', os.path.basename(recording))
    fig.update_layout(
        title=f"Recording {recording_id}",
        height=300 + (n_rows * 250),
        hovermode='x unified',
        legend=dict(
            yanchor='top', y=1, xanchor='left', x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
        ),
        margin=dict(r=200),
    )

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.write_html(output_filename)

    # calculate disk space used by the recording
    disk_space_used = os.path.getsize(output_filename)
    track_sum("disk_space_used", disk_space_used)



#
# Main process loop
#

def plot_recordings(source_directory, destination_directory, flatten, ignore_errors, max_count, verbose):
    # List all recordings from the source dir
    all_recordings = list(os.listdir(source_directory))
    all_recordings = [os.path.join(source_directory, recording) for recording in sorted(all_recordings)]

    # While testing, useful to limit the number of recs to process
    if max_count > 0:
        random.shuffle(all_recordings)
        all_recordings = all_recordings[:max_count]

    # Loop through recording by recording and make static html plots
    for recording in tqdm(all_recordings, desc="Plotting recordings", unit="rec", disable=verbose):
        if not os.path.isdir(recording):
            continue

        try:
            with Timer(operation_name="PlotRecording", writelog=False):
                plot_single_recording(recording, destination_directory, flatten)
        except Exception as e:
            logger.error(f"Error processing recording {recording}: {e}")
            if not ignore_errors:
                raise



#
# Entry point
#

def main():
    PLOT_DIRECTORY = NORMALIZED_DIRECTORY

    argparser = argparse.ArgumentParser(description='Plot all normalized recordings', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    argparser.add_argument('--source', '-s', type=str, default=NORMALIZED_DIRECTORY, help='Data directory to process')
    argparser.add_argument('--destination', '-d', type=str, default=PLOT_DIRECTORY, help='Destination directory to save results')
    argparser.add_argument('--flatten', action='store_true', help='Flatten the recordings into a single directory')
    argparser.add_argument('--ignore', '-i', action='store_true', help='Ignore errors and skip to next recording')
    argparser.add_argument('--max-count', '-m', type=int, default=0, help='Maximum number of recordings to process')
    argparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (debug level)')
    argparser.add_argument('--very-verbose', '-vv', action='store_true', help='Very verbose output (debug level)')
    args = argparser.parse_args()

    # Set up logging
    if not args.verbose:
        logger.setLevel(logging.WARNING)

    if not args.very_verbose:
        vv_logger.setLevel(logging.CRITICAL)
        logging.getLogger('mytimer').setLevel(logging.CRITICAL)

    track_latest("plot_start_time", datetime.now().isoformat())
    track_latest("max_count", args.max_count)
    track_latest("ignore_errors", args.ignore)
    track_latest("flatten", args.flatten)

    # Print some info
    print(f"* Sleep Study Pipeline - Plotter (chrifren@ifi.uio.no)")
    print(f"Plotting all signal data into {args.destination}...")
    print(f"This will take a while... {get_life_tip()}\n")

    with Timer(operation_name="Plot", writelog=True):
        plot_recordings(args.source, args.destination, args.flatten, args.ignore, args.max_count, args.verbose)

    track_latest("plot_end_time", datetime.now().isoformat())

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)