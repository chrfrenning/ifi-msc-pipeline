# #####################################################################
#
# Harvester
#
# First step in our process is to gather files from the same
# overnight recording. Our data sources are PSG recording,
# scoring file, somnofy recording, and somnofy sleep staging file
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
import re
import sys
import json
import shutil
import random
import logging
import pyedflib
import argparse
import threading
import traceback
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utilities.mytips import get_life_tip
from src.utilities.myutilities import guard_dos_newlines, convert_to_strings, read_start_of_file, read_end_of_file
from src.utilities.myutilities import is_between, is_overlapping_time_ranges
from src.utilities.myutilities import dp, bn
from src.utilities.mytimer import Timer
from src.utilities.mymetrics import track_count, track_latest, track_sum

#
# Configuration, make changes in your local .env file
#

from src.utilities.mylogging import create_logger

logger = create_logger("harvester")
vv_logger = create_logger("harvester_verbose") # very verbose logger



#
# Helper functions to extract key data from our source files
#

def read_scoring_file(filename):
    header = read_start_of_file(filename)
    footer = read_end_of_file(filename)

    # Find "Events Included:" and extract all data until a double newline
    events = []
    match = re.search(r'Events Included:(.*?)(?:\n\s*\n|$)', header, re.DOTALL)
    if match:
        events_data = match.group(1).strip()
        # Split by newlines and filter out empty lines
        events = [line.strip() for line in events_data.split('\n') if line.strip()]

    # Find first timestamp in header after "Time [hh:mm:ss.xxx]" line
    first_ts = None
    # Look for the header line followed by timestamp data
    match = re.search(r'Time \[hh:mm:ss\.xxx\]\s+Event\s+Duration\[s\]\s*\n(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', header, re.MULTILINE)
    if match:
        timestamp_str = match.group(1)
        try:
            # Parse the ISO format timestamp
            first_ts = datetime.fromisoformat(timestamp_str)
        except ValueError:
            logger.warning(f"Could not parse first timestamp: {timestamp_str}")
            first_ts = None

    # Find last timestamp in footer
    last_ts = None
    # Find all timestamps in the footer and get the last one
    timestamps = re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', footer)
    if timestamps:
        try:
            # Parse the last timestamp
            last_ts = datetime.fromisoformat(timestamps[-1])
        except ValueError:
            logger.warning(f"Could not parse last timestamp: {timestamps[-1]}")
            last_ts = None

    # the patient id is the last word that is not 'EVENT'
    filename_without_extension = "".join(os.path.basename(filename).split('.')[:-1])
    parts = re.split(r'[.\- ]', filename_without_extension)
    filtered_parts = [p for p in parts if p.strip() and not p.upper().startswith('EVENT')]
    patient_id = filtered_parts[-1] if filtered_parts else ''

    return {
        'events': events,
        'first_ts': first_ts,
        'last_ts': last_ts,
        'patient_id': patient_id
    }

def read_edf_metadata(filename):
    with Timer(operation_name="read_edf_metadata", writelog=False) as timer:
        try:
            with pyedflib.EdfReader(filename) as reader:
                return {
                    'patient_id': reader.getPatientCode(),
                    'first_ts': reader.getStartdatetime(),
                    'last_ts': reader.getStartdatetime() + timedelta(seconds=reader.file_duration),
                    'num_channels': reader.signals_in_file
                }
        except Exception as e:
            logger.error(f"Error reading '{filename}': {e}")
            track_count("read_edf_metadata_error")
            return None

def get_somnofy_metadata(filename):
    header = read_start_of_file(filename)
    footer = read_end_of_file(filename)

    # find the first timestamp in the header, this is on line 2, until first comma
    start_ts = None
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', header.split('\n')[1])
    if match:
        start_ts = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')

    # find the last timestamp in the footer, this is on the last line, until first comma
    end_ts = None
    lines = footer.split('\n')
    while lines[-1] == '':
        lines.pop()
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', lines[-1])
    if match:
        end_ts = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')

    return {
        'first_ts': start_ts,
        'last_ts': end_ts
    }

def get_first_line(text_block):
    lines = text_block.split('\n')
    for line in lines:
        if line.startswith('#'):
            continue
        if line.startswith('timestamp'):
            continue
        return line

    raise ValueError(f"No data line found in {text_block}")

def get_last_line(text_block):
    lines = text_block.split('\n')
    lines.reverse()
    for line in lines:
        if not line.strip():
            continue
        if line.startswith('#'):
            continue
        return line

    raise ValueError(f"No data line found in {text_block}")

def extract_timestamp(line):
    return datetime.strptime(line.split(',')[0], '%Y-%m-%d %H:%M:%S.%f%z')

def get_sleepstage_metadata(filename):
    ''' These are pure CSV files, with timestamp,sleep_stage columns '''
    header = read_start_of_file(filename)
    footer = read_end_of_file(filename)

    # find the first timestamp in the header, this is on line 2, until first comma
    first_ts = extract_timestamp(get_first_line(header))
    last_ts = extract_timestamp(get_last_line(footer))

    return {
        'first_ts': first_ts,
        'last_ts': last_ts
    }


#
# Key functions to harvest all source data for an overnight recording
#

def harvest(filename, scoring_file_index, somnofy_source, sleepstage_file_index, report_state):
    track_count("harvest_start")

    # Open the EDF to read metadata
    metadata = read_edf_metadata(filename)
    if not metadata:
        logger.warning(f"Skipping {os.path.basename(filename)} because it is not a valid EDF file")
        report_state['error_invalid_edf'].append(filename)
        return None
    else:
        #logger.debug(f"EDF metadata for {os.path.basename(filename)}: {metadata}")
        pass

    # Find the recording ID in the filename, an integer
    match = re.search(r'\d+', os.path.basename(filename))
    if not match:
        logger.warning(f"No ID found in {filename}, skipping this recording.")
        report_state['error_no_id'].append(filename)
        return None
    
    # Save the recording ID to the metadata
    recording_id = int(match.group())
    metadata['recording_id'] = recording_id
    logger.debug(f"Recording ID: {recording_id}, source: {bn(filename)}")

    # Inform about start and end times
    first_date = metadata['first_ts'].date()
    start_time = metadata['first_ts'].strftime('%H:%M:%S')
    end_time = metadata['last_ts'].strftime('%H:%M:%S')
    logger.debug(f"Date: {first_date}, start time: {start_time}, end time: {end_time}")


    #
    # Find the scoring files for this recording
    #

    num_matches = len(scoring_file_index[int(match.group())])
    if num_matches == 0:
        logger.warning(f"No matching scoring files found for {filename}, skipping this recording.")
        report_state['error_no_scoring_file'].append(filename)
        return None

    scoring_files = []
    for scoring_file in scoring_file_index[int(match.group())]:
        scoring_metadata = read_scoring_file(scoring_file)
        logger.debug(f"LDS metadata for {bn(scoring_file)} with {len(scoring_metadata['events'])} events")
        scoring_files.append((scoring_file, scoring_metadata))

    # Pick the metadata file with the most events
    if len(scoring_files) > 1:
        logger.warning(f"Multiple scoring files found for {filename}, using the one with the most events")
        report_state['warn_multiple_scoring_files'].append(filename)
        
        # Sort by number of events, highest first
        scoring_files.sort(key=lambda x: len(x[1]['events']), reverse=True)

        logger.debug(f"Scoring: {scoring_files[0][0]}, patient ID: {scoring_files[0][1]['patient_id']}")
        if not metadata['patient_id']:
            metadata['patient_id'] = scoring_files[0][1]['patient_id']
        metadata['scoring_metadata'] = scoring_files[0][1]
    else:
        logger.debug(f"Single scoring file found for {bn(filename)}, using it")
        metadata['scoring_metadata'] = scoring_files[0][1]


    #
    # Find matching Somnofy files
    #

    first_date = metadata['first_ts'].date()
    last_date = metadata['last_ts'].date()
    all_dates = [first_date + timedelta(days=i) for i in range((last_date - first_date).days + 1)]

    logger.debug(f"All dates: {[d.strftime('%Y-%m-%d') for d in all_dates]}")
    if len(all_dates) < 2:
        logger.warning(f"Less than 2 dates found for {filename}, this is not normal for an overnight recording")
    elif len(all_dates) > 2:
        logger.warning(f"More than 2 dates found for {filename}, this is not normal for an overnight recording")

    # Gather the matching Somnofy files
    somnofy_matches = []
    for date in all_dates:
        somnofy_path = os.path.join(somnofy_source, f"{date.year}{date.month:02d}{date.day:02d}/{scoring_files[0][1]['patient_id'].lower()}")
        vv_logger.debug(f"Somnofy path: {somnofy_path} { '(exists)' if os.path.exists(somnofy_path) else '(does not exist)' }")
        if os.path.exists(somnofy_path):
            # list somnofy files in the directory
            somnofy_files = os.listdir(somnofy_path)
            for somnofy_file in somnofy_files:
                somnofy_metadata = get_somnofy_metadata(os.path.join(somnofy_path, somnofy_file))
                is_overlapping = is_overlapping_time_ranges(somnofy_metadata['first_ts'], somnofy_metadata['last_ts'], metadata['first_ts'], metadata['last_ts'])
                vv_logger.debug(f"Somnofy {somnofy_path}: { 'overlapping' if is_overlapping else 'not overlapping' }")                    
                if is_overlapping:
                    logger.debug(f"Somnofy {dp(somnofy_path, somnofy_source)} is overlapping with {bn(filename)}")
                    somnofy_matches.append(os.path.join(somnofy_path, somnofy_file))
                    if 'somnofy_metadata' not in metadata:
                        metadata['somnofy_metadata'] = []
                    metadata['somnofy_metadata'].append({
                        'file': somnofy_file,
                        'metadata': somnofy_metadata
                        })


    #
    # Find somnofy sleep staging file
    #

    sleepstage_matches = []
    for somnofy_file in somnofy_matches:
        id = os.path.splitext(os.path.basename(somnofy_file))[0]
        for sleepstage_file in sleepstage_file_index[id]:
            sleepstage_matches.append(sleepstage_file)

            if 'somnofy_sleepstage' not in metadata:
                metadata['somnofy_sleepstage'] = []
            metadata['somnofy_sleepstage'].append({
                'file': os.path.basename(sleepstage_file),
                'metadata': get_sleepstage_metadata(sleepstage_file)
            })


    #
    # We're done, return all files and metadata
    #

    track_count("harvest_success")
    report_state['success'].append((filename, recording_id))
    return recording_id, metadata, scoring_files[0][0], somnofy_matches, sleepstage_matches

def index_scoring_files(source_directory : str) -> dict:
    ''' Enumerate all text files and index them by recording ID
        Returns a dictionary with recording ID as key and list of text files as value '''

    all_files = os.listdir(source_directory)

    directory = defaultdict(list)
    for file in all_files:
        if os.path.splitext(file)[1].upper() == '.TXT':
            match = re.search(r'\d+', file)
            if match:
                vv_logger.debug(f"Scoring file: {file}, ID: {match.group()}")
                directory[int(match.group())].append(os.path.join(source_directory, file))

    logger.info(f"Indexed {len(directory)} scoring files")
    return directory

def index_sleepstage_files(source_directory : str) -> dict:
    ''' Enumerate all sleep staging files and index them by recording ID
        Returns a dictionary with recording ID as key and list of sleep staging files as value '''

    all_files = os.listdir(source_directory)
    directory = defaultdict(list)
    for file in all_files:
        name, ext = os.path.splitext(file)
        # skip some files we don't need
        if 'registry' in name.lower() or '~' in name or name.startswith('.'):
            continue

        if ext.upper() == '.CSV':
            # filename is like 'sleep_stages_027_TkNVQxgIDBMzAwAA.csv'
            id = name.split('_')[-1]
            vv_logger.debug(f"Sleep stage file: {file}, ID: {id}")
            directory[id].append(os.path.join(source_directory, file))

    logger.info(f"Indexed {len(directory)} sleep stage files")
    return directory
            

def gather_files(recording_id, metadata, edf_file, scoring_file, somnofy_files, sleepstage_files, destination_directory):
    with Timer(operation_name="gather_files", writelog=False) as timer:
        recording_dir = os.path.join(destination_directory, str(recording_id))
        os.makedirs(recording_dir, exist_ok=True)

        shutil.copy(edf_file, os.path.join(recording_dir, os.path.basename(edf_file)))
        shutil.copy(scoring_file, os.path.join(recording_dir, os.path.basename(scoring_file)))
        num_files_copied = 2
        
        for somnofy_file in somnofy_files:
            shutil.copy(somnofy_file, os.path.join(recording_dir, os.path.basename(somnofy_file)))
            num_files_copied += 1

        for sleepstage_file in sleepstage_files:
            shutil.copy(sleepstage_file, os.path.join(recording_dir, os.path.basename(sleepstage_file)))
            num_files_copied += 1

        with open(os.path.join(recording_dir, 'metadata.json'), 'w') as f:
            metadata_as_strings = convert_to_strings(metadata)
            json.dump(metadata_as_strings, f, indent=4, sort_keys=True)
            num_files_copied += 1

        logger.info(f"{num_files_copied} files copied to {recording_dir}")


# #####################################################################
#
# Harvest and gather
#
# Our Primary Key (so to speak) is the recording ID of the EDF files
# No EDF file - nothing else matters!
#
# We match it with the scoring TXT file based on the ID
# We extract start and end time to match with Somnofy recordings
# We extract the Somnofy ID to match with the sleep staging file
#
#

def harvest_single_recording(args, scoring_file_index, sleepstage_file_index, file, report_state, max_limit):
    # Bail if we already reached the max job limit (for testing)
    with max_limit['lock']:
        if max_limit['remaining'] <= 0:
            return

    logger.info(f"Harvesting {file}")
    track_count("harvest_single_recording_start")

    if os.path.splitext(file)[1].upper() != '.EDF':
        return

    # Ignore 0-byte files, we have some duds in the source dataset
    if os.path.getsize(os.path.join(args.psg, file)) == 0:
        logger.warning(f"Skipping {file} because it is a 0-byte file")
        return
    
    try:

        logger.info(f"Harvesting {file}")
        full_file_path = os.path.join(args.psg, file)

        ret = harvest(full_file_path, scoring_file_index, args.somnofy, sleepstage_file_index, report_state)
        if ret:
            # Abort if another thread took the last slot
            with max_limit['lock']:
                if max_limit['remaining'] <= 0:
                    return
                max_limit['remaining'] -= 1

            if not args.dryrun:
                recording_id, metadata, scoring_file, somnofy_matches, sleepstage_matches = ret
                gather_files(recording_id, metadata, full_file_path, scoring_file, somnofy_matches, sleepstage_matches, args.destination)

            track_count("harvest_single_recording_success")

            # calculate disk space used by the recording
            disk_space_used = os.path.getsize(os.path.join(args.destination, str(recording_id)))
            track_sum("disk_space_used", disk_space_used)

    except Exception as e:
        if not args.ignore:
            traceback.print_exc()
            sys.exit(1)
        else:
            logger.error(f"Skipping recording '{file}' due to error ({e})")
            return

def harvest_all_recordings(args):
    # First make sure we have quick lookup of all scoring TXT files
    scoring_file_index = index_scoring_files(args.psg)
    sleepstage_file_index = index_sleepstage_files(args.sleepstage)


    # Find all EDF files, we will iterate over them to gather all data for
    # each overnight recording they represent. The PSG machine will be on
    # for the full night.

    all_files = os.listdir(args.psg)
    if args.random:
        random.shuffle(all_files) # randomize for testing to meet problems faster while in dev
    else:
        all_files.sort()

    # Thread-safe counter so workers stop after max_count successful harvests
    args.max_count = args.max_count if args.max_count > 0 else len(all_files)
    max_limit = {'remaining': args.max_count, 'lock': threading.Lock()}

    # Now, for each EDF file, find the related files from the
    # other directories

    report_state = defaultdict(list)

    concurrent_harvest = ThreadPoolExecutor(max_workers=args.workers)
    futures = [concurrent_harvest.submit(harvest_single_recording, args, scoring_file_index, sleepstage_file_index, file, report_state, max_limit) for file in all_files]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Harvesting", disable=args.verbose):
        future.result()
    concurrent_harvest.shutdown(wait=True)

    # Write the error state to a file
    with open('harvester_report.json', 'w') as f:
        json.dump(report_state, f, indent=4, sort_keys=True)


#
# Entry point to one-time utility, somewhat long running process
# depending on IO speed of the underlying storage devices
#

def main():
    from src.utilities.myconfig import LDS_SOURCE, SOMNOFY_SOURCE, SLEEPSTAGE_SOURCE
    from src.utilities.myconfig import HARVEST_DIRECTORY
    from src.utilities.myconfig import REPROCESS_ALL

    argparser = argparse.ArgumentParser(description='Harvest data from the source directories', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    argparser.add_argument('--psg', '-p', type=str, default=LDS_SOURCE, help='Source directory for PSG recordings (EDF and TXT)')
    argparser.add_argument('--somnofy', '-s', type=str, default=SOMNOFY_SOURCE, help='Source directory for Somnofy files (CSV)')
    argparser.add_argument('--sleepstage', '-t', type=str, default=SLEEPSTAGE_SOURCE, help='Source directory for SleepStage files (CSV)')
    argparser.add_argument('--destination', '-d', type=str, default=HARVEST_DIRECTORY, help='Destination directory for harvested files')
    argparser.add_argument('--random', '-r', action='store_true', help='Shuffle the recordings before harvesting')
    argparser.add_argument('--max-count', '-m', type=int, default=0, help='Maximum number of recordings to harvest')
    argparser.add_argument('--force', '-f', action='store_true', default=REPROCESS_ALL, help='Force reprocessing of all recordings')
    argparser.add_argument('--ignore', '-i', action='store_true', help='Ignore errors and skip to next recording')
    argparser.add_argument('--dryrun', '-n', action='store_true', help='Dry run, don\'t actually copy files')
    argparser.add_argument('--workers', '-w', type=int, default=8, help='Number of workers to use for harvesting')
    argparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (debug level)')
    argparser.add_argument('--very-verbose', '-vv', action='store_true', help='Very verbose output (debug level)')
    args = argparser.parse_args()

    if not args.psg or not args.somnofy or not args.sleepstage or not args.destination:
        argparser.print_help()
        sys.exit(1)

    #
    # Set up logging levels
    #

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    if args.very_verbose:
        vv_logger.setLevel(logging.DEBUG)
    else:
        vv_logger.setLevel(logging.CRITICAL)

    #
    # Run the main thing
    #

    track_latest("harvest_start_time", datetime.now().isoformat())
    track_latest("max_count", args.max_count)
    track_latest("workers", args.workers)
    track_latest("ignore_errors", args.ignore)

    # Print some info
    print(f"* Sleep Study Pipeline - Harvester (chrifren@ifi.uio.no)")
    print(f"Sorting, tinkering, and gathering all data from the sources into {args.destination}...")
    print(f"This will take a while... {get_life_tip()}\n")

    with Timer(operation_name="Harvest", writelog=False) as timer: 
        harvest_all_recordings(args)
        print(f"Harvesting completed in {timer.get_duration():.2f}s")

    track_latest("harvest_end_time", datetime.now().isoformat())

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)