import os
import json
from pathlib import Path

from .mytimer import Timer


#
# Path display
# Keeping these short as they are used frequenly in logging statements
#

def dp(path, relative_to=None):
    try:
        return Path(path).relative_to(relative_to).as_posix()
    except:
        return str(path)

def bn(filename):
    return os.path.basename(filename)



#
# Path manipulation
#

def postfix_filename(filename, postfix):
    filename, extension = os.path.splitext(filename)
    return filename + '_' + postfix + extension



#
# Some utilities for large text files
#

def guard_dos_newlines(content):
    return content.replace('\r\n', '\n')

def convert_to_strings(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_strings(item) for item in obj]
    else:
        return str(obj)

def read_start_of_file(filename):
    BYTES_TO_READ = 4096
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(BYTES_TO_READ)
    return guard_dos_newlines(content)

def read_end_of_file(filename):
    BYTES_TO_READ = 4096

    file_size = os.path.getsize(filename)
    if file_size <= BYTES_TO_READ:
        # If file is smaller than BYTES_TO_READ, read the entire file
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        # Calculate the position to start reading from
        start_pos = file_size - BYTES_TO_READ
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_pos)
            content = f.read(BYTES_TO_READ)
    
    return guard_dos_newlines(content)



#
# Convenience on time ranges
#

def is_between(ts, start_ts, end_ts):
    return start_ts <= ts <= end_ts

def is_overlapping_time_ranges(start, end, range_start, range_end):
    if start < range_start and end > range_end:
        return True
    return is_between(start, range_start, range_end) or is_between(end, range_start, range_end)



#
# Files with metadata about an overnight recording
#

def load_metadata(recording_dir):
    with Timer(operation_name="LoadMetadata", writelog=False):
        metadata_file = os.path.join(recording_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata

def save_metadata(recording_dir, metadata):
    with Timer(operation_name="SaveMetadata", writelog=False):
        metadata_file = os.path.join(recording_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)