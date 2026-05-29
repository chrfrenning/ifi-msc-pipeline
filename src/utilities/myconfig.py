# #####################################################################
#
# Configuration
#
# This sets some global variables that are needed throughout the
# project. Some of these will be specific to your machine or
# environment. 
# 
# Call this script with --create to create a template .env file.
#
# On IFI machines, the paths will be /projects/respire/...
#
# #####################################################################

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()



#
# Data source and destination directories
#

LDS_SOURCE = os.getenv('LDS_SOURCE')
SOMNOFY_SOURCE = os.getenv('SOMNOFY_SOURCE')
SLEEPSTAGE_SOURCE = os.getenv('SLEEPSTAGE_SOURCE')

HARVEST_DIRECTORY = os.getenv('HARVEST_DIRECTORY', './1_harvest')
PROCESSED_DIRECTORY = os.getenv('PROCESSED_DIRECTORY', './2_processed')
NORMALIZED_DIRECTORY = os.getenv('NORMALIZED_DIRECTORY', './3_normalized')
EPOCHS_DIRECTORY = os.getenv('EPOCHS_DIRECTORY', './4_epochs')


if not __name__ == '__main__':
    if not LDS_SOURCE or not SOMNOFY_SOURCE or not SLEEPSTAGE_SOURCE:
        raise ValueError("LDS_SOURCE, SOMNOFY_SOURCE, and SLEEPSTAGE_SOURCE must be set in the .env file")



#
# Global configuration
#

# The system outputs CSV by default, but you can also save to HDF5
SAVE_HDF5 = os.getenv('SAVE_HDF5', '0') == '1'

# Write high resolution data, warning big files!
# Especially the CSV is slow to write and significantly increases processing time
SAVE_RAW_CSV = os.getenv('SAVE_RAW_CSV', '0') == '1'
SAVE_RAW_HDF5 = os.getenv('SAVE_RAW_HDF5', '0') == '1'

# Intermediate data formats, useful for inspection but not directly used
SAVE_DERIVED_CSV = os.getenv('SAVE_DERIVED_CSV', '0') == '1'
SAVE_DERIVED_HDF5 = os.getenv('SAVE_DERIVED_HDF5', '0') == '1'

# Formats for transformed and output data
SAVE_CSV = os.getenv('SAVE_CSV', '0') == '1'
SAVE_HDF5 = os.getenv('SAVE_HDF5', '0') == '1'

# Reprocess all data, useful for debugging
# Default is that process is incremental, only processing new or unseen data
REPROCESS_ALL = os.getenv('REPROCESS_ALL', '0') == '1'



# 
# Signal processing config
#

PSG_SIGNALS = os.getenv('PSG_SIGNALS', 'Abdomen,Chest,SpO2,Pulse,Flow_DR').split(',')
PSG_ALIGNMENT_SIGNAL = os.getenv('PSG_ALIGNMENT_SIGNAL', 'Abdomen')

SOMNOFY_COLUMNS = os.getenv('SOMNOFY_COLUMNS', 'timestamp,distance,relative_distance,setting_distance,movement_instant').split(',')
SOMNOFY_SIGNALS = os.getenv('SOMNOFY_SIGNALS', 'distance,relative_distance,setting_distance,movement_instant').split(',')

SAMPLING_FREQUENCY = os.getenv('SAMPLING_FREQUENCY', '1s')

SAVGOL_WINDOW_SIZE = int(os.getenv('SAVGOL_WINDOW_SIZE', '101'))
SAVGOL_POLYORDER = int(os.getenv('SAVGOL_POLYORDER', '3'))

ALIGNMENT_SEARCH_WINDOW = int(os.getenv('ALIGNMENT_SEARCH_WINDOW', '60'))
ALIGNMENT_MAX_LAG = int(os.getenv('ALIGNMENT_MAX_LAG', '30'))
SOMNOFY_LENGTH_FOR_ALIGNMENT = int(os.getenv('SOMNOFY_LENGTH_FOR_ALIGNMENT', '1000'))



#
# Normalization config
#

SIGNALS_TO_NORMALIZE = ['psg_', 'somnofy_', 'aligned_']
SLEEP_SCORE_COLUMNS = ['scoring_sleep-rem', 'scoring_sleep-s1', 'scoring_sleep-s2', 'scoring_sleep-s3']

Z_SCORE_THRESHOLD = int(os.getenv('Z_SCORE_THRESHOLD', '3'))


#
# Splitter config
#

SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', '30'))
Y_THETA = int(os.getenv('Y_THETA', '1'))
SCORING_EVENTS = ['scoring_apnea-obstructive', 'scoring_apnea-mixed', 'scoring_apnea-central']
EPOCH_SIGNALS = os.getenv('EPOCH_SIGNALS', '*')



#
# Create the .env file if it doesn't exist
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the .env file if it doesn\'t exist', epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)')
    parser.add_argument('--create', action='store_true', help='Create the .env file if it doesn\'t exist')
    parser.add_argument('--ignore', action='store_true', help='Silently ignore errors')
    args = parser.parse_args()

    if not args.create:
        parser.print_help()
        sys.exit(1)
    else:
        # abort if .env file already exists
        if os.path.exists('.env'):
            print(f".env file already exists, aborting")
            if args.ignore: sys.exit(0)
            if not args.ignore: sys.exit(1)

        # iterate over all globals in this file and write them to the .env file
        with open('.env', 'w') as f:
            for var in list(globals()):
                if var.isupper():
                    f.write(f"{var}={globals()[var] or ''}\n")
        print(f"Created .env file with default values")
        sys.exit(0)