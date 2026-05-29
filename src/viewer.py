# #####################################################################
#
# Viewer
#
# Very simple web server using the html files produced by the
# plotter utility to let users navigate and inspect overnight
# recordings from a sleep study dataset.
#
# Copyright 2026 Christopher Frenning
# chrifren@ifi.uio.no, christopher@frenning.com
#
# Licensed under the MIT License - see LICENSE.md for details
#
# Please cite my thesis if you use this code in your own work.
#
# #####################################################################


import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_file, render_template

app = Flask(__name__, template_folder='templates')

DATA_DIR = None

def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

def calculate_timespan(first_ts, last_ts):
    first = parse_timestamp(first_ts)
    last = parse_timestamp(last_ts)
    if first and last:
        delta = last - first
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f}h"
    return "N/A"

def get_recordings():
    recordings = []
    
    if not DATA_DIR.exists():
        return recordings
    
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            metadata_file = item / 'metadata.json'
            html_plot_file = item / 'normalized_1s.h5'
            if metadata_file.exists() and html_plot_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    recording_id = metadata.get('recording_id', item.name)
                    patient_id = metadata.get('patient_id', 'Unknown')
                    first_ts = metadata.get('first_ts', '')
                    last_ts = metadata.get('last_ts', '')
                    
                    date = first_ts.split()[0] if first_ts else 'N/A'
                    timespan = calculate_timespan(first_ts, last_ts)

                    ahi_block = metadata.get('ahi')
                    ahi_value = None
                    if isinstance(ahi_block, dict):
                        raw = ahi_block.get('ahi')
                        if raw is not None:
                            try:
                                ahi_value = float(raw)
                            except (TypeError, ValueError):
                                pass
                    
                    recordings.append({
                        'recording_id': recording_id,
                        'patient_id': patient_id,
                        'first_ts': first_ts,
                        'last_ts': last_ts,
                        'date': date,
                        'timespan': timespan,
                        'ahi': ahi_value
                    })
                except Exception as e:
                    print(f"Error reading metadata for {item.name}: {e}")
    
    recordings.sort(key=lambda x: int(x['recording_id']))
    return recordings

@app.route('/plot/<recording_id>')
def plot(recording_id):
    html_file = DATA_DIR / str(int(recording_id)) / 'plot_1s.html'
    if html_file.exists():
        return send_file(html_file)
    return "Recording not found", 404

@app.route('/api/metadata/<recording_id>')
def metadata(recording_id):
    metadata_file = DATA_DIR / str(int(recording_id)) / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    return "Metadata not found", 404

@app.route('/api/recordings')
def api_recordings():
    recordings = get_recordings()
    return jsonify(recordings)

@app.route('/')
def index():
    return render_template('index.html')

#
# Entry point
#

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Sleep Study Recording Viewer (HTTP Server)', 
        epilog='(C) 2026 Christopher Frenning (chrifren@ifi.uio.no)'
    )
    
    argparser.add_argument('--source', '-s', type=str, default=None, help='Source directory to serve')
    argparser.add_argument('--port', '-p', type=int, default=27182, help='Port to serve on')
    argparser.add_argument('--bind-address', '-b', type=str, default='127.0.0.1', help='Host to serve on')

    args = argparser.parse_args()

    if not args.source:
        print("Error: --source is required")
        exit(1)

    DATA_DIR = Path(os.path.abspath(args.source))

    print(f"Serving recordings from {DATA_DIR} on http://{args.bind_address}:{args.port}")
    app.run(debug=True, host=args.bind_address, port=args.port)
