# Preprocessing toolkit for PSG and Somnofy data for the Lovisenberg Pediatric Sleep Disorder Dataset

This is the first of two code repositories supporting my MSc thesis
"Small Patients, Sparse Events: An End-to-End Framework for Machine Learning
for Pediatric Sleep Apnea Detection".

## What it is
This project preprocesses sleep apnea datasets, specifically the LDS, into 
a format that facilitates analysis, review, and machine learning.

The second repository supporting my thesis is available at 
https://github.com/chrfrenning/ifi-msc-classifier and trains and evaluates
ML classifiers based on the preprocessing in this project.

## How to use
Use `make install` to set up this project with necessary dependencies. Then
edit the `.env` file with paths to source data. There is a bunch of other configuration
options that can be set in the environment file, see `myconfig.py` for the overview
and descriptions - or as a place to start searching the code to see the minute details.

To see how this works without spending too much time, run `make demo` which will
randomly select a few overnight recordings and spin up a webserver to help you
visually inspect and enjoy the results.

To run the entire pipeline with the configuration options used at the time of delivery
of my thesis, run `make all` and go grab a nice, big cup of coffee.

After processing, you will have the whole Lovisenberg Sleep Study Dataset
with signals from PSG, Somnofy, and VitalThings Sleep Staging cleaned, aligned,
normalized, and ready for experiments.

Use `make clean` to reset your repository - but `.gitignore` should already keep
you sane if you check in modified code.

## About the results
Each overnight recording will be placed in its own directory. Inspect the
`metadata.json` files for information about each recording and the data
behind it.

The project saves all processing steps (with the high frequency step optional 
and disabled by default). This means you can inspect the results of all my code,
our use it for your own research - either inspecting this data as-is or adding
own experiments to the pipeline.

## About errors in the source data
There exists current errors - and probably will arise new ones in the future -
in the source data. By default the steps run with the --ignore command line
parameter which will log but ignore any errors. This ensures complete datasets
make it to the end, ignoring others. If you intend to fix the errors in the source,
iteratively run without -i and fix the errors one by one.

## Runtime metrics
Each tool will produce a `<toolname>.metrics.json` file with observability metrics
in the working directory. This contains vital parameters when run to document
choices that affect the processing results, as well as timing of operations and
captures of some stats and data that are essential to my work. If you base work
on these tools, I suggest you capture the metrics file as part of your report.

The tools also create a `<toolname>.log` file with very verbose logging. You may
want to run the program with `-v` or `-vv` parameters to see output on stdout 
while running the tools for the first time to understand their inner workings.

## Data inspection
Run `make serve` to spin up a webserver on `http://localhost:27182`. It will show
all of the processed recordings, and provide a visualization of all the raw signals
and derivations of them.

## Do I need CSV files?
Writing CSV files, especially at high resolution, is consuming both in time and space.
I recommend using HDF5, and by default CSV is turned off. Use `./hfls` and `./hfcat`
to see a HDF5 file in CSV format. You will save an order of magnitude of time and cut
disk space in half with HDF5. If you use `pandas`, it reads HDF5 as easy as CSV.
You may also like `https://myhdf5.hdfgroup.org/`

## Understanding this project
To understand what is happening under the hood it may be especially useful to
run `./harvester` and `./correlator` with `-v` and `-vv` command line options for verbose
and very verbose logging output. Combine this with `-m 1` to process only one
recording, then it will be possible to read through and see each processing step
and what is produced in each step.

# Reading direction
Read my code from bottom up - it starts with main() in the end of the file... I have
written much C code and don't like to maintain header files so my Python code has ended 
up with pretty much the same structure. Also I assume you have a large monitor, as
I don't follow the 72 character convention unless it really makes the code more readable
- that way more code fits on the screen and gives me much faster overview. Not everyone 
likes it; there's automatic code formatters that will help you if you don't agree with me.

## Source
If you received this code somehow other than through GitHub, you can always
refer to https://github.com/chrfrenning/ifi-msc-pipeline which is the original
source repository.

## Citation
If you use this repository, please cite the associated thesis:
Frenning, Christopher. *Small Patients, Sparse Events: An End-to-End Framework
for Machine Learning for Pediatric Sleep Apnea Detection*. 
University of Oslo, 2026. https://doi.org/10.5281/zenodo.11098337

## Author
Christopher Frenning, 2025-2026
chrifren@ifi.uio.no, christopher@frenning.com
