# Sleep Study Pipeline (c) chrifren@ifi.uio.no 2026
#
# This is not a "build-perfect" pipeline, some shortcuts are taken
# to make this readable and maintainable. The core idea is that
# all dataset and ML decisions are documented in code and fully
# reproducible (but the imperfection means if something goes wrong
# it will not perfectly recover without clean restart)
#

# Get up and running with needed dependencies
install:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	. .venv/bin/activate && python myconfig.py --create --ignore
	
clean:
	rm -rf dataset/ 1_harvest/ 2_processed/ 3_normalized/ 4_plots/ 5_epochs/ *.metrics.json *.log

# Best way to explore is to run demo and serve (quick, not complete dataset!)
demo:
	rm -rf 1_harvest/ 2_processed/ 3_normalized/ 4_plots/ 5_epochs/
	.venv/bin/python -m src.harvester -i -w 3 -m 3 -r
	.venv/bin/python -m src.correlator --require-somnofy --require-sleepstage -i -w 8
	.venv/bin/python -m src.normalizer --trim-sleep-stages --z-score-threshold 3 -i
	.venv/bin/python -m src.plotter --destination 3_normalized/
	.venv/bin/python -m src.plotter --destination 4_plots/ --flatten
	.venv/bin/python -m src.splitter --sequence-length 30 --theta 1 --destination 5_epochs/demo_epochs.h5
	.venv/bin/python -m src.viewer --source 3_normalized/

serve:
	.venv/bin/python -m src.viewer --source ./dataset/ss_som_trim/


# #####################################################################
#
# Datasets for my thesis and useful to others
#
# #####################################################################

./dataset/original:
# Harvest all the data and limit to recordings where we have sleep staging data (we ignore rest of set)
	.venv/bin/python -m src.harvester -i -w 8 -d ./dataset/original

# Full dataset for analysis
./dataset/full: ./dataset/original
	.venv/bin/python -m src.correlator -s ./dataset/original -d ./dataset/full -i -w 8
	.venv/bin/python -m src.normalizer -s ./dataset/full -d ./dataset/full --z-score-threshold 3 -i
	.venv/bin/python -m src.plotter -s ./dataset/full -d ./dataset/full

# Sleep staged dataset for analysis
./dataset/ss: ./dataset/original
	.venv/bin/python -m src.correlator -s ./dataset/original -d ./dataset/ss --require-sleepstage -i -w 8
	.venv/bin/python -m src.normalizer -s ./dataset/ss -d ./dataset/ss --trim-sleep-stages --z-score-threshold 3 -i
	.venv/bin/python -m src.plotter -s ./dataset/ss -d ./dataset/ss

# Intersection of sleep-staging and somnofy data
./dataset/ss_som: ./dataset/original
	.venv/bin/python -m src.correlator -s ./dataset/original -d ./dataset/ss_som --require-somnofy --require-sleepstage -i -w 8
	.venv/bin/python -m src.normalizer -s ./dataset/ss_som -d ./dataset/ss_som --trim-sleep-stages --z-score-threshold 3 -i 
	.venv/bin/python -m src.plotter -s ./dataset/ss_som -d ./dataset/ss_som



# #####################################################################
#
# Splits for training and analysis, specifically for my thesis or repro
#
# #####################################################################

#
# Experiment 1: Effect of trimming to sleep periods only
#

./dataset/experiment1a.h5: ./dataset/ss_som 
# Normalize the data and trim to sleep periods only, standard psg signals
	echo "---------------------------------------"
	echo "Preparing datasets for Experiment 1a..."
	.venv/bin/python -m src.normalizer -s ./dataset/ss_som -d ./dataset/experiment1a --trim-sleep-stages --z-score-threshold 3 -i 
	.venv/bin/python -m src.splitter -s ./dataset/experiment1a -d ./dataset/experiment1a.h5 --signals "psg_chest_rn_mean,psg_flow_dr_rn_mean,psg_pulse_rn_mean,psg_spo2_rn_mean" --sequence-length 30 --theta 1 --scoring-events "scoring_apnea-central,scoring_apnea-mixed,scoring_apnea-obstructive" --ignore

# Normalize data without trimming, standard psg signals
./dataset/experiment1b.h5: ./dataset/full
	echo "---------------------------------------"
	echo "Preparing datasets for Experiment 1b..."
	.venv/bin/python -m src.normalizer -s ./dataset/full -d ./dataset/experiment1b --z-score-threshold 3 -i 
	.venv/bin/python -m src.splitter -s ./dataset/experiment1b -d ./dataset/experiment1b.h5 --signals "psg_chest_rn_mean,psg_flow_dr_rn_mean,psg_pulse_rn_mean,psg_spo2_rn_mean" --sequence-length 30 --theta 5 --scoring-events "scoring_apnea-central,scoring_apnea-mixed,scoring_apnea-obstructive" --ignore

experiment1: ./dataset/experiment1a.h5 ./dataset/experiment1b.h5


#
# Experiment 2: Normalization strategies, based on trimmed data
# Taking a shortcut here with dependencies otherwise way too verbose
# ie just using last file output
#

./dataset/experiment2.h5: ./dataset/ss
	echo "--------------------------------------" > nul
	echo "Preparing datasets for Experiment 2..." > nul
	.venv/bin/python -m src.splitter -s ./dataset/ss -d ./dataset/experiment2.h5 --sequence-length 30 --theta 1 --ignore

experiment2: ./dataset/experiment2.h5


#
# Experiment 3: Labelling thresholds on model performance
# Again shortcut see above
#
# TODO: Need best result of normalization strategies to setup this, assuming recnorm for now

./dataset/experiment3: ./dataset/ss
	.venv/bin/python -m src.normalizer -s ./dataset/ss -d ./dataset/experiment3 --trim-sleep-stages --z-score-threshold 3 -i 

./dataset/e3_theta10.h5: ./dataset/experiment3
	echo "--------------------------------------" > nul
	echo "Preparing datasets for Experiment 3..." > nul
	
	.venv/bin/python -m src.splitter -s ./dataset/experiment3 -d ./dataset/e3_theta1.h5 --signals "*" --sequence-length 30 --theta 1
	.venv/bin/python -m src.splitter -s ./dataset/experiment3 -d ./dataset/e3_theta5.h5 --signals "*" --sequence-length 30 --theta 5
	.venv/bin/python -m src.splitter -s ./dataset/experiment3 -d ./dataset/e3_theta10.h5 --signals "*" --sequence-length 30 --theta 10

experiment3: ./dataset/e3_theta10.h5


# Experiment 4: KFold vs Classic
# This is a classifier experiment only, using best of normalization strategies with k-fold vs classic split


# Experiment 5: PSGCNN vs PSGMultiScaleCNN
# This is a classifier experiment only, using best of normalization strategies with best of train strategy with PSGCNN vs PSGMultiScaleCNN


#
# Experiment 6: Signal selection
#
./dataset/experiment6: ./dataset/ss_som
	.venv/bin/python -m src.normalizer -s ./dataset/ss_som -d ./dataset/experiment6 --trim-sleep-stages --z-score-threshold 3 -i 
	.venv/bin/python -m src.splitter -s ./dataset/experiment6 -d ./dataset/experiment6.h5 --signals "*" --sequence-length 30 --theta 1 --scoring-events "scoring_apnea-central,scoring_apnea-mixed,scoring_apnea-obstructive,scoring_desat,scoring_hypopnea" --ignore

experiment6: ./dataset/experiment6


# Experiment 7:
# AHI level, this can be done in the classifier by filtering on AHI metadata when building the dataset


#
# Experiment 8: Sleep stage with Somnofy
#
experiment8:
	.venv/bin/python -m src.normalizer -s ./dataset/ss_som -d ./dataset/experiment8 --z-score-threshold 3 -i 
	.venv/bin/python -m src.splitter -s ./dataset/experiment8 -d ./dataset/experiment8.h5 --signals "scoring_sleep-rem,scoring_sleep-s0,scoring_sleep-s1,scoring_sleep-s2,scoring_sleep-s3,vtss_sleep_stage" --ignore


#
# Experiment 9: Classify with Somnofy Relative Distance
# We can use the total dataset as for experiment 2 and create specialized training sets
#
experiment9: experiment2


#
# Run all experiments (big batch job!)
#
datasets: ./dataset/full ./dataset/ss ./dataset/ss_som
experiments: experiment1 experiment2 experiment3 experiment6 experiment9
all: experiments



#
# Push from GPU workstation to IFI server
#
push:
	rsync -avh --delete ./dataset/ IFI:/projects/respire/chrifren/dataset/
	ssh IFI 'chmod -R o+r /projects/respire/chrifren/dataset/'
	ssh IFI 'chmod -R o+X /projects/respire/chrifren/dataset/'
