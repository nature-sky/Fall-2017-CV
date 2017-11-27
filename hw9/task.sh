#! /bin/bash

# TODO: Change threshold here!
thr_robert="12"
thr_prewitt="24"
thr_sobel="38"
thr_frei="30"
thr_kirsch="135"
thr_robinson="43"
thr_nevatia="12500"

python ./hw9.py robert ${thr_robert}
python ./hw9.py prewitt ${thr_prewitt}
python ./hw9.py sobel ${thr_sobel}
python ./hw9.py frei ${thr_frei}
python ./hw9.py kirsch ${thr_kirsch}
python ./hw9.py robinson ${thr_robinson}
python ./hw9.py nevatia ${thr_nevatia}
