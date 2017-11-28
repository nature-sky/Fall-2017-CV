#! /bin/bash

# TODO: Change threshold here!
thr_laplace1="15"
thr_laplace2="15"
thr_mvl="20"
thr_log="3000"
thr_dog="1"

python ./hw10.py laplace1 ${thr_laplace1}
python ./hw10.py laplace2 ${thr_laplace2}
python ./hw10.py MVL ${thr_mvl}
python ./hw10.py LOG ${thr_log}
python ./hw10.py DOG ${thr_dog}
