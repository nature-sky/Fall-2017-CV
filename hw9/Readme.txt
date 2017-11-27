OS: ubuntu 16.04 64bits
Language: python

How to use: 
  * usage: $ python ./hw9.py [-h] [operator] [threshold]
  * -h: print this help message and exit
  * operator: robert, prewitt, sobel, frei, kirsch, robinson, nevatia
  * threshold: an integer for the operator
  
or execute 'task.sh' to get all results
  * usage: $ ./task.sh  

Please install opencv library first before running the program. 
> Step1: install pip
> $ sudo apt-get install python-pip python-dev python-tk
> $ sudo pip install --upgrade pip

> Step2: install opencv2
> $ pip install opencv-python
