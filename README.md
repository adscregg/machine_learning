# machine_learning
This repo contains simple analysis and visualisations of open source datasets as well as common machine learning algorithms that I have written from scatch in Python.

## Purpose of this repository

This repository serves as a portfolio of code relating to Data Science to demonstrate my interest in the field. It also serves as a practice ground for me to develop my coding knowledge and tools that are commonly used in the Data Science industry.

## Prerequisites

##### Python version

All of the files in this repository are written in Python and were initially built using version 3.6.8 x64, it is recommended that this version of Python is used as some modules used did not support Python 3.7.x or above at the time of creation.
To install this version of Python follow this [link](https://www.python.org/downloads/) and find the correct version.
To ensure that you have the correct version of Python running, start a Python shell by typing `python` into the terminal then running the following:

```python
import sys
print(sys.version)
exit()
```

You should see the output as being something like:

`'3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]'`

noting that this is the 64 bit version.

##### Python packages

The following is a list of required packages to be install before full use of all the code can be achieved on a local system.

1. numpy
2. pandas
3. scikit-learn
4. matplotlib
5. seaborn
6. keras
7. tensorflow (gpu version recommended)
8. cv2
9. nltk

All but tensorflow can be installed straight away using Python's package manager, pip. Run the following at the terminal:

```
pip install numpy pandas scikit-learn matplotlib seaborn keras cv2 nltk
```
The requirements for tensorflow's gpu version is more complicated. All of the below are required, check the [tensorflow website](https://www.tensorflow.org/install/gpu) to see the exact versions you should download of each piece of software.

1. An Nvidia graphics card
2. This graphics card has to be CUDA enabled (see this [link](https://developer.nvidia.com/cuda-gpus) to check if yours is CUDA enabled)
3. The lib, bin and include folder must be added to your path
4. You must have CuDNN installed on your system (this can be downloaded from [here](https://developer.nvidia.com/cudnn), you have to create an account)
5. A version of Microsoft's VisualStudio IDE which can be downloaded from [here](https://visualstudio.microsoft.com/) and the C/C++ tools box must be checked upon first installation

I recommend following a video tutorial on how to get tensorflow installed but the above is an outline of what will be required.

If you have satisfied all the conditions to get tensorflow gpu installed then you can now run `pip install tensorflow-gpu` to install the gpu version of tensorflow. The version used to build the models in these files is 1.14.0.

To test the installation open a python shell in the terminal and run the code below.

```Python
import tensorflow as tf
c = tf.constant('Hello World')
with tf.Session() as sess:
  print(sess.run(c))
exit()
```

The output should be `b'Hello World'` in addition to many lines of information about your GPU.

If this is not successful or you do not wish to install the GPU version, simply run `pip install tensorflow` in the terminal and this will install the CPU version of tensorflow.

## Setup

To setup the project run `git clone https://github.com/adscregg/machine_learning.git .`.

You will need to install nltk packages for the code to run in its current form without adding in a few extra lines. There are several ways to do this but I recommend the following:

Open a python shell and run the code below.

```Python
import nltk
nltk.download('all')
exit()
```
This will install all nltk modules from the package and is not completely necessary to install them all but still recommended.

## Runtime info
##### System details
- TensorFlow-GPU version 1.14.0
- NVidia GeForce GTX 1050 Ti with Max-Q Design
##### Approx runtimes
- reddit_NLP_analysis.py: 12 minutes
- wine_classification.py: 30 - 40 seconds
- cats_and_dogs.py: 2 minutes
- mnist.py: 30 - 40 seconds


## About the Folders

### analysis

##### scikit-learn analysis
This folder contains files where a variety of methods will be used to analysis data with the vast majority of methods being built-in scikit-learn functions. The outline of the structure that each file will have is:
1. read in dataset
2. preprocess data
3. fit and tune models
4. test models

I hope to have a variety of types of datasets from clean, easy to use datasets, to ones that need significant preprocessing in order to be useful. I also hope to be able to analyse different types of data such as numerical, categorical and NL and get meaningful results.

Classification will likely be the predominant type of analysis done in these files

##### self-written_algorithms

The files ending in \_ADS are machine learning algorithms that I have written from scratch in an attempt to better understand the mathematics behind the algorithms.

The random_data.py file contains convenience methods for creating mock datasets from scikit-learns make datasets methods which are then imported into the other files within this folder

##### tf-keras analysis

This folder contains models that use Neural Networks to do classification tasks on a variety of datasets. The files will vary as to which module they use (tensorflow or keras) depending on preference at the time of writing the models. It is possible to run (most of) the models with the CPU version of tensorflow on a local system, but this will be very slow and is not recommended. I will indicate which files are suitable to be run on the CPU version with comments at the top of each file.

### datasets

Contains all of the datasets that I will be working on in this repository. Most of the files will be .csv files but this may vary.

## Notes for running files on local machine

Always run the files from the terminal as there have been issues with file paths not being found when running scripts within an IDE using an add-in package.

## Useful links

- CNN: https://github.com/nehal96/Deep-Learning-ND-Exercises/blob/master/Convolutional%20Neural%20Networks/convolutional-neural-networks-notes.md
- Logistic Regression: https://medium.com/deep-math-machine-learning-ai/chapter-2-0-logistic-regression-with-math-e9cbb3ec6077
- Linear Regression: https://brilliant.org/wiki/linear-regression/



## Acknowledgements

- Sentdex: https://www.youtube.com/user/sentdex
- 3Blue1Brown: https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw
