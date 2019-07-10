# machine_learning
This repo contains simple analysis and visualisations of open source datasets as well as common machine learning algorithms that I have written from scatch in Python.

## Prerequisites

##### Python version

All of the files in this repository are written in Python and were initially built using version 3.6.8 x64, it is recommended that this version of Python is used as some modules used did not support Python 3.7.x or above at the time of creation.
To install this version of Python follow this [link](https://www.python.org/downloads/) and find the correct version.
To ensure that you have the correct version of Python running, start a Python shell by typing `python` into the terminal then running the following:

```python
import sys
print(sys.version)
```

You should see the output as being something like:

`'3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)]'`

noting that this is the 64 bit version.

##### Python packages

The following is a list of required packages to be install before full use of all the code can be achieved.

1. numpy
2. pandas
3. scikit-learn
4. matplotlib
5. seaborn
6. keras
7. tensorflow (gpu version recommended)

All but tensorflow can be installed straight away using Python's package manager, pip. Run the following at the terminal:

```
pip install numpy pandas scikit-learn matplotlib seaborn keras
```
The requirements for tensorflow's gpu version is more complicated. All of the below are required, check the [tensorflow website]() to see the exact versions you should download of each piece of software.

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
```

The output should be `b'Hello World'` in addition to many lines of information about your GPU.


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

### tf-keras analysis

This folder contains models that use Neural Networks to do classification tasks on a variety of datasets. The files will vary as to which module they use (tensorflow or keras) depending on preference at the time of writing the models.

### visualisations

No model analysis will be done within the files in this folder, it is purely to visualise datsets and experiment with different tools that visualisation libraries offer.

### datasets

Contains all of the datasets that I will be working on in this repository. Most of the files will be .csv files but this may vary.
