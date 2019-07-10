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
4. You must have CuDNN installed on your system (this can be downloaded from [here](), you have to create an account)
5. A version of Microsoft's VisualStudio which can be downloaded from [here]() and the C/C++ tools box must be checked upon first installation

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
