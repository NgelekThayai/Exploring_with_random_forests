# RandomForestFunctions

RandomForestFunctions is a way to functionalize the process of the random forest classifier and explore datasets quickly and effectively. 

## Installation:
------------

First, clone this repository 

```bash
git clone https://codeberg.com/ggelek/ProjectName.git
```
Then make sure to download miniconda. Conda environments help organize a computer's system directories and allow one to run experiments in a self-contained, sandbox environment with packages that may be unstable. Download Miniconda for your operating system  with this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

When in your terminal, you will know Conda is running when the first word in your terminal is (base).

to create a conda environment execute the following commands:

```bash
$ conda create --name myenv 
```
replace "myenv" with what you want your environment name to be.

Conda will then ask if you want to proceed, type y.

Once in your environment install the program dependencies:
```bash 
$ conda install pandas 
$ conda install numpy
$ conda install scikit-learn
```

if you want too see all the packages installed in the active conda environment type:
```bash
$ conda list
```

I chose to run these programs in Jupyter Notebook which is a ["web-based interactive computing platform"](https://jupyter.org/)
It allows me to run code line by line so I can fully comprehend the work that each line does. To install Jupyter Notebook execute the following commands in your conda environment:

```bash
$ conda activate myenv 
$ conda install jupyter
```
and follow the prompts to download Jupyter Notebook. 

once Jupyter Notebook is installed, make sure you are in your conda environment, and then execute the following command:
``` bash
$ jupyter notebook
```
and your Jupyter Notebook interface will appear in your default browser. 

Now in your Jupyter Notebook you can create a new file with the "New" button in the righthand corner and choose to open your notebook with Python 3. 

Now in your Jupyter Notebook file import mltool.py and etl.py and the program dependencies with the following:
```bash
#make sure you are in the directory you cloned earlier using 
cd ProjectName
import mltool #make sure to shift+enter after typing to execute each line
import etl.py #make sure to shift+enter after typing to execute each line
import numpy as numpy #make sure to shift+enter after typing to execute each line
import pandas as pandas #make sure to shift+enter after typing to execute each line
import sklearn  #make sure to shift+enter after typing to execute each line
```

## Usage
------------
Import your dataset and read it.
Then create an object that contains the functions ie:
```bash
transformer=etl.PandasTransformer()
model = mltool.Model()
```
and then the rest is up to you and what functions you want to execute.

## Thank You

I would like to thank my mentor Mr. Damian Eads for all of his help while I explored machine learning. He taught me so much about machine learning, computing, and life. I am truly indebted to Mr.Eads and am forever greatful for his guidence. I would also like to thank the Institue for Computing in Research for this oppurtunity. 
