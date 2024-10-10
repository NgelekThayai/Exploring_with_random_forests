# Exploring_With_Random_Forests

Exploring_With_Random_Forests is a way to functionalize the process of the random forest classifier and explore datasets quickly and effectively. 

## Installation:
------------

First, clone this repository 

```bash
git clone https://github.com/ngelekthayai/Exploring_with_random_forests.git
```
Then make sure to download miniconda. Conda environments help organize a computer's system directories and allow one to run experiments in a self-contained, sandbox environment with packages that may be unstable. Download Miniconda for your operating system  with this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

When in your terminal, you will know Conda is running when the first word in your terminal is (base).

to create a conda environment execute the following commands:

```bash
conda create --name random_forest
```
Conda will then ask if you want to proceed, type y.
Once in your environment install the program dependencies using the bash script provided in the repo called "install.sh"
I chose to run these programs in Jupyter Notebook which is a ["web-based interactive computing platform"](https://jupyter.org/)
It allows me to run code line by line so I can fully comprehend the work that each line does. 
This bash script also downloads [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), and [scikit-learn](https://scikit-learn.org/stable/).

```bash 
conda activate random_forest
chmod +x install.sh
./install.sh
```

if you want too see all the packages installed in the active conda environment type:
```bash
conda list
```

## Usage
------------
In your Conda environment called "random_forest" run the "run.sh" bash script to run the Jupyter Notebook included in the repository.
```bash
conda activate random_forest
chmod +x run.sh
./run.sh
```
Once this is done you will be in the Jupyter Notebook home directory.
Navigate to the Breast_Cancer.ipynb file and click on that file. 
Once there you should see a filled up Jupyter Notebook. 
To run the whole Jupyter Notebook at once first click on "cells" and then click on "run all".
To execute each cell one at a time shift + enter on the selected cell.






