﻿# Banknote authentication with Machine Learning

The goal was to detect real versus fake banknotes. The data used in this problem was collected from images of real banknotes. By building the Machine learning classifier at the end of the pipeline, we are able to detect real vs fake banknotes based on the information parsed from it.

## Dataset

Dataset used in this project can be found [here]( https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

## Install

### &nbsp;&nbsp;&nbsp; Supported Python version
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Python version used in this project: 3.5+

### &nbsp;&nbsp;&nbsp; Libraries used

> *  [Pandas](http://pandas.pydata.org) 0.18.0
> *  [Numpy](http://www.numpy.org) 1.10.4
> *  [Matplotlib](https://matplotlib.org) 1.5.1
> *  [Scikit-learn](http://scikit-learn.org/stable/) 0.17.1

## Code

The main code used in this project is inside **banknote_classification.ipynb**. Another part of the code, which is optional, is inside **knn_numpy.py** located in the same folder. This file contains knn algorithm wrote from scratch, if you want to learn how it works examine the file.

## Run

To run this project you will need some software, like Anaconda, which provides support for running .ipynb files (Jupyter Notebook).

After making sure you have that, you can run from a terminal or cmd next lines:

`ipython notebook banknote_classification.ipynb`

or

`jupyter notebook banknote_classification.ipynb`




