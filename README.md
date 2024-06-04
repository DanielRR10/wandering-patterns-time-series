# Dementia Wandering Patterns Analysis

This repository contains the code for a research paper that addresses the intricate challenge of dementia-related wandering using advanced techniques. 

## Overview

With the aging of the global population, dementia has emerged as a significant health concern, particularly among the elderly. This degenerative brain disorder leads to cognitive decline, encompassing memory loss, impaired communication skills, reduced ability to perform routine tasks, and shifts in personality and mood. 

While dementia lacks a definitive cure, accurate diagnosis and suitable treatment can greatly enhance the quality of life for those affected. Wandering behavior is common in individuals with dementia, and a link between wandering patterns and the disease’s severity has been established.

## Methodology

Our work utilizes data imputation methods and feature extraction via Discrete Wavelet Transformation on an extensive dataset. These methodologies are coupled with established machine learning algorithms, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest (RF). Furthermore, Bayesian optimization is employed to fine-tune model performance.

## Results

Promising outcomes are witnessed in our approach, yielding an accuracy of 98%. This underscores the potential of our methodology in effectively categorizing wandering patterns linked to dementia cases. Our results offer valuable insights into the contribution of advanced data analysis and machine learning techniques to the realm of dementia research and caregiving.

## Files in this Repository


- `model_data/patterns_dataset.pkl`: This file contains the datasets used in this project.
- `models_analysis/`: This directory contains the support code used to automating the model evaluation 
  - `bo_models_evaluation.py`: This script is used for defining functions to apply while the bayesian optimization.
  - `classic_models_evaluation.py`: This script contains the machine learning models (KNN, SVM, RF) and the Bayesian optimization process.
  - `raw_scaled_data.py`: This script is used for data preprocessing.
  - `time_series_model.py`: This script define the TimeSeriesData class used for the analysis.
  - `utils.py`: This script is used to graph a confusion matrix.
  - `wave_scaled_data.py`: This script is used for feature extraction using Discrete Wavelet Transformation.
- `2 Analysis Time Series Default Params.ipynb`: This Jupyter notebook contains the training and results of the models using their default params and raw data.
- `3 Analysis Time Series Bayesian Optimization.ipynb`: This Jupyter notebook contains the training and results of the models after tunning their hyperparams using the Bayesian Optimization raw data.
- `4 Analysis Time Series Signals.ipynb`: This Jupyter notebook contains the training and results of the models using the data after the feature extraction by using the Discrete Wavelet transformation 
- `requirements.txt`: This file lists all the Python dependencies required to run the project.

## How to Run the Code

1. Clone the repository to your local machine.

```bash
git clone https://github.com/DanielRR10/wandering-patterns-time-series.git
```
2. Navigate to the project directory.
```bash
cd wandering-patterns-time-series
```
3. Installl the required Python dependencies.
```bash
pip install -r requirements.txt
```
4. To run the Jupyter notebooks, start Jupyter notebook server in the project directory.
```bash
jupyter notebook
```
This will open a browser window. From there, you can select and run the notebooks

## Dependencies

All dependencies are listed in the requirements.txt

## Contact

For any queries, please contact the authors of the paper.