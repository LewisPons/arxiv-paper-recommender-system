arxiv-paper-recommender
==============================

## Table of Contents

- [arxiv-paper-recommender](#arxiv-paper-recommender)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
- [Project Organization](#project-organization)

## Overview

This  Paper Recommendation System is a natural language processing (NLP) project that aims to build a recommendation system for machine learning (ML) and artificial intelligence (AI) papers. The system utilizes a trained model to recommend relevant papers based on the summaries of ML/AI papers. This repository contains the source code and resources necessary to develop, train, and evaluate the recommendation system.


## Installation

To install and set up the ML Paper Recommendation System, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/LewisPons/arxiv-paper-recommender-system.git
   cd arxiv-paper-recommender
   ```

2. Set up the Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The AI & ML Paper Recommendation System utilizes a corpus of summaries of ML/AI papers as the training dataset. [This dataset](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts) contains paper titles, paper abstracts, and their subject categories collected from the arXiv portal.

-----

## Model Architecture
The ML Paper Recommendation System employs a state-of-the-art NLP model architecture for recommendation purposes. The architecture consists of [provide brief details about the model architecture].

For more detailed information about the model architecture and implementation, please refer to the source code and documentation within the repository.

-----


## Results
[Results in this section]




# Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
****



