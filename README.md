awesome_deepLearning_remoteSensing
==============================

_**a project to apply my knowledge in  remote sensing and deep learning**_

This repository is intended to accommodate my passion and learning process in advancing Earth Observation(EO) techniques 
using Deep Learning.

Currently, I am very interested with the **transfer learning**, **self-supervised learning**, and **foundation model**. It is driven by the special case
of Earth Observation data that the mainstream computer vision treatment can not be straightly implemented to the Satellite 
Image. For instance, satellite imagery could have more than 3 channels or even hundreds where each channel has correlation to certain Earth
properties. On the other hand, Deep Learning has a big potential power to understand the earth objects with the current abundance satellite data.  

Goals and what's currently going
------------
1. Implementing production standard of deep learning project. Some resources I follow to implement it are:
   - [Developmentseed.org](https://developmentseed.org/blog/2022-09-27-ml-tooling). They talk about how they implement their GeoML project
   - QuantumBlack, AI by McKinsey in the article about [production code](https://medium.com/quantumblack/data-scientists-the-only-useful-code-is-production-code-8b4806f2fe75)
   - [fullstackdeeplearning.com](https://fullstackdeeplearning.com). Explaining how the DL project from end-to-end.
2. Training, Testing, and Deploying the deepl learning model for Land Cover analysis.
   - Currently I am trying to reproduce the [MAE](https://www.google.com/search?client=safari&rls=en&q=mask+autoencoder&ie=UTF-8&oe=UTF-8) to learn earth representation. 
     This modelling framework is interesting in my opinion. And also, it is suitable for the EO cases and my goals

Project Organization
------------

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
