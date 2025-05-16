# mastercard_challenge

![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)
[//]: # (![Build Status]&#40;https://github.com/<USERNAME>/<REPO>/actions/workflows/tests.yml/badge.svg&#41;)
![Coverage](https://img.shields.io/codecov/c/github/<USERNAME>/<REPO>.svg)
![License](https://img.shields.io/github/license/<USERNAME>/<REPO>.svg)
![Last Commit](https://img.shields.io/github/last-commit/<USERNAME>/<REPO>.svg)
![Notebooks](https://img.shields.io/badge/notebooks-%E2%9C%94-green.svg)
![Project Stage](https://img.shields.io/badge/stage-dev-lightgrey.svg)

"SGH x Mastercard Hackathon - May 2025"

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mastercard_challenge and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mastercard_challenge   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mastercard_challenge a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

