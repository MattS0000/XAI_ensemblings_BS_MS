# `EnsembleXAI` python package

## Build

To build from source run following command in main project directory.

```python -m build```

It will create `dist` directory with build archives.

## Installation

To install the package use pip. Run the following command from main project directory.

```pip install dist/EnsembleXAI-0.0.1.tar.gz```


## Documntation

Documentation available through file [./docs/_build/html/index.html](./docs/_build/html/index.html)


## Directory structure

Such directory structure enables using Kaggle notebooks without changing paths to input files in the notebook.

- Main project directory:
    + XAI_ensemblings_BS_MS (git repo)
    + input:
        * crop-and-weed-detection-data-with-bounding-boxes/agri_data/data/{.jpeg's, .txt's} (downloaded from kaggle)
        * inz-data-prep - requires manual creation
    + python venv
