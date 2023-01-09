# `EnsembleXAI` python package

## Build

To build from source run following command in main project directory.

```bash
python -m build
```

It will create `dist` directory with build archives.

## Installation

To install the package use pip. Having build the package, just run the following command from main project directory.

```bash
 pip install dist/EnsembleXAI-0.0.1.tar.gz
 ```

If you have access to project GitHub repository you can download, build and install the package via command:

```bash
pip install git+https://github.com/MattS0000/XAI_ensemblings_BS_MS.git
```

## Documentation

Documentation available through file [./docs/_build/html/index.html](./docs/_build/html/index.html)


## Directory structure

Such directory structure enables using Kaggle notebooks without changing paths to input files in the notebook.

- Main project directory:
    + XAI_ensemblings_BS_MS (git repo)
    + input:
        * crop-and-weed-detection-data-with-bounding-boxes/agri_data/data/{.jpeg's, .txt's} (downloaded from kaggle)
        * inz-data-prep - requires manual creation
    + python venv
