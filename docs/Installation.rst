Installation
============

Usage of a virtual environment is strongly encouraged.
Clone the repository from github. Install the requirements in commandline using pip or conda.

`pip install -r /correct_path/XAI_ensemblings_BS_MS/requirements.txt`

# To be tested, downloading as a package
`pip install git+https://github.com/MattS0000/XAI_ensemblings_BS_MS#egg=EnsembleXAI&subdirectory=EnsembleXAI`

To run documentation creation additional packages are required:

`pip install sphinx sphinx_rtd_theme nbsphinx`

nbsphinx extension to convert notebooks to html requires installed `pandocs`, installation with conda recommended.
Then in the project directory run:

`.\\docs\\make html`

For a clean install run `.\\docs\\make clean` before that.
If using a virtual environment, it must be activated before the command.