User Installation
=================

Usage of a virtual environment is strongly encouraged.

#. Clone the repository from github.
#. Install the requirements in commandline.

    #. Activate virtual environment if applicable.
    #. `pip install -r /correct_path/XAI_ensemblings_BS_MS/requirements.txt`.

Downloading as a package

`pip install git+https://github.com/MattS0000/XAI_ensemblings_BS_MS#egg=EnsembleXAI&subdirectory=EnsembleXAI`

currently unavailable due to repository currently being private.

Package Development Additional Installation
===========================================
To run documentation creation additional packages are required:

#. Nbsphinx extension to convert notebooks to html requires seperate installation of `pandocs`, installation with conda is recommended.
#. Optionally activate the virtual environment.
#. Run `pip install sphinx sphinx_rtd_theme nbsphinx`.
#. Optionally for a clean install run `.\\docs\\make clean`.
#. Then in the project directory run: `.\\docs\\make html`.