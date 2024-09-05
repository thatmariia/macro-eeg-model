# Preparation

Before running the code, make sure you have installed Python 3.10 or higher
(the project is initially developed with 3.10 and tested with that version).

Create a new conda environment and activate it:

```sh
conda create -n macro-eeg-model python=3.10
conda activate macro-eeg-model
```

Install the packages with pip: 
```sh
pip install -e .
```
This will install the project and its dependencies in editable mode, 
allowing you to make changes to the source code and have them reflected immediately.
You only need to do this once, unless you change the dependencies 
in `requirements.txt` or the contents of `setup.py`.
