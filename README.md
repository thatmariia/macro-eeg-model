# macro-eeg-model
### Macroscopic EEG modeling with axon propagation delays

The code simulates scalp-recorded EEG dynamics by implementing a linear macroscopic network model based on long-range axon delays. 
It uses a linear vector autoregressive framework with a few parameters to replicate features of real EEG data, 
such as resting-state alpha power and coherence.

## Getting started

#### Clone the repository and navigate to the root folder:
```sh
git clone https://github.com/thatmariia/macro-eeg-model.git
cd macro-eeg-model
```

#### Open the documentation:

* On macOS:
```sh
open docs/build/html/index.html
```

* On Linux:
```sh
firefox docs/build/html/index.html
```
(replace `firefox` with `google-chrome` or another browser if necessary).
If running on Ubuntu, you may simply run 
```sh
xdg-open docs/build/html/index.html
```

* On Windows:
```sh
start docs/build/html/index.html
```

The documentation provides all further details on how to
install required packages, run the code, and access the results.
Additionally, it contains the API reference.

## Citation

If you use this software in your work, please cite it using the following metadata.

You can use the following BibTeX entry:

```bibtex
@software{Steeghs-Turchina_Macro_EEG_model_2024,
    author = {Steeghs-Turchina, Mariia and Srinivasan, Ramesh and Nunez, Paul L. and Nunez, Michael},
    doi = {10.5281/zenodo.13692201},
    month = aug,
    title = {{Macro EEG model}},
    url = {https://github.com/thatmariia/macro-eeg-model},
    version = {1.0},
    year = {2024}
}
```

Or the following APA citation:

```
Steeghs-Turchina, M., Srinivasan, R., Nunez, P. L., & Nunez, M. (2024). Macro EEG model (Version 1.0) [Computer software]. https://doi.org/10.5281/zenodo.13692201
```

