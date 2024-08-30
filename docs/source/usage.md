# Usage

## Preparation

Before running the code, make sure you have installed Python 3.10 or higher
(the project is initially developed with 3.10 and tested with that version).

Install the packages with pip: 
```sh
pip install -e .
```
This will install the project and its dependencies in editable mode, 
allowing you to make changes to the source code and have them reflected immediately.
You only need to do this once, unless you change the dependencies 
in `requirements.txt` or the contents of `setup.py`.

## Running the code

The project provides two command line tools: one to run the simulations and another to evaluate them.

### Running simulation

To run a simulation, use this command:
```sh
py_simulate [options]
```

After running the simulation, the resulting data is saved in the `output/<model_name>` folder, 
and the generated plots are saved in the `plots/<model_name>` folder.

### Running evaluation

To run the evaluation, use this command:
```sh
py_evaluate [options]
```

After running the evaluation, the resulting plots are saved in the `plots` folder.

### Arguments (`[options]`)

By default, the commands use parameters from the following configuration file: `configs/model_params.yml`.

You can modify this YAML file to change the default parameters.
Alternatively, you can provide the following arguments to the commands. They will override the default parameters.

| arg name                | type (+ options)                                | default                                                                 | help                                                                                  |
|-------------------------|-------------------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `--model_name`          | `str`                                           | `"Simulated macro EEG model"`                                           | name of the model                                                                     |
| `--nodes`               | `str` (a list of labels separated by semicolon) | `"frontal lobe; parietal lobe; occiptal lobe; temporal lobe; thalamus"` | brain areas where the nodes are placed (according to Julich brain labels)             |
| `--relay_station`       | `str` (a label or `"none"`)                     | `"none"`                                                                | brain area to use as a relay station (according to Julich brain labels or 'none')     |
| `--custom_connectivity` | `bool`                                          | `True`                                                                  | whether to use custom connectivity (from connectivity_weights.csv) or not             |
| `--sample_rate`         | `int`                                           | `1000`                                                                  | sample rate                                                                           |
| `--t_lags`              | `int`                                           | `300`                                                                   | lagged time in ms                                                                     |
| `--t_secs`              | `int`                                           | `500`                                                                   | simulation time in seconds                                                            |
| `--t_burnit`            | `int`                                           | `10`                                                                    | number of seconds to delete to ensure model convergence                               |
| `--noise_color`         | `str`                                           | `"white"`                                                               | color of the noise ('white' or 'pink')                                                |
| `--std_noise`           | `float`                                         | `1950`                                                                  | scalar standard deviation of the noise (effectively controls the scale of the output) |
| `--dist_shape`          | `float`                                         | `-0.25`                                                                 | shape param for the lag distribution.                                                 |
| `--dist_scale`          | `float`                                         | `0.09`                                                                  | scale param for the lag distributions                                                 |
| `--dist_location`       | `float`                                         | `0.25`                                                                  | location param for the lag distributions                                              |
| `--dist_trunc_percent`  | `float`                                         | `0.0`                                                                   | tail truncation percentile for the lag distributions                                  |

As you can see from the `--custom_connectivity` argument, you can specify a custom connectivity weights matrix in the `configs/connectivity_weights.csv` file. 
Use either a symmetrical of an upper triangular matrix (it will be converted to symmetrical). The diagonal values are ignored.

