# Reproducing results

To reproduce the slow wave dynamics of scalp EEG, the simulations are to be run (and evaluated) with the parameters outlined below.

* Differing parameters:

|                         | relay    | dist params                                                                             | noise_color | std_noise | connectivity weights  |
|-------------------------|----------|-----------------------------------------------------------------------------------------|-------------|-----------|-----------------------|
| `Simulation #1 (white)` | none     | Custom (-0.25, 0.09, 0.25)                                                              | white       | 1950      | Custom                |
| `Simulation #1 (pink)`  | none     | Custom (-0.25, 0.09, 0.25)                                                              | pink        | 27000     | Custom                |
| `Simulation #2 (white)` | thalamus | [Sepehrband et al., 2016](https://doi.org/10.3389/fnana.2016.00059) (-0.21, 0.16, 0.43) | white       | 1950      | BOLD-based            |
| `Simulation #2 (pink)`  | thalamus | [Sepehrband et al., 2016](https://doi.org/10.3389/fnana.2016.00059) (-0.21, 0.16, 0.43) | pink        | 27000     | BOLD-based            |

* Common parameters:

| nodes                                                               | sample_rate | t_lags | t_secs | t_burnit | dist_trunc_percent |
|---------------------------------------------------------------------|-------------|--------|--------|----------|--------------------|
| frontal lobe; parietal lobe; occiptal lobe; temporal lobe; thalamus | 1000        | 300    | 500    | 10       | 0.0                |

The sequence of steps to reproduce the results is outlined in the `reproduce.sh` script and can be run in one go:
```sh
./reproduce.sh
```