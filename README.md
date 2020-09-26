# PyMC3 Extras & Extenstions

This library include various experimental or otherwise special purpose extras for use with PyMC3 that have been extracted from the [exoplanet](https://docs.exoplanet.codes) project.
The most widely useful component is probably the custom tuning schedule that is [described below](#NUTS-tuning-schedule), but it also includes some helper functions for [non-linear optimization](#Optimization) and [some custom distributions](#Distributions).

You'll find the usage instructions below and automatically generated tutorial notebooks on the [`tutorials`](https://github.com/exoplanet-dev/pymc3-ext/tree/tutorials) branch.

## Installation

You'll need a Python installation (tested on versions 3.7 and 3.8) and it can often be best to install PyMC3 and Theano using [`conda`](https://docs.conda.io/en/latest/) so that it can handle all the details of compiler setup.
This step is **optional**, but I would normally create a clean conda environment for projects that use PyMC3:

```bash
# Optional
conda create -n name-of-my-project python=3.8 pymc3
conda activate name-of-my-project
```

The easiest way to install this package is using `pip`:

```bash
python -m pip install -U pymc3-ext
```

This will also update the dependencies like PyMC3 and Theano, which is probably what you want because this is only tested on recent versions of both of those packages.

## NUTS tuning schedule


## Optimization


## Distributions
