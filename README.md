# PyMC3 Extras

This library include various experimental or otherwise special purpose extras for use with PyMC3 that have been extracted from the [exoplanet](https://docs.exoplanet.codes) project.
The most widely useful component is probably the custom tuning functions for the PyMC3 NUTS sampler that is [described below](#NUTS-tuning), but it also includes some helper functions for [non-linear optimization](#Optimization) and [some custom distributions](#Distributions).

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

## NUTS tuning


## Optimization

When PyMC3 added a warning to the `pm.find_MAP` function, we implemented a custom non-linear optimization framework in `exoplanet` because it is often useful to be able to optimize (at least) some parameters when initializing the sampler for many problems in astrophysics (and probably elsewhere).
While `pm.find_MAP` no longer complains, the `pymc3_ext.optimize` function is included here for backwards compatibility even though it should have similar behavior to `pm.find_MAP`.
To use this function, you'll do something like the following:

```python
import pymc3_ext as pmx

with model:
    soln = pmx.optimize(vars=[var1, var2])
    soln = pmx.optimize(start=soln, vars=[var3])
```

You can find more examples in the Optimization tutorial notebook.

## Distributions

Most of the custom distributions in this library are there to make working with periodic parameters (like angles) easier.
All of these reparameterizations could be implemented manually without too much trouble, but it can be useful to have them in a more compact form.
Here is a list of the included distributions and a short description:

- `pmx.UnitVector`: A vector where the sum of squares is fixed to unity. For a multidimensional shape, the normalization is performed along the last dimension.
- `pmx.UnitDisk`: Two dimensional parameters constrianed to live within the unit disk. This will be useful when you have an angle and a magnitude that must be in the range zero to one (for example, an eccentricity vector for a bound orbit). This distribution is constrained such that the sum of squares along the zeroth axis will always be less than one. Note that the shape of this distribution must be two in the zeroth axis.
- `pmx.Angle`: An angle constrained to be in the range -pi to pi. The actual sampling is performed in the two dimensional vector space ``(sin(theta), cos(theta))`` so that the sampler doesn't see a discontinuity at pi. As a technical detail, the performance of this distribution can be affected using the `regularization` parameter which helps deal with pathelogical geometries introduced when this parameter is well/poorly constrained. The default value (`10.0`) was selected as a reasonable default choice, but you might get better performance by adjusting this.
- `pmx.Periodic`: An extension to `pmx.Angle` that supports arbitrary upper and lower bounds for the allowed range.
- `pmx.UnitUniform`: This distribution is equivalent to `pm.Uniform(lower=0, upper=1)`, but it can be more numerically stable in some cases.
