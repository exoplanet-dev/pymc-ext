# PyMC Extras

This library includes various experimental or otherwise special-purpose extras
for use with [PyMC](https://www.pymc.io) that have been extracted from the
[exoplanet](https://docs.exoplanet.codes) project. It's primary components are
some helper functions for [non-linear optimization](#Optimization) and [some
custom distributions](#Distributions).

## Installation

You'll need a Python installation, and it can often be best to install PyMC
using [`conda`](https://docs.conda.io/en/latest/) so that it can handle all the
details of compiler setup. This step is **optional**, but I would normally
create a clean conda environment for projects that use PyMC:

```bash
# Optional
conda create -n name-of-my-project pymc
conda activate name-of-my-project
```

The easiest way to install this package is using `pip`:

```bash
python -m pip install -U pymc-ext
```

This will also update the dependencies like PyMC, which is probably what you
want because this is only tested on recent versions of that package.

## Optimization

When PyMC added a warning to the `pm.find_MAP` function, we implemented a custom
non-linear optimization framework in `exoplanet` because it is often useful to
be able to optimize (at least) some parameters when initializing the sampler for
many problems in astrophysics (and probably elsewhere). While `pm.find_MAP` no
longer complains, the `pymc_ext.optimize` function is included here for backward
compatibility even though it should have similar behavior to `pm.find_MAP`. To
use this function, you'll do something like the following:

```python
import pymc_ext as pmx

with model:
    soln = pmx.optimize(vars=[var1, var2])
    soln = pmx.optimize(start=soln, vars=[var3])
```

## Distributions

Most of the custom distributions in this library are there to make working with
periodic parameters (like angles) easier. All of these reparameterizations could
be implemented manually without too much trouble, but it can be useful to have
them in a more compact form. Here is a list of the included distributions and
short descriptions:

- `pmx.unit_disk`: Two dimensional parameters constrained to live within the
  unit disk. This will be useful when you have an angle and a magnitude that
  must be in the range from zero to one (for example, an eccentricity vector for
  a bound orbit). This distribution is constrained such that the sum of squares
  along the zeroth axis will always be less than one. Note that the shape of
  this distribution must be two in the zeroth axis.
- `pmx.angle`: An angle constrained to be in the range -pi to pi. The actual
  sampling is performed in the two-dimensional vector space `(sin(theta), cos(theta))`
  so that the sampler doesn't see a discontinuity at pi. As a
  technical detail, the performance of this distribution can be affected using
  the `regularization` parameter which helps deal with pathological geometries
  introduced when this parameter is well/poorly constrained. The default value
  (`10.0`) was selected as a reasonable default choice, but you might get better
  performance by adjusting this.

It's important to note that these are not `Distribution` _objects_, but rather
functions that will add `Distribution` objects to the model, and return the
reparameterized variable of interest. The ergonomics of this interface are
questionable, but it's easier to maintain this interface than one that
implements custom `Distribution` objects.

## License

Copyright 2020-2022 Dan Foreman-Mackey and contributors.

pymc-ext is free software made available under the MIT License. For details see
the LICENSE file.
