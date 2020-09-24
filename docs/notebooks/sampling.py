# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%
# %run notebook_setup

# %% [markdown]
# # Sampling
#
# *pymc3-ext* comes with some functions to make sampling more flexible in some cases and improve the default parameter choices for the types of problems I work on.
# These features are accessed through the `pymc3_ext.sample` function that behaves mostly like the `pymc3.sample` function with a couple of different arguments.
# The two main differences for all users is that the `pymc3_ext.sample` function defaults to a target acceptance fraction of `0.9` (which will be better for most problems in my experience) and to adapting a full dense mass matrix (instead of diagonal) using the schedule described in the TBD section.
# Therefore, if there are covariances between parameters, this method will generally perform better.
#
# ## Correlated parameters
#
# A thorough discussion of this [can be found elsewhere online](https://dfm.io/posts/pymc3-mass-matrix/), but here is a simple demo where we sample a covariant Gaussian using `pymc3_ext.sample`.
#
# First, we generate a random positive definite covariance matrix for the Gaussian:

# %%
import numpy as np

ndim = 5
np.random.seed(42)
L = np.random.randn(ndim, ndim)
L[np.diag_indices_from(L)] = 0.1 * np.exp(L[np.diag_indices_from(L)])
L[np.triu_indices_from(L, 1)] = 0.0
cov = np.dot(L, L.T)

# %% [markdown]
# And then we can set up this model using PyMC3:

# %%
import pymc3 as pm

with pm.Model() as model:
    pm.MvNormal("x", mu=np.zeros(ndim), chol=L, shape=ndim)

# %% [markdown]
# If we sample this using PyMC3 default sampling method, things don't go so well (we're only doing a small number of steps because we don't want it to take forever, but things don't get better if you run for longer!):

# %%
with model:
    trace = pm.sample(tune=500, draws=500, chains=2, cores=2)

# %% [markdown]
# But, we can use `pymc3.sample` as a drop in replacement to get much better performance:

# %%
import pymc3_ext as pmx

with model:
    tracex = pmx.sample(tune=1000, draws=1000, chains=2, cores=2)

# %% [markdown]
# As you can see, this is substantially faster (even though we generated twice as many samples).
#
# We can compare the sampling summaries to confirm that the default method did not produce reliable results in this case, while the `pymc3_ext` version did:

# %%
pm.summary(trace).head()

# %%
pm.summary(tracex).head()

# %% [markdown]
# ## Parameter groups
#
# If you are fitting a model with a large number of parameters, it might not be computationally or numerically tractable to

# %%
with pm.Model() as model2:
    pm.MvNormal("x", mu=np.zeros(ndim), chol=L, shape=ndim)
    pm.MvNormal("y", mu=np.zeros(ndim), chol=L, shape=ndim)
    pm.Normal("z", shape=ndim)  # Uncorrelated

# %%
with model2:
    tracex2 = pmx.sample(tune=1000, draws=1000, chains=2, cores=2)

# %%
with model2:
    tracex2 = pmx.sample(
        tune=1000,
        draws=1000,
        chains=2,
        cores=2,
        parameter_groups=[
            pmx.sampling.ParameterGroup([model2.x]),
            pmx.sampling.ParameterGroup([model2.y]),
            pmx.sampling.ParameterGroup([model2.z], "diag"),
        ],
    )

# %%
