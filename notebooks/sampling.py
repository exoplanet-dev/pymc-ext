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
# `pymc3-ext` comes with some functions to make sampling more flexible in some cases and improve the default parameter choices for the types of problems encountered in astrophysics.
# These features are accessed through the `pymc3_ext.sample` function that behaves mostly like the `pymc3.sample` function with a couple of different arguments.
# The two main differences for all users is that the `pymc3_ext.sample` function defaults to a target acceptance fraction of `0.9` (which will be better for many models in astrophysics) and to adapting a full dense mass matrix (instead of diagonal).
# Therefore, if there are covariances between parameters, this method will generally perform better than the PyMC3 defaults.
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
# But, we can use `pymc3_ext.sample` as a drop in replacement to get much better performance:

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
# In this particular case, you could get similar performance using the `init="adapt_full"` argument to the `sample` function in PyMC3, but the implementation in `pymc3-ext` is somewhat more flexible.
# Specifically, `pymc3_ext` implements a tuning procedure that it more similar to [the one implemented by the Stan project](https://mc-stan.org/docs/2_24/reference-manual/hmc-algorithm-parameters.html).
# The relevant parameters are:
#
# - `warmup_window`: The length of the initial "fast" window. This is called "initial buffer" in the Stan docs.
# - `adapt_window`: The length of the initial "slow" window. This is called "window" in the Stan docs.
# - `cooldown_window`: The length of the final "fast" window. This is called "term buffer" in the Stan docs.
#
# Unlike the Stan implementation, here we have support for updating the mass matrix estimate every `recompute_interval` steps based on the previous window and all the steps in the current window so far.
# This can improve warm up performance substantially so the default value is `1`, but this might be intractable for high dimensional models.
# To only recompute the estimate at the end of each window, set `recompute_interval=0`.
#
# If you run into numerical issues, you can try increasing `adapt_window` or use the `regularization_steps`and `regularization_variance` to regularize the mass matrix estimator.
# The `regularization_steps` parameter sets the effective number of steps that are used for regularization and `regularization_variance` is the effective variance for those steps.

# %% [markdown]
# ## Parameter groups
#
# If you are fitting a model with a large number of parameters, it might not be computationally or numerically tractable to estimate the full dense mass matrix.
# But, sometimes you might know something about the covariance structure of the problem that you can exploit.
# Perhaps some parameters are correlated with each other, but not with others.
# In this case, you can use the `parameter_groups` argument to exploit this structure.
#
# Here is an example where `x`, `y`, and `z` are all independent with different covariance structure.
# We can take advantage of this structure using `pmx.ParameterGroup` specifications in the `parameter_groups` argument.
# Note that by default each group will internally estimate a dense mass matrix, but here we specifically only estimate a diagonal mass matrix for `z`.

# %%
with pm.Model():
    x = pm.MvNormal("x", mu=np.zeros(ndim), chol=L, shape=ndim)
    y = pm.MvNormal("y", mu=np.zeros(ndim), chol=L, shape=ndim)
    z = pm.Normal("z", shape=ndim)  # Uncorrelated

    tracex2 = pmx.sample(
        tune=1000,
        draws=1000,
        chains=2,
        cores=2,
        parameter_groups=[
            [x],
            [y],
            pmx.ParameterGroup([z], "diag"),
        ],
    )

# %%
