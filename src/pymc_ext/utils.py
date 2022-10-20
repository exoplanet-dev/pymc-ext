__all__ = ["Evaluator", "eval_in_model", "sample_inference_data"]

import numpy as np
import pymc as pm


class Evaluator:
    def __init__(self, outs, **kwargs):
        if isinstance(outs, (tuple, list)):
            self.out_values = pm.aesaraf.rvs_to_value_vars(outs)
        else:
            self.out_values = pm.aesaraf.rvs_to_value_vars([outs])[0]
        self.in_values = pm.inputvars(self.out_values)
        self.func = pm.aesaraf.compile_pymc(
            self.in_values, self.out_values, **kwargs
        )

    def __call__(self, point):
        args = [point[x.name] for x in self.in_values]
        return self.func(*args)


def eval_in_model(outs, point=None, model=None, seed=None, **kwargs):
    """Evaluate a Theano tensor or PyMC3 variable in a PyMC3 model

    This method builds a Theano function for evaluating a node in the graph
    given the required parameters. This will also cache the compiled Theano
    function in the current ``pymc3.Model`` to reduce the overhead of calling
    this function many times.

    Args:
        outs: The variable, tensor, or list thereof to evaluate.
        point (Optional): A ``dict`` of input parameter values. This can be
            ``model.initial_point`` (default), the result of ``pymc.find_MAP``,
            a point in a ``pymc3.MultiTrace`` or any other representation of
            the input parameters.
    """
    if point is None:
        model = pm.modelcontext(model)
        point = model.initial_point(seed=seed)
    return Evaluator(outs, **kwargs)(point)


def sample_inference_data(idata, size=1, random_seed=None, group="posterior"):
    """Generate random samples from an InferenceData object

    Args:
        idata: The ``InferenceData``.
        size: The number of samples to generate.
        random_seed: The seed for the random number generator.
        group: The ``InferenceData`` group to sample from.
    """
    random = np.random.default_rng(random_seed)
    idata = idata[group].stack(sample=("chain", "draw"))
    num_sample = idata.dims["sample"]

    for i in random.integers(num_sample, size=size):
        yield {k: v.values for k, v in idata.isel({"sample": i}).items()}


def sample(**kwargs):
    kwargs["target_accept"] = kwargs.get("target_accept", 0.9)
    kwargs["init"] = kwargs.get("init", "adapt_full")
    return pm.sample(**kwargs)
