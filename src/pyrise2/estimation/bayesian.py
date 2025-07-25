import emcee
import numpy as np

def bayesian_estimation(model, data, initial_params, nwalkers=100, nsteps=1000):
    """
    Estimate the model parameters using Bayesian estimation.

    Args:
        model: The model to be estimated. It must have a log_probability method.
        data: The data to be used for the estimation.
        initial_params: The initial values for the parameters.
        nwalkers: The number of walkers to use in the MCMC simulation.
        nsteps: The number of steps to take in the MCMC simulation.

    Returns:
        The estimated parameters.
    """

    def log_prob(params):
        return model.log_probability(params, data)

    ndim = len(initial_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    p0 = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler.get_chain(discard=100, thin=1, flat=True)
