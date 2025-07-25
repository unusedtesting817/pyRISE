import scipy.optimize as opt

def mle(model, data, initial_params):
    """
    Estimate the model parameters using Maximum Likelihood Estimation.

    Args:
        model: The model to be estimated. It must have a log_likelihood method.
        data: The data to be used for the estimation.
        initial_params: The initial values for the parameters.

    Returns:
        The estimated parameters.
    """

    def objective_function(params):
        return -model.log_likelihood(params, data)

    result = opt.minimize(objective_function, initial_params)
    return result.x
