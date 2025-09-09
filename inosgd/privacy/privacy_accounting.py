from opacus.accountants import create_accountant
from scipy import optimize


EPSILON_TOL = 0.01
DEFAULT_ALPHAS = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 99)) + list(range(100, 999, 10))


def get_sampling_rate(
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        n_iters: int
    ) -> float:
    r"""
    Calculates the proper sampling rate that achieves :math:`(\epsilon, \delta)`-DP in `n_iters` iterations given `noise_multiplier`, using a bisection method.

    Args:
        target_epsilon: The target :math:`\epsilon` to satisfy.
        target_delta: The target :math:`\delta` to satisfy.
        noise_multiplier: The noise scale divided by clipping threshold.
        n_iters: Total number of iterations.
    """
    eps_high = float('inf')
    accountant = create_accountant(mechanism='rdp')

    q_low, q_high = 0.0, 1.0

    accountant.history = [(noise_multiplier, q_high, n_iters)]
    eps_high = accountant.get_epsilon(delta=target_delta, alphas=DEFAULT_ALPHAS)
    if eps_high < target_epsilon:
        return 1.0
    
    def bisect_objective(q):
        accountant.history = [(noise_multiplier, q, n_iters)]
        epsilon = 0 if q == 0 else accountant.get_epsilon(delta=target_delta, alphas=DEFAULT_ALPHAS)   
        return epsilon - target_epsilon
    
    return optimize.bisect(bisect_objective, q_low, q_high, maxiter=100, disp=False)


def get_sampling_rate_new(
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    steps: int,
    accountant: str = "rdp",
    precision: float = 0.001,
    **kwargs,
) -> float:
    r"""
    Computes via binary search the sampling frequency q to reach a total budget
    of (target_epsilon, target_delta) at the end of epochs, with a given
    noise_multiplier.
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        noise_multiplier: relation between noise std and clipping threshold
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        precision: relation between limits of binary search interval
    Returns:
        The sampling frequency q to ensure privacy budget of
        (target_epsilon, target_delta)
    """
    accountant = create_accountant(mechanism=accountant)
    q_low, q_high = 1e-9, 0.5
    accountant.history = [(noise_multiplier, q_low, steps)]
    eps_low = accountant.get_epsilon(delta=target_delta, **kwargs)
    if eps_low > target_epsilon:
        raise ValueError("The privacy budget is too low.")
    accountant.history = [(noise_multiplier, q_high, steps)]
    eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
    while eps_high < 0:     # decrease q_high whenever a numerical error happens
        q_high *= 0.9
        accountant.history = [(noise_multiplier, q_high, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
    if eps_high < target_epsilon:
        raise ValueError(f"The given noise_multiplier {noise_multiplier} is "
                         f"too high.")

    while q_low / q_high < 1 - precision:
        q = (q_low + q_high) / 2
        accountant.history = [(noise_multiplier, q, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)
        if eps < target_epsilon:
            q_low = q
        else:
            q_high = q

    return q_low
