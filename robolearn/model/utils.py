import torch
from torch.distributions import uniform


class ExponentialScheduler:
    """Exponential learning rate schedule for Langevin sampler."""

    def __init__(self, init, decay):
        self._decay = decay
        self._latest_lr = init

    def get_rate(self, index):
        """Get learning rate. Assumes calling sequentially."""
        del index
        self._latest_lr *= self._decay
        return self._latest_lr


class PolynomialScheduler:
    """Polynomial learning rate schedule for Langevin sampler."""

    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        """Get learning rate for index."""
        return (
            (self._init - self._final)
            * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))
        ) + self._final


def compute_grad_norm(de_da, grad_norm_type):
    """Given de_dact and the type, compute the norm."""
    ord_dict = {"inf": float("inf"), "1": 1, "2": 2}

    if grad_norm_type is not None:
        grad_norms = torch.linalg.norm(de_da, axis=1, ord=ord_dict[grad_norm_type])
    else:
        # It will be easier to manage downstream if we just fill this with zeros.
        # Rather than have this be potentially a None type.
        grad_norms = torch.zeros_like(de_da[:, 0])
    return grad_norms


# TODO: update to pass variable dimension
def sample_uniform_actions(
    batch_size,
    num_counter_examples,
    max_sampling_specs,
    min_sampling_specs,
):
    distribution = uniform.Uniform(min_sampling_specs, max_sampling_specs)
    random_actions = distribution.sample([batch_size, num_counter_examples])

    return random_actions
