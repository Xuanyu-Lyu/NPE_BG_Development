"""One-dimensional AE prior and posterior for the fixed-variance NPE study."""

import torch
from torch.distributions import Distribution, constraints
from torch.nn import functional as F


class AELogitPrior(Distribution):
    """Beta(1, 1) on A, represented as log(A/E), with E=1-A."""

    arg_constraints = {}
    support = constraints.independent(constraints.real, 1)
    has_rsample = False

    def __init__(self, device="cpu", validate_args=None):
        self.alpha = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.beta = torch.tensor(1.0, dtype=torch.float32, device=device)
        super().__init__(torch.Size(), torch.Size([1]), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        a = torch.distributions.Beta(self.alpha, self.beta).sample(
            torch.Size(sample_shape)
        )
        a = a.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(a).unsqueeze(-1)

    def log_prob(self, value):
        z = value[..., 0]
        # Beta(1,1) is uniform in A; only the logit Jacobian remains.
        return F.logsigmoid(z) + F.logsigmoid(-z)

    def to(self, device):
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        return self


class AEPosterior:
    """Expose a one-dimensional latent posterior as exact-boundary ACE draws."""

    def __init__(self, latent_posterior):
        self.latent_posterior = latent_posterior

    def sample(self, sample_shape=torch.Size(), x=None, **kwargs):
        latent = self.latent_posterior.sample(sample_shape, x=x, **kwargs)
        a = torch.sigmoid(latent[..., 0])
        return torch.stack((a, torch.zeros_like(a), 1.0 - a), dim=-1)

    def to(self, device):
        self.latent_posterior.to(device)
        return self
