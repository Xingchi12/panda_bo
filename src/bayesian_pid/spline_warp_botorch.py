import torch
from botorch.models.transforms.input import InputTransform
from gpytorch import Module as GPyTorchModule
from gpytorch.priors import NormalPrior
from torch import Tensor, nn


class MonotoneSplineWarp(InputTransform, GPyTorchModule):
    """Independent learned C1 rational-quadratic splines on [0, 1]^d.

    For each dimension and K bins, there are K-1 free height logits and K-1
    log derivatives at the internal knots: 2*(K-1) parameters per dimension.
    Equal segment heights and unit derivatives initialize the identity map.
    The Gaussian priors encourage the transform to remain near identity.

    This implementation assumes a single (unbatched) output model and warps
    all d dimensions. It has no learnable knot locations.
    """

    is_one_to_many = False
    transform_on_train = True
    transform_on_eval = True
    transform_on_fantasize = True

    def __init__(self, d: int, num_bins: int = 3, prior_sd: float = 0.5) -> None:
        super().__init__()
        if d < 1 or num_bins < 2 or prior_sd <= 0:
            raise ValueError("Require d >= 1, num_bins >= 2 and prior_sd > 0.")
        self.d = d
        self.num_bins = num_bins

        # The last height logit is fixed to zero; softmax yields positive
        # heights summing to one. The boundary derivatives are fixed at one.
        self.height_logits = nn.Parameter(torch.zeros(d, num_bins - 1))
        self.log_internal_derivatives = nn.Parameter(torch.zeros(d, num_bins - 1))

        # GPyTorch includes registered priors in the exact marginal log
        # likelihood, so MAP fitting regularizes the flexible input map.
        self.register_prior(  # type: ignore
            "height_logits_prior", NormalPrior(0.0, prior_sd), "height_logits"
        )
        self.register_prior(  # type: ignore
            "log_derivatives_prior",
            NormalPrior(0.0, prior_sd),
            "log_internal_derivatives",
        )

    def transform(self, X: Tensor) -> Tensor:  # noqa: N803
        """Apply the learned spline without changing X's shape or dtype."""
        if X.shape[-1] != self.d:
            raise ValueError(f"Expected {self.d} input dimensions; got {X.shape[-1]}.")
        if X.device != self.height_logits.device or X.dtype != self.height_logits.dtype:
            raise ValueError("Move the warp/model to X's dtype and device first.")
        if torch.any((X < -1e-7) | (X > 1 + 1e-7)):
            raise ValueError("Normalize inputs to [0, 1] using the BO bounds first.")

        x = X.reshape(-1, self.d).clamp(0.0, 1.0)
        n = x.shape[0]
        k = self.num_bins

        # Positive segment heights h_1,...,h_K, summing to 1.
        zeros = self.height_logits.new_zeros(self.d, 1)
        h = torch.softmax(torch.cat((self.height_logits, zeros), dim=-1), dim=-1)
        y_knots = torch.cat((zeros, h.cumsum(dim=-1)), dim=-1)

        # Positive derivatives delta_0,...,delta_K; boundary slopes = 1.
        ones = self.log_internal_derivatives.new_ones(self.d, 1)
        delta = torch.cat((ones, self.log_internal_derivatives.exp(), ones), dim=-1)

        # Find the segment. At x=1 choose the final segment with t=1.
        segment = (x * k).floor().long().clamp(max=k - 1)
        t = x * k - segment

        def select(values: Tensor, index: Tensor) -> Tensor:
            return (
                values.unsqueeze(0)
                .expand(n, -1, -1)
                .gather(dim=-1, index=index.unsqueeze(-1))
                .squeeze(-1)
            )

        height = select(h, segment)
        y_left = select(y_knots, segment)
        delta_left = select(delta, segment)
        delta_right = select(delta, segment + 1)
        secant = k * height  # h / (1/K)
        curved = t * (1.0 - t)

        numerator = secant * t.square() + delta_left * curved
        denominator = secant + (delta_left + delta_right - 2 * secant) * curved
        warped = y_left + height * numerator / denominator
        return warped.reshape_as(X)
