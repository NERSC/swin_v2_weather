import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

#loss function with rel/abs Lp loss; from FNO github
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, relative=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

        self.relative = relative

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        if self.relative:
            return self.rel(x, y)
        else:
            return self.abs(x, y)

# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """Geometric Lp loss"""

    def __init__(
        self,
        img_size,
        device,
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = False,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
        jacobian: Optional[str] = "s2",
        quadrature_rule: Optional[str] = "naive",
    ):  # pragma: no cover
        super(GeometricLpLoss, self).__init__()

        self.p = p
        self.img_size = img_size
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        if jacobian == "s2":
            jacobian = torch.sin(
                torch.linspace(0, torch.pi, self.img_size[0])
            ).unsqueeze(1).to(device)
        else:
            jacobian = torch.ones(self.img_size[0], 1).to(device)

        if quadrature_rule == "naive":
            dtheta = torch.pi / self.img_size[0]
            dlambda = 2 * torch.pi / self.img_size[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        self.register_buffer("quad_weight", quad_weight)

    def abs(
        self, prd: torch.Tensor, tar: torch.Tensor
    ):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]
        if self.pole_mask:
            all_norms = torch.sum(
                torch.abs(
                    prd[..., self.pole_mask : -self.pole_mask, :]
                    - tar[..., self.pole_mask : -self.pole_mask, :]
                )
                ** self.p
                * self.quad_weight[..., self.pole_mask : -self.pole_mask, :],
                dim=(-2, -1),
            )
        else:
            all_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.quad_weight,
                dim=(-2, -1),
            )

        all_norms = all_norms.reshape(num_examples, -1).sum()

        if not self.squared:
            all_norms = all_norms ** (1 / self.p)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(
        self,
        prd: torch.Tensor,
        tar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pragma: no cover
        """Computes the relative loss"""
        num_examples = prd.size()[0]

        if self.pole_mask:
            diff_norms = torch.sum(
                torch.abs(
                    prd[..., self.pole_mask : -self.pole_mask, :]
                    - tar[..., self.pole_mask : -self.pole_mask, :]
                )
                ** self.p
                * self.quad_weight[..., self.pole_mask : -self.pole_mask, :],
                dim=(-2, -1),
            )
        else:
            diff_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.quad_weight, dim=(-2, -1)
            )

        diff_norms = diff_norms.reshape(num_examples, -1)

        tar_norms = torch.sum(torch.abs(tar) ** self.p * self.quad_weight, dim=(-2, -1))
        tar_norms = tar_norms.reshape(num_examples, -1)

        if not self.squared:
            diff_norms = diff_norms ** (1 / self.p)
            tar_norms = tar_norms ** (1 / self.p)

        # setup return value
        retval =  (diff_norms / tar_norms)
        if mask is not None:
            retval = retval * mask

        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(
        self,
        prd: torch.Tensor,
        tar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss

