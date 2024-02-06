from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .th_operator import calc_grad


class PinnBaseline(nn.Module):
    
    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.ffn_layers = []
        input_dim = 3
        for hidden_dim in hidden_dims:
            self.ffn_layers.append(nn.Linear(input_dim, hidden_dim))
            self.ffn_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.ffn_layers.append(nn.Linear(input_dim, 2))
        self.ffn = nn.Sequential(*self.ffn_layers)

        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor = None,
        u: Tensor = None,
        v: Tensor = None,
    ):
        """
        inputs: x, y, t
        labels: p, u, v
        """
        inputs = torch.stack([x, y, t], dim=1)
        hidden_output = self.ffn(inputs)
        psi = hidden_output[:, 0]
        p_pred = hidden_output[:, 1]
        u_pred = calc_grad(psi, y)
        v_pred = -calc_grad(psi, x)

        preds = torch.stack([p_pred, u_pred, v_pred], dim=1)
        u_t = calc_grad(u_pred, t)
        u_x = calc_grad(u_pred, x)
        u_y = calc_grad(u_pred, y)
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        v_t = calc_grad(v_pred, t)
        v_x = calc_grad(v_pred, x)
        v_y = calc_grad(v_pred, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        p_x = calc_grad(p_pred, x)
        p_y = calc_grad(p_pred, y)

        f_u = (
            u_t
            + self.lambda1 * (u_pred * u_x + v_pred * u_y)
            + p_x
            - self.lambda2 * (u_xx + u_yy)
        )
        f_v = (
            v_t
            + self.lambda1 * (u_pred * v_x + v_pred * v_y)
            + p_y
            - self.lambda2 * (v_xx + v_yy)
        )
        loss, losses = self.loss_fn(u, v, u_pred, v_pred, f_u, f_v)
        return {
            "preds": preds,
            "loss": loss,
            "losses": losses,
        }

    def loss_fn(self, u, v, u_pred, v_pred, f_u_pred, f_v_pred):
        """
        u: (b, 1)
        v: (b, 1)
        p: (b, 1)
        """
        
        u_loss    = F.mse_loss(u, u_pred, reduction="sum")
        v_loss    = F.mse_loss(v, v_pred, reduction="sum")
        f_u_loss  = F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred), reduction="sum")
        f_v_loss  = F.mse_loss(f_v_pred, torch.zeros_like(f_v_pred), reduction="sum")
        loss = u_loss + v_loss + f_u_loss + f_v_loss
        return loss, {
            "u_loss": u_loss,
            "v_loss": v_loss,
            "f_u_loss": f_u_loss,
            "f_v_loss": f_v_loss,
        }