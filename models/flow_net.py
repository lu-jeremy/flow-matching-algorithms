import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from typing import Tuple


class MLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_channels: int,
            sigma: float,
            h_dim: int = 512,
            num_hidden: int = 8,
            time_dim: int = 1,
    ):
        
        super().__init__()

        # keep in mind that batch dimensions are implicit, and will heavily impact training time
        self.in_dim = in_dim
        # self.time_dim = time_dim
        self.time_dim = in_dim
        self.h_dim = h_dim
        self.num_hidden = num_hidden
        self.vgg_out_dim = 1000
        self.num_channels = num_channels

        self.sigma = sigma

        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU()
        )

        self.input_blocks = nn.Sequential(
            nn.Linear(self.in_dim + self.time_dim, self.h_dim),
            nn.SiLU()
        )

        self.hidden_blocks = nn.ModuleList([
            nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_hidden)
        ])

        self.fc2 = nn.Sequential(
            nn.Linear(self.h_dim, self.in_dim), 
            nn.Tanh(),
        )

        self.linear_probe = nn.Sequential(
            nn.Linear(self.vgg_out_dim, self.in_dim), 
            nn.SiLU(),
        )

        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        
        x, t = inputs
        size = x.size()

        # vgg19 backbone
        if self.num_channels == 3:
            x = self.vgg19(x)
            x = self.linear_probe(x)

        x = x.view(-1, self.in_dim)

        t = self.time_embedding(t)
        t = t.view(-1, self.in_dim)

        if t.dim() <= 1:
            t = t.unsqueeze(-1)

        # concatenate works better
        # x = self.input_blocks(x + t)
        x = self.input_blocks(torch.cat((x, t), dim=-1))

        for module in self.hidden_blocks:
            x = module(x)
            x = F.silu(x)

        x = self.fc2(x)
        return x.reshape(*size)

    def sample_path(
            self,
            x_0: torch.Tensor,
            x_1: torch.Tensor,
            t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # dependent on user input
        if len(t.size()) != len(x_1.size()):
            t = t[:, None, None].expand(x_1.shape)

        # \mu_{t} = tx_{1}, \sigma_{t} = 1 - (1 - \sigma_{\min})t
        x_t = (1. - (1. - self.sigma) * t) * x_0 + t * x_1
        target = (x_1 - (1. - self.sigma) * x_t) / (1. - (1. - self.sigma) * t)

        return x_t, target
