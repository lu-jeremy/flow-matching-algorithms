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
            h_dim: int = 64,
            enable_time_embed: bool = True,
            num_hidden: int = 2,
            out_time_dim: int = 1,
    ):
        
        super().__init__()

        # keep in mind that batch dimensions are implicit, and will heavily impact training time
        self.in_dim = in_dim
        self.out_time_dim = out_time_dim
        self.h_dim = h_dim
        self.num_hidden = num_hidden
        self.vgg_out_dim = 1000
        self.num_channels = num_channels
        self.enable_time_embed = enable_time_embed

        self.input_blocks = nn.Sequential(
            nn.Linear(self.in_dim + self.out_time_dim, self.h_dim),
            nn.SiLU(),
        )

        self.hidden_blocks = nn.ModuleList([
            nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_hidden)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim, self.in_dim),
            # nn.Tanh(),
        )

        self.linear_probe = nn.Sequential(
            nn.Linear(self.vgg_out_dim, self.in_dim), 
            nn.SiLU(),
        )

        # freeze vgg19 backbone
        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in self.vgg19.parameters():
            param.requires_grad = False

        if self.enable_time_embed:
            self.time_embedding = nn.Sequential(
                nn.Linear(1, self.out_time_dim),
                nn.SiLU(),
            )

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        
        x, t = inputs
        size = x.size()

        if self.num_channels == 3:
            x = self.vgg19(x)
            x = self.linear_probe(x)
        
        x = x.view(-1, self.in_dim)

        if self.time_embedding:
            t = self.time_embedding(t)
            
            # in the case of 1 item batch
            if len(t.size()) <= 1:
                t = t.unsqueeze(0)

        # ODE solver only allows 1D time trajectory inputs
        t = t.expand(len(x), -1)
    
        if t.dim() <= 1:
            t = t.unsqueeze(-1)
        
        # concatenate works better than add
        x = self.input_blocks(torch.cat((x, t), dim=-1))

        for module in self.hidden_blocks:
            x = module(x)
            x = F.silu(x)

        x = self.fc1(x)
        return x.reshape(*size)


class Path:
    def __init__(self, sigma, path_type):
        self.sigma = sigma
        self.path_type = path_type

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
        if self.path_type == 'CFM':
            x_t = (1. - (1. - self.sigma) * t) * torch.randn_like(x_1) + t * x_1
            target = (x_1 - (1. - self.sigma) * x_t) / (1. - (1. - self.sigma) * t)
        elif self.path_type == 'iCFM':
            # self.sigma * torch.randn_like(x_1) is not necessary here for moons
            x_t = (1. - t) * x_0 + t * x_1
            target = x_1 - x_0

        return x_t, target