import numpy as np
import torch
from torchvision import transforms
from torchdiffeq import odeint
from typing import List, Tuple, Union
from tqdm import tqdm
import os
import argparse
from cleanfid import fid

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models import MLP, load_config, display_results
from models import UNetModel


def train(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int, 
        lr: float,
        display_train_freq: int,
        display_val_freq: int
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    agg_train_loss = []
    agg_val_loss = []

    for epoch in range(epochs):
        total_val_loss = 0
        total_loss = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)):
            # sample training data ~ p_1(x_1)
            x_1, _ = batch
            x_1 = x_1.to(device)

            # t ~ U[0, 1]
            t = torch.rand(size=(len(x_1), 1), device=device)
            
            # x_{0} ~ p_{0}(x_{0}; 0, I)
            x_0 = torch.randn_like(x_1, device=device)

            # x_{t} ~ p_{t}(x_{t}; tx_{1}, (1 - (1 - \sigma_{\min})*t)I)
            x_t, u_t = model.sample_path(x_0=x_0, x_1=x_1, t=t)
            
            # predict the ODE on the transformed sample
            optimizer.zero_grad()

            loss = torch.pow(model([x_t, t]) - u_t, 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()

            # if i % display_train_freq == display_train_freq - 1:
        avg_loss = total_loss / display_train_freq
        agg_train_loss.append(avg_loss)

        if epoch % display_val_freq == 0:
            with torch.no_grad():
                for val_batch in val_loader:
                    x_1, _ = val_batch
                    x_1 = x_1.to(device)

                    if len(x_1) != batch_size:
                        continue

                    x_0 = torch.randn_like(x_1).to(device)
                    t = torch.rand(size=(len(x_1), 1), device=device)

                    x_t, u_t = model.sample_path(x_0=x_0, x_1=x_1, t=t)

                    val_loss = torch.pow(model([x_t, t]) - u_t, 2).mean()
                    total_val_loss += val_loss.detach().cpu().item()

            avg_val_loss = total_val_loss / len(val_loader)
            agg_val_loss.append(avg_val_loss)

        print(f'\nBatch: {i + 1} \t Training loss: {avg_loss:.3e}')
        print(f'Validation loss: {avg_val_loss:.3e}')

        scheduler.step()

    display_results({'Training' : agg_train_loss, 'Validation' : agg_val_loss})


def sample(
        model: torch.nn.Module, 
        n_steps: int, 
        num_channels: int, 
        size: Union[List, Tuple, torch.Size], 
        ode_method: str,
        num_samples: int = 1,
        display_plot: bool = False,
    ):

    def ode_func(t, x):
        if t.dim() <= 1:
            t = t.unsqueeze(-1)
        return model([x, t])

    _, axs = plt.subplots(num_samples, n_steps)
    
    with torch.no_grad():
        for s in range(num_samples):
            # size: B x C x H x W
            x_0 = torch.rand(size=(1, num_channels, *size))
            time_traj = torch.linspace(0, 1, n_steps)

            # Euler's method: assume that each time t applies to the whole batch
            sol = odeint(func=ode_func, y0=x_0, t=time_traj, rtol=1e-4, atol=1e-4, method=ode_method)             
            sol = (sol - sol.min()) / (sol.max() - sol.min())

            if display_plot:
                for i in range(n_steps):
                    # size: n_steps x B x C x H x W, always take n_samples
                    if num_samples != 1:
                        c_ax = axs[s, i]
                    else:
                        c_ax = axs[i]
                        
                    c_ax.imshow(sol[i, 0, :, :, :].permute(1, 2, 0).numpy())
                    c_ax.set_aspect('equal')
                    c_ax.axis('off')
                    c_ax.set_title(f't: {time_traj[i]:.2f}')
                
                plt.tight_layout()
                plt.show()

    return sol[-1, 0, :, :, :] * 255.


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to config file', default='./config/config.json')
    args = parser.parse_args()

    config = load_config(args.config)

    # setup config parameters
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    model_path = config['model_path']
    img_reshape_size = config['img_reshape_size']
    train_val_split = config['train_val_split']
    model_type = config['model']
    dataset_type = config['dataset_type']
    epochs = config["epochs"]
    lr = config["learning_rate"]
    display_train_freq = config["display_train_freq"]
    display_val_freq = config["display_val_freq"]
    training = config['training']
    num_samples = config['num_samples']
    n_steps = config['inference_steps']
    sigma_min = config['sigma_min']
    ode_method = config['ode_method']

    # o.w. prevent overriding previous model file
    if training and os.path.isfile(model_path):
        os.remove(model_path)

    # normalize on image channels depending on dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_reshape_size),
        transforms.RandomHorizontalFlip(),
    ])

    if dataset_type == 'cifar10':
        transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # only import when necessary
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(root="./data", transform=transform, download=False)
        train_set, val_set = torch.utils.data.random_split(dataset, train_val_split)
        test_set = CIFAR10(root="./data", train=False, transform=transform, download=False)

        num_channels = len(dataset[0][0])

    elif dataset_type == 'mnist':
        transform.transforms.append(transforms.Normalize(mean=0.5, std=0.5))
        
        from torchvision.datasets import MNIST

        MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

        dataset = MNIST(root='./data', transform=transform, download=True)
        train_set, val_set = torch.utils.data.random_split(dataset, train_val_split)
        test_set = MNIST(root='./data', transform=transform, train=False, download=True)

        num_channels = len(dataset[0][0])

    # do not include batch size
    in_dim = np.prod(img_reshape_size) * num_channels

    if model_type == 'mlp':
        model = MLP(
            in_dim=in_dim,
            num_channels=num_channels,
            sigma=sigma_min
        ).to(device)
    elif model_type == 'unet':
        if dataset_type == 'mnist':
            model = UNetModel(
                dim=(num_channels, *img_reshape_size), 
                num_channels=32, 
                num_res_blocks=1,
                # num_classes=10,
                class_cond=False,
                sigma=sigma_min,
            ).to(device)
        elif dataset_type == 'cifar10':
            model = UNetModel(
                dim=(num_channels, *img_reshape_size),
                num_res_blocks=2,
                num_channels=128,
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions="16",
                dropout=0.1,
                sigma=sigma_min,
            ).to(device)
        

    if not os.path.isfile(model_path):
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        train(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            display_train_freq=display_train_freq,
            display_val_freq=display_val_freq,
        )

        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

    sample(
        model=model.cpu(),
        n_steps=n_steps,
        num_channels=num_channels,
        size=img_reshape_size,
        ode_method=ode_method,
        num_samples=num_samples,
        display_plot=True,
    )

    fid.compute_fid(gen=sample, dataset_name='cifar10', dataset_res=img_reshape_size[0], batch_size=1024)

