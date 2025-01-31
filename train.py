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

from models import MLP, load_config, display_results, Path
from models import UNetModel


def train(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int, 
        lr: float,
        display_train_freq: int,
        display_val_freq: int,
        model_path,
        path,
        scheduler_type,
        progress_bar,
):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    agg_train_loss = []
    agg_val_loss = []

    best_loss = float('inf')

    for epoch in range(epochs):
        total_val_loss = 0
        total_loss = 0

        if progress_bar:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

        for i, batch in enumerate(train_loader):
            # sample training data ~ p_1(x_1)
            x_1, _ = batch
            x_1 = x_1.to(device)

            # t ~ U[0, 1]
            t = torch.rand(size=(len(x_1), 1), device=device)
            
            # x_{0} ~ p_{0}(x_{0}; 0, I)
            x_0 = torch.randn_like(x_1, device=device)

            # x_{t} ~ p_{t}(x_{t}; tx_{1}, (1 - (1 - \sigma_{\min})*t)I)
            x_t, u_t = path.sample_path(x_0=x_0, x_1=x_1, t=t)
            
            # predict the ODE on the transformed sample
            optimizer.zero_grad()

            loss = torch.pow(model([x_t, t]) - u_t, 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()

            # if i % display_train_freq == display_train_freq - 1:
        avg_loss = total_loss / len(train_loader)
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

                    x_t, u_t = path.sample_path(x_0=x_0, x_1=x_1, t=t)

                    val_loss = torch.pow(model([x_t, t]) - u_t, 2).mean()
                    total_val_loss += val_loss.detach().cpu().item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            agg_val_loss.append(avg_val_loss)

        print(f'\nBatch: {i + 1} \t Training loss: {avg_loss:.3e}')
        print(f'Validation loss: {avg_val_loss:.3e}')

        if scheduler_type is not None:
            scheduler.step()

    display_results({'Training' : agg_train_loss, 'Validation' : agg_val_loss})
    
    # checkpointing based on loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.splitext(model_path)[0] + f'_epoch{epoch}.pt')


def sample(
        model: torch.nn.Module, 
        n_steps: int, 
        num_channels: int, 
        size: Union[List, Tuple, torch.Size, int], 
        ode_method: str,
        num_samples: int = 1,
        display_plot: str = 'img',
    ):

    def ode_func(t, x):
        if t.dim() <= 1:
            t = t.unsqueeze(-1)
        return model([x, t])

    _, axs = plt.subplots(num_samples, n_steps, figsize=(30, 4), sharex=True, sharey=True)
    axs[0].set_xlim(-3.0, 3.0)
    axs[0].set_ylim(-3.0, 3.0)
    
    with torch.no_grad():
        for s in range(num_samples):
            # size: B x C x H x W
            x_0 = torch.randn(size=(1, num_channels, *size))
            time_traj = torch.linspace(0, 1, n_steps)

            sol = odeint(func=ode_func, y0=x_0, t=time_traj, atol=1e-4, rtol=1e-4, method=ode_method)

            if display_plot == 'img':
                sol = (sol - sol.min()) / (sol.max() - sol.min())

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
            elif display_plot == '2d':
                for i in range(n_steps):
                    c_ax = axs[i]
                    c_ax.scatter(sol[i, :, :, :, 0].numpy(), sol[i, :, :, :, 1].numpy(), s=10)
                    c_ax.set_aspect('equal')
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

    # retrieve config params based on dataset
    dataset_type = config["dataset_type"]
    data_config = config['data_config'][dataset_type]
    training = config['training']

    # setup config parameters
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    model_path = data_config['model_path']
    img_reshape_size = data_config['img_reshape_size']
    train_val_split = data_config['train_val_split']
    model_type = data_config['model']
    epochs = data_config["epochs"]
    lr = data_config["learning_rate"]
    display_train_freq = data_config["display_train_freq"]
    display_val_freq = data_config["display_val_freq"]
    num_samples = data_config['num_samples']
    n_steps = data_config['inference_steps']
    sigma_min = data_config['sigma_min']
    ode_method = data_config['ode_method']
    scheduler_type = data_config['scheduler_type']
    path_type = data_config['path_type']
    progress_bar = data_config['progress_bar']

    # normalize on image channels depending on dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        *([transforms.Resize(img_reshape_size)] if img_reshape_size is not None else []),
        transforms.RandomHorizontalFlip(),
    ])

    if dataset_type == 'cifar10':
        transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # only import when necessary
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(root="./data", transform=transform, download=False)
        train_set, val_set = torch.utils.data.random_split(dataset, train_val_split)
        test_set = CIFAR10(root="./data", train=False, transform=transform, download=False)

        # do not include batch size
        rgb_channels = len(dataset[0][0])
        in_dim = np.prod(img_reshape_size) * rgb_channels
        size = img_reshape_size
        display_plot = 'img'

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

        # do not include batch size
        rgb_channels = len(dataset[0][0])
        in_dim = np.prod(img_reshape_size) * rgb_channels
        size = img_reshape_size
        display_plot = 'img'

    elif dataset_type == 'moons':
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split

        num_points = 100_000
        num_sample_points = 128
        dataset = make_moons(num_points, noise=0.05)
        x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        class MoonDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.long)
            
            def __len__(self):
                return min(len(self.x), len(self.y))
        
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        train_set = MoonDataset(x_train, y_train)
        val_set = MoonDataset(x_val, y_val)
        test_set = MoonDataset(x_test, y_test)
        
        # training
        in_dim = 2
        # sampling
        size = (num_sample_points, in_dim)
        rgb_channels = 1
        display_plot = '2d'

    if model_type == 'mlp':
        model = MLP(
            in_dim=in_dim,
            num_channels=rgb_channels,
        ).to(device)
    elif model_type == 'unet':
        # TODO: setup model config parameters
        if dataset_type == 'mnist':
            model = UNetModel(
                dim=(rgb_channels, *img_reshape_size), 
                num_channels=32, 
                num_res_blocks=1,
                # num_classes=10,
                class_cond=False,
            ).to(device)
        elif dataset_type == 'cifar10':
            model = UNetModel(
                dim=(rgb_channels, *img_reshape_size),
                num_res_blocks=2,
                num_channels=32,
                channel_mult=[1, 2, 2, 2],
                num_heads=4,
                num_head_channels=64,
                attention_resolutions="16",
                dropout=0.1,
                class_cond=False,
            ).to(device)

    if training:
        # reset model checkpoint
        if os.path.isfile(model_path):
            os.remove(model_path)

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

        # x_{t} ~ p_{t}(x_{t} | x_{1})
        path = Path(sigma=sigma_min, path_type=path_type)
        train(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            display_train_freq=display_train_freq,
            display_val_freq=display_val_freq,
            model_path=model_path,
            path=path,
            scheduler_type=scheduler_type,
            progress_bar=progress_bar,
        )

        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

    sample(
        model=model.cpu(),
        n_steps=n_steps,
        num_channels=rgb_channels,
        size=size,
        ode_method=ode_method,
        num_samples=num_samples,
        display_plot=display_plot,
    )

    # score = fid.compute_fid(gen=sample, dataset_name='cifar10', dataset_res=img_reshape_size[0], batch_size=1024, model_name="clip_vit_b_32")
    # print(f"FID: {score}")
