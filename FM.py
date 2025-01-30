import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchdiffeq import odeint
from typing import List
from tqdm import tqdm
import os
import argparse

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


config = {
    "epochs" : 50,
    "display_val_freq" : 1,
    "learning_rate" : 0.1,
    "num_samples" : 1,
    "inference_steps" : 10,
    "display_train_freq" : 20,
    "batch_size" : 4096,
    "num_workers": 4,
    'model_path': './model.pt',
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((32, 32)),
])


class FlowMatchingNet(nn.Module):
    def __init__(self, in_dim, h_dim=512, num_hidden=8, time_dim=1):
        super().__init__()

        # keep in mind that batch dimensions are implicit, and will heavily impact training time
        self.in_dim = in_dim
        # self.time_dim = time_dim
        self.time_dim = in_dim
        self.h_dim = h_dim
        self.num_hidden = num_hidden
        self.vgg_out_dim = 1000

        self.time_embedding = nn.Sequential(nn.Linear(1, self.time_dim), nn.SiLU())

        self.fc1 = nn.Sequential(nn.Linear(self.in_dim + self.time_dim, self.h_dim), nn.SiLU())
        self.main = nn.Sequential(*[nn.Linear(self.h_dim, self.h_dim), nn.SiLU()] * self.num_hidden)
        self.fc2 = nn.Sequential(nn.Linear(self.h_dim, self.in_dim), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(self.vgg_out_dim, self.in_dim), nn.SiLU())

        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        x, t = inputs
        size = x.size()

        x = self.vgg19(x)
        x = self.fc3(x)

        t = self.time_embedding(t)

        x = x.view(-1, self.in_dim)

        t = t.view(-1, self.in_dim)
        if t.dim() <= 1:
            t = t.unsqueeze(-1)

        # x = self.fc1(x + t)
        x = self.fc1(torch.cat((x, t), dim=-1))
        x = self.main(x)
        x = self.fc2(x)
        return x.reshape(*size)


def display_results(losses: dict):
    _, ax = plt.subplots(1, len(losses))

    for i, type in enumerate(losses):
        # ax[i].set_ylim(0, 20)
        ax[i].plot(range(len(losses[type])), losses[type])
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(f'{type} Loss')

    # plt.axis('off')
    plt.show()


def train(model, train_loader, val_loader):
    epochs = config["epochs"]
    lr = config["learning_rate"]
    display_train_freq = config["display_train_freq"]
    display_val_freq = config["display_val_freq"]
    epoch = 0

    # prior x_{0} ~ N(0, 0.05I)
    # sigma_min = 5e-2
    sigma_min = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=45_000)

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
            t_expanded = t[:, None, None].expand(x_1.shape)

            # x_0 ~ p_0(x_0)
            x_0 = torch.randn_like(x_1, device=device)

            x_t = (1. - (1. - sigma_min) * t_expanded) * x_0 + t_expanded * x_1
            target = x_1 - (1. - sigma_min) * x_0

            # predict the ODE on the transformed sample
            optimizer.zero_grad()

            loss = torch.pow(model([x_t, t]) - target, 2).mean()
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
                    t_expanded = t[:, None, None].expand(x_1.shape)

                    x_t = (1 - (1 - sigma_min) * t_expanded) * x_0 + t_expanded * x_1
                    target = x_1 - (1 - sigma_min) * x_0

                    val_loss = torch.pow(model([x_t, t]) - target, 2).mean()
                    total_val_loss += val_loss.detach().cpu().item()

            avg_val_loss = total_val_loss / len(val_loader)
            agg_val_loss.append(avg_val_loss)

        print(f'\nBatch: {i + 1} \t Training loss: {avg_loss:.3e}')
        print(f'Validation loss: {avg_val_loss:.3e}')

        # scheduler.step()

    display_results({'Training' : agg_train_loss, 'Validation' : agg_val_loss})


def sample(model, size):
    n_steps = config["inference_steps"]

    x_0 = torch.rand(size=(1, 3, 32, 32))
    time_traj = torch.linspace(0, 1, n_steps)

    with torch.no_grad():
        # Euler's method: assume that each time t applies to the whole batch
        # delta_t = 1 / n_steps
        # for t in tqdm(time_traj, position=0, leave=True):
        #     x_t += model([x_t, t.expand(len(x_t), 1)]) * delta_t

        def ode_func(t, x):
            if t.dim() <= 1:
                t = t.unsqueeze(-1)
            return model([x, t])

        sol = odeint(func=ode_func, y0=x_0, t=time_traj, rtol=1e-5, atol=1e-5)
        sol = (sol - sol.min()) / (sol.max() - sol.min())

        # breakpoint()

        _, axs = plt.subplots(1, n_steps)
        for i in range(n_steps):
            axs[i].imshow(sol[i, 0, :, :, :].permute(1, 2, 0).numpy())
            axs[i].set_aspect('equal')
            axs[i].axis('off')
            axs[i].set_title(f't: {time_traj[i]:.2f}')

        plt.tight_layout()
        plt.show()

        # x_t = torch.tanh(x_t)
        # x_t = x_t / 2. + 0.5

        # fig = plt.figure(figsize=(8, 8))
        # for i in range(1, rows * cols + 1):
        #     fig.add_subplot(rows, cols, i)
        #     plt.title(f"img #: {i}")
        #     plt.axis('off')
        #     plt.imshow(x_t[i - 1].permute(1, 2, 0).squeeze().detach().cpu().numpy())
        # plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', help='Training', default='y')

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    model_save = config['model_path']

    args = parser.parse_args()

    if args.training == 'y' and os.path.isfile(model_save):
        os.remove(model_save)

    # rows = 3
    # cols = 3
    # figure = plt.figure(figsize=(8, 8))
    # for i in range(1, rows * cols + 1):
    #      sample_idx = T.randint(len(train_set), size=(1,)).item()
    #      img, label = train_set[sample_idx]

    #      figure.add_subplot(rows, cols, i)
    #      plt.title(label)
    #      plt.axis('off')
    #      plt.imshow(img.squeeze().permute(1, 2, 0))
    # plt.show()

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # def plot_img(img):
    #      img = 0.5 * img + 0.5
    #      img = img.cpu().numpy()
    #      plt.imshow(np.transpose(img, (1, 2, 0)))
    #      plt.show()

    # dataiter = iter(train_loader)
    # train_img, labels = next(dataiter)

    # plot_img(torchvision.utils.make_grid(train_img))

    IMG_SHAPE = (batch_size, 3, 32, 32)

    model = FlowMatchingNet(in_dim=int(np.prod(IMG_SHAPE[1:], axis=0))).to(device)
    if not os.path.isfile(model_save):
        from torchvision.datasets import CIFAR10
        from torch.utils.data import DataLoader

        dataset = CIFAR10(
            root="./data",
            transform=transform,
            download=False
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
        test_set = CIFAR10(
            root="./data",
            train=False,
            transform=transform,
            download=False
        )

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

        train(model, train_loader, test_loader)
        torch.save(model.state_dict(), model_save)
    else:
        model.load_state_dict(torch.load(model_save, weights_only=True))
        model.eval()


    sample(model.cpu(), IMG_SHAPE)