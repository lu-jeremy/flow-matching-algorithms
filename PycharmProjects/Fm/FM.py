import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.datasets import ImageNet, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

config = {
    "epochs" : 100,
    "val_epochs" : 3,
    "learning_rate" : 0.1,
    "test_steps": 10,
    "num_samples" : 1,
    "inference_steps" : 1000,
    "print_loss" : 100,
    "batch_size" : 64,
    "num_workers" : 4,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((32, 32)),
])


class FM(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims = np.prod(np.array(IMG_SHAPE), axis=0)

        self.fc1 = nn.Sequential(
            nn.Linear(2 * input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dims),
        )

    def forward(self, inputs):
        x, t = inputs
        x = torch.cat((x, t), dim=0).flatten()
        x = self.fc1(x)
        x = torch.reshape(x, IMG_SHAPE)

        return x
    
    
def train(model, train_loader, val_loader):
    epochs = config["epochs"]
    lr = config["learning_rate"]
    test_steps = config["test_steps"]
    print_loss = config["print_loss"]
    val_epochs = config["val_epochs"]
    epoch = 0

    # prior x_{0} ~ N(0, 0.05I)
    sigma_min = 5e-2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=45_000)

    for epoch in range(epochs):
        total_val_loss = 0
        total_loss = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)):
            # sample training data ~ p_1(x_1)
            x_1, _ = batch
            x_1 = x_1.to(device)

            # t ~ U[0, 1]
            t = torch.rand(size=(len(x_1),), device=device)[:, None, None, None].expand(x_1.shape)

            # x_0 ~ p_0(x_0)
            x_0 = torch.randn_like(x_1, device=device)

            psi = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
            target = x_1 - (1 - sigma_min) * x_0

            # predict the ODE on the transformed sample
            optimizer.zero_grad()
            loss = torch.pow(model([psi, t]) - target, 2).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

            if i % print_loss == print_loss - 1:
                print(f'\nBatch: {i + 1} \t Training_loss: {total_loss / print_loss:.2f}')
                total_loss = 0

        if epoch % val_epochs == 0:
            for val_batch in val_loader:
                with torch.no_grad():
                    x_1, _ = val_batch
                    x_1 = x_1.to(device)

                    if len(x_1) != batch_size:
                        continue

                    x_0 = torch.randn_like(x_1).to(device)
                    t = torch.rand(size=(len(x_1),), device=device)[:, None, None, None].expand(x_1.shape)

                    psi = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
                    target = x_1 - (1 - sigma_min) * x_0

                    val_loss = torch.pow(model([psi, t]) - target, 2).mean()
                    total_val_loss += val_loss

            print(f'val loss: {total_val_loss / len(val_loader):.2f}')

        # scheduler.step()



def inference(model, tesT_loader):
    num_examples = config["num_samples"]
    n_steps = config["inference_steps"]

    rows = 8
    cols = 8

    for test_batch in test_loader:
        x_1, _ = test_batch
        x_1 = x_1.to(device)
        x_t = torch.randn_like(x_1, device=device)
        
        time_traj = torch.linspace(0, 1, n_steps, device=device)
        with torch.no_grad():
            # Euclidean solver: assume that each time t applies to the whole batch
            prev = model([x_t, time_traj[0][None, None, None, None].expand(x_1.shape)])
            for t in tqdm(range(len(time_traj) - 1), position=0, leave=True):
                next = model([x_t, time_traj[t + 1][None, None, None, None].expand(x_1.shape)])
                ode_solve = (1 / n_steps) * (prev + next) / 2
                x_t += ode_solve
                prev = next

        fig = plt.figure(figsize=(8, 8))
        for i in range(1, rows * cols + 1):
            fig.add_subplot(rows, cols, i)
            plt.title(f"img #: {i}")
            plt.axis('off')
            plt.imshow(x_t[i - 1].permute(1, 2, 0).detach().cpu().numpy().squeeze())
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config['batch_size']
    num_workers = config['num_workers']

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

    IMG_SHAPE = next(iter(train_loader))[0].shape

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

    model_save = './model.pt'

    model = FM().to(device)
    if not os.path.isfile(model_save):
        train(model, train_loader, test_loader)
        torch.save(model.state_dict(), model_save)
    else:
        model.load_state_dict(torch.load(model_save, weights_only=True))
        model.eval()
    inference(model, test_loader)