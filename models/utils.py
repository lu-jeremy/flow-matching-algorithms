import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path: str):
    import json
    with open(config_path, 'r') as f:
        return json.load(f)
    

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


def verify_image(dataloader):
        def plot_img(img):
            img = 0.5 * img + 0.5
            img = img.cpu().numpy()
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.show()
    
        dataiter = iter(dataloader)
        train_imgs, _ = next(dataiter)
        import torchvision
        plot_img(torchvision.utils.make_grid(train_imgs[0]))


def verify_image2(dataset):
    rows = 3
    cols = 3
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, rows * cols + 1):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]

            figure.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis('off')
            plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.show()