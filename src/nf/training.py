import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow
import kagglehub 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Automatically download the dataset using kagglehub
dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

# Define training arguments
class Args:
    path = dataset_path
    batch = 16
    epochs = 15
    n_flow = 32
    n_block = 4
    no_lu = False
    affine = False
    n_bits = 5
    lr = 1e-4
    img_size = 64
    temp = 0.7
    n_sample = 20

args = Args()

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []
    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes

def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p
    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def train(args, model, optimizer):
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    steps = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for image, _ in pbar:
            image = image.to(device)
            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if steps == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)
                steps += 1
                continue

            log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)
            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=loss.item(), logP=log_p.item(), logdet=log_det.item()
            )

            # # Save sample every 100 steps
            # if steps % 100 == 0:
            #     with torch.no_grad():
            #         utils.save_image(
            #             model_single.reverse(z_sample).cpu().data,
            #             f"sample/{str(steps).zfill(6)}.png",
            #             normalize=True,
            #             nrow=10,
            #             range=(-0.5, 0.5),
            #         )

            steps += 1

    print("✅ Training complete. Saving model...")
    torch.save(model.state_dict(), "glow_trained_model.pth")
    print("📦 Model saved as 'glow_trained_model.pth'")

def generate_samples(model, args):
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    z_sample = [torch.randn(args.n_sample, *z) * args.temp for z in z_shapes]
    z_sample = [z.to(device) for z in z_sample]

    with torch.no_grad():
        samples = model.reverse(z_sample).cpu().data
        utils.save_image(
            samples,
            "generated_samples.png",
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )
    print("🖼️ New samples saved as 'generated_samples.png'")

def load_and_display_image(image_path):
    """Load and display the generated image."""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def load_model_and_generate_image(model_path, args):
    """Load the saved model, generate samples, and display the image."""
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model_single.load_state_dict(torch.load(model_path))
    model_single = model_single.to(device).eval()

    generate_samples(model_single, args)
    load_and_display_image("generated_samples.png")

if __name__ == "__main__":
    print("🚀 Training with the following settings:")
    print(vars(args))

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)

    # # Load and display the generated image
    # load_and_display_image("generated_samples.png")
    # # Load the model and generate a new image
    # load_model_and_generate_image("glow_trained_model.pth", args)
    # # Load and display the generated image
    # load_and_display_image("generated_samples.png")

    image_path = generate_samples(model_single, args)
    load_and_display_image(image_path) 