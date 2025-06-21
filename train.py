import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler
import wandb
from PIL import Image
import numpy as np


class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size, image_size, augment=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.in_channels = 1 if dataset_name in ["MNIST", "OMNIGLOT", "FashionMNIST"] else 3

        self.train_transform = self._get_transforms(augment=True)
        self.val_transform = self._get_transforms(augment=False)

    def _get_transforms(self, augment=False):
        base_transforms = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ]
        if augment and self.augment:
            base_transforms = [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ] + base_transforms
        base_transforms += [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if self.in_channels == 1 
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transforms.Compose(base_transforms)

    def setup(self, stage=None):
        if self.dataset_name == "MNIST":
            self.train_dataset = datasets.MNIST(
                root="data", train=True, download=True, transform=self.train_transform
            )
            self.val_dataset = datasets.MNIST(
                root="data", train=False, download=True, transform=self.val_transform
            )
        elif self.dataset_name == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(
                root="data", train=True, download=True, transform=self.train_transform
            )
            self.val_dataset = datasets.CIFAR10(
                root="data", train=False, download=True, transform=self.val_transform
            )
        elif self.dataset_name == "OMNIGLOT":
            self.train_dataset = datasets.Omniglot(
                root="data", background=True, download=True, transform=self.train_transform
            )
            self.val_dataset = datasets.Omniglot(
                root="data", background=False, download=True, transform=self.val_transform
            )
        elif self.dataset_name == "LFW":
            self.train_dataset = datasets.LFWPeople(
                root="data", split="train", download=True, transform=self.train_transform
            )
            self.val_dataset = datasets.LFWPeople(
                root="data", split="test", download=True, transform=self.val_transform
            )
        elif self.dataset_name == "FashionMNIST":
            self.train_dataset = datasets.FashionMNIST(
                root="data", train=True, download=True, transform=self.train_transform
            )
            self.val_dataset = datasets.FashionMNIST(
                root="data", train=False, download=True, transform=self.val_transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        image_size,
        model_channels,
        num_timesteps,
        beta_schedule,
        learning_rate,
        guidance_scale=0.0,
        p_uncond=0.1,
        num_classes=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Для CFG: специальный "null" класс с индексом num_classes
        if guidance_scale > 0:
            self.embedding_dim = 4
            # embedding размером num_classes+1, последний индекс = null token
            self.class_emb = nn.Embedding(num_classes + 1, self.embedding_dim)

        self.initial_in_channels = in_channels
        unet_in_channels = in_channels + (self.embedding_dim if guidance_scale > 0 else 0)
        self.unet = UNet2DModel(
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=(
                model_channels, 
                model_channels * 2, 
                model_channels * 4, 
                model_channels * 4
            ),
            down_block_types=(
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D", 
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"
            ),
            dropout=0.1,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps, 
            beta_schedule=beta_schedule
        )

    def forward(self, x, timesteps, labels=None):
        # Подготовка условного ввода
        if self.hparams.guidance_scale > 0:
            # Если нет меток, используем null token
            if labels is None:
                labels = torch.full(
                    (x.size(0),), self.hparams.num_classes, device=x.device, dtype=torch.long
                )
            print(labels)
            emb = self.class_emb(labels)
            cond = emb.view(x.size(0), self.embedding_dim, 1, 1)
            cond = cond.expand(-1, -1, x.size(2), x.size(3))
            net_input = torch.cat([x, cond], dim=1)
        else:
            net_input = x
        return self.unet(net_input, timesteps).sample

    def training_step(self, batch, batch_idx):
        images, labels = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # Маскируем часть меток на null token
        if self.hparams.guidance_scale > 0:
            uncond_mask = torch.rand(images.size(0), device=self.device) < self.hparams.p_uncond
            labels_uncond = labels.clone()
            labels_uncond[uncond_mask] = self.hparams.num_classes
            context_labels = labels_uncond
        else:
            context_labels = None

        noise_pred = self.forward(noisy_images, timesteps, labels=context_labels)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        
        # Настоящий CFG на валидации
        if self.hparams.guidance_scale > 0:
            # Unconditional prediction
            noise_uncond = self.forward(noisy_images, timesteps, labels=None)
            # Conditional prediction
            noise_cond = self.forward(noisy_images, timesteps, labels=labels)
            noise_pred = noise_uncond + self.hparams.guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.forward(noisy_images, timesteps, labels=labels)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('val_loss', loss, prog_bar=True)

        if batch_idx == 0:
            self.generate_samples(labels[:9])

    def generate_samples(self, labels=None, num_samples=9):
        # Генерация при помощи полного цикла денойзинга
        with torch.no_grad():
            self.noise_scheduler.set_timesteps(1000)
            img = torch.randn(
                (num_samples, self.initial_in_channels, self.hparams.image_size, self.hparams.image_size),
                device=self.device
            )
            for t in self.noise_scheduler.timesteps:
                ts = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                if self.hparams.guidance_scale > 0:
                    uncond = self.forward(img, ts, labels=None)
                    cond = self.forward(img, ts, labels=labels)
                    pred = uncond + self.hparams.guidance_scale * (cond - uncond)
                else:
                    pred = self.forward(img, ts, labels=labels)
                img = self.noise_scheduler.step(pred, t, img).prev_sample

            img = (img / 2 + 0.5).clamp(0, 1)
            img = img.cpu().permute(0, 2, 3, 1).numpy()

            log_images = []
            for im in img:
                if self.initial_in_channels == 1:
                    im = im.squeeze(-1)
                    log_images.append(wandb.Image(im * 255, mode='L'))
                else:
                    log_images.append(wandb.Image(im * 255, mode='RGB'))
            self.logger.experiment.log({'samples': log_images})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','OMNIGLOT','LFW','FashionMNIST','CIFAR10'])
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--augment', action='store_true')

    parser.add_argument('--model_channels', type=int, default=96)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2', choices=['linear','squaredcos_cap_v2'])
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--p_uncond', type=float, default=0.2)

    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    dataset_to_num_classes = {'MNIST':10,'FashionMNIST':10,'CIFAR10':10,'OMNIGLOT':1623,'LFW':2}
    num_classes = dataset_to_num_classes.get(args.dataset)
    wandb_logger = WandbLogger(project='diffusion-experiments-improved')

    data_module = DiffusionDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=args.augment
    )
    data_module.setup()

    model = DiffusionModel(
        in_channels=data_module.in_channels,
        image_size=args.image_size,
        model_channels=args.model_channels,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        learning_rate=args.learning_rate,
        guidance_scale=args.guidance_scale,
        p_uncond=args.p_uncond,
        num_classes=num_classes,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='diffusion-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
