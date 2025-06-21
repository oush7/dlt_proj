import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from train import DiffusionModel, DiffusionDataModule

def load_model(checkpoint_path, device):
    """Загружает модель из чекпоинта и определяет параметры датасета"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']
    
    model = DiffusionModel(
        in_channels=hparams['in_channels'],
        image_size=hparams['image_size'],
        model_channels=hparams['model_channels'],
        num_timesteps=hparams['num_timesteps'],
        beta_schedule=hparams['beta_schedule'],
        learning_rate=hparams['learning_rate'],
        guidance_scale=hparams['guidance_scale'],
        p_uncond=hparams['p_uncond'],
        num_classes=hparams['num_classes'],
    )
    
    # Загружаем веса модели
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, hparams

def get_dataset_classes(dataset_name):
    """Возвращает метки классов для разных датасетов"""
    if dataset_name == "MNIST":
        return [str(i) for i in range(10)]
    elif dataset_name == "FashionMNIST":
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    elif dataset_name == "CIFAR10":
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    elif dataset_name == "OMNIGLOT":
        return [str(i) for i in range(1623)]
    elif dataset_name == "LFW":
        return [str(i) for i in range(2)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def generate_class_samples(model, class_label, num_samples, device, guidance_scale=7.5):
    """Генерирует изображения для конкретного класса с CFG"""
    context = torch.full((num_samples,), class_label, device=device).long()
    
    with torch.no_grad():
        model.noise_scheduler.set_timesteps(model.hparams.num_timesteps)
        
        image = torch.randn(
            (num_samples, model.hparams.in_channels, 
             model.hparams.image_size, model.hparams.image_size),
            device=device
        )
        
        for t in model.noise_scheduler.timesteps:
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)

            if model.hparams.guidance_scale > 0:
                noise_uncond = model(image, timesteps, labels=None)
                noise_cond = model(image, timesteps, labels=context)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(image, timesteps, labels=context)

            image = model.noise_scheduler.step(noise_pred, t, image).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu()

def save_samples(images, output_dir, class_label, class_name, in_channels):
    """Сохраняет изображения и создает превью"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, img_tensor in enumerate(images):
        img_np = img_tensor.permute(1, 2, 0).numpy()
        if in_channels == 1:  # Grayscale
            img_np = img_np.squeeze(-1)
            img = Image.fromarray((img_np * 255).astype(np.uint8), 'L')
        else:  # RGB
            img = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
        
        img_path = os.path.join(output_dir, f"sample_{i+1}.png")
        img.save(img_path)
        saved_paths.append(img_path)
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    fig.suptitle(f'Class {class_label}: {class_name}', fontsize=16)
    
    for ax, img_tensor in zip(axes, images):
        img_np = img_tensor.permute(1, 2, 0).numpy()
        if in_channels == 1:
            ax.imshow(img_np.squeeze(), cmap='gray')
        else:
            ax.imshow(img_np)
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"class_{class_label}_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    
    return saved_paths, grid_path

def main():
    parser = argparse.ArgumentParser(description='Generate class-conditioned samples from DDPM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Samples per class')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    
    model, hparams = load_model(args.checkpoint, device)
    
    dataset_name = hparams.get('dataset', 'FashionMNIST')
    class_names = get_dataset_classes(dataset_name)
    num_classes = hparams['num_classes']
    in_channels = hparams['in_channels']
    
    print(f"Generating samples for {dataset_name} (classes: {num_classes})")
    print(f"Using guidance scale: {args.guidance_scale}")
    
    output_root = Path(args.output_dir) / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for class_label in range(num_classes):
        class_name = class_names[class_label]
        safe_class_name = class_name.replace('/', '_').replace(' ', '_')
        class_dir = output_root / f"class_{class_label}_{safe_class_name}"
        
        print(f"\nGenerating {args.num_samples} samples for class {class_label}: {class_name}")
        
        # Генерация изображений
        samples = generate_class_samples(
            model=model,
            class_label=class_label,
            num_samples=args.num_samples,
            device=device,
            guidance_scale=args.guidance_scale
        )
        
        # Сохранение и визуализация
        sample_paths, grid_path = save_samples(
            samples, class_dir, class_label, class_name, in_channels
        )
        
        results[class_label] = {
            'class_name': class_name,
            'samples': sample_paths,
            'grid': grid_path
        }
    
    metadata_path = output_root / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Generation completed! Results saved to: {output_root}")
    print(f"Metadata saved to: {metadata_path}")
    
    for class_label, data in results.items():
        print(f"\nClass {class_label}: {data['class_name']}")
        img = plt.imread(data['grid'])
        plt.figure(figsize=(10, 2))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Class {class_label}: {data['class_name']}")
        plt.show()

if __name__ == '__main__':
    main()