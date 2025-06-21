## Training

To train the diffusion model:

```bash
python train.py \
    --dataset MNIST \
    --image_size 32 \
    --batch_size 64 \
    --max_epochs 100 \
    --beta_schedule linear \
    --num_timesteps 1000
```

For conditional generation with classifier-free guidance:

```bash
python train.py \
    --dataset MNIST \
    --guidance_scale 3.0
```

## License

MIT 