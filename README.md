# FFTNetViT-UNet for Semantic Segmentation

FFTNetViT-UNet is a semantic image segmentation model that combines a frequency domain-based Vision Transformer with a UNet decoder. This model effectively captures features in the frequency domain using Fourier Transform (FFT) and generates high-resolution segmentation outputs through the UNet structure.

## Model Architecture

![Model Architecture](https://mermaid.ink/img/pako:eNqNkk9v2zAMxb-K4FMHbEma_1hOyQY0GNYMaNceih0KWqITLbLlSXKxIsh3HyUlWRds2E4U-fP7kRRF3ShpFbOsT63-1iC2LT7_evE0S3FBB6c1QosCDxJ5RzkNLgK7uI1Tyo1u9S2xH7uO9qnTOhjIXHq0MZLT3QnVuQC94GWEk3Y5uPZSR-6IXRwdSk88cY-iXm4bTjGH0aOvg6PYgfDOJt-TjH9mFR8CPzG7oj2_mAOhWFy7vCHXnCjCXdkaNXSoZ1Lk8LdKfsFv29vvpxGGPxwnw3ZMsS_hQJGbxuUfMzuaZxXs0qnXYTqW1TDNd6c8lEcTaRdlR3y79yTMG9gvX0nRbHw0nv4--vk8k38Jz3gVP3hUWMHDCsR1uy0ceCQ-GBuSw2yeFbf61FZnpgQaPMqyh4FLlrlG1_R-fxmC8bQeWOGllV3dX66XT-uSZY0IQZrwR9ywznTwRDzLmm5w2lqPdxqfmDU9QWudTEVbM1t9_gEPDcrQ?type=png)

### Key Components

1. **FFTNetViT Encoder**:
   - Splits images into patches and embeds them
   - Applies class tokens and positional embeddings
   - Utilizes frequency domain-based multi-head attention
   - Enhances features through adaptive frequency filtering

2. **UNet Decoder**:
   - Expands feature maps through progressive upsampling
   - Generates high-resolution segmentation masks
   - Preserves information through residual connections

3. **Hybrid Loss Function**:
   - Combines Binary Cross Entropy (BCE) loss and Dice coefficient loss
   - Mitigates class imbalance issues

## Key Features

- **Frequency Domain Processing**: Leverages Fourier transforms to utilize both spatial and frequency domain information
- **Adaptive Frequency Filtering**: Emphasizes important frequency bands with learnable parameters
- **High-Resolution Output**: Preserves detailed boundaries through the UNet architecture
- **Efficient Training**: Optimizes performance through learning rate schedulers and data augmentation

## Dataset

The model was trained on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which includes:
- Images of 37 breeds of cats and dogs
- Approximately 200 images per breed
- Pixel-level segmentation annotations

## Implementation and Usage

### Required Packages

```
torch
torchvision
numpy
matplotlib
pillow
tqdm
einops
```

### Training Execution

```bash
python train_FFTNetViT_Unet_seg.py
```

When training, the dataset is automatically downloaded, and the following directory structure is created:
```
├── data/              # Dataset storage directory
│   └── oxford-iiit-pet/
├── models/            # Trained model storage directory
│   ├── fftnet_unet_pet_best.pth
│   └── fftnet_unet_pet_final.pth
└── results/           # Visualization results and evaluation metrics storage
    ├── segmentation_predictions.png
    └── fftnet_unet_pet_training_curves.png
```

### Inference GUI Execution

```bash
python inference_gui.py
```

You can perform segmentation on new images through the GUI interface.

## Performance Evaluation

The model shows the following performance on the Oxford-IIIT Pet dataset:

- **IoU (Intersection over Union)**: ~0.85
- **Dice Coefficient**: ~0.91

### Inference Result Example

The following is an example of the segmentation results from the FFTNetViT-UNet model:

![Inference Result Example](ScrShot%206.png)

## References

- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

## License

MIT License

## Contact

- Developer: Gromit Park
- GitHub: [@bemoregt](https://github.com/bemoregt)
