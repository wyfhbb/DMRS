# DMRS: Long-tailed Remote Sensing Recognition via Semantic-aware Mixing and Diversity Experts

This repository contains the official implementation of our paper: **"DMRS: Long-tailed remote sensing recognition via semantic-aware mixing and diversity experts"**

📄 **Paper**: [https://doi.org/10.1016/j.jag.2025.104623](https://doi.org/10.1016/j.jag.2025.104623)

## Requirements

### Environment Setup

We recommend using Python 3.11. Install dependencies using uv (recommended) or pip:

```bash
# Using uv (recommended)
uv sync
```

### Main Dependencies

- PyTorch >= 2.7.1
- torchvision >= 0.22.1
- CLIP (OpenAI)
- PEFT >= 0.15.2
- scikit-learn >= 1.7.0
- matplotlib >= 3.10.3

## Dataset Structure

Organize your remote sensing dataset in the following structure(The program will process it as long-tail data):

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Supported datasets:
- NWPU-RESISC45
- RSD46-WHU
- Custom remote sensing datasets

## Usage

### Basic Training

```bash
python CLIP_Lora.py \
    --dataset_path ./NWPU-RESISC45 \
    --imb_type exp \
    --imb_factor 0.01 \
    --epochs 40 \
    --batch_size 16 \
    --lr 1e-1
```

### Multi-Expert Training with MME Loss

```bash
python CLIP_Lora.py \
    --dataset_path ./NWPU-RESISC45 \
    --imb_type exp \
    --imb_factor 0.01 \
    --MME_loss True \
    --num_experts 3 \
    --mixrs True \
    --epochs 40 \
    --batch_size 16 \
    --lr 1e-1
```

### Key Parameters

- `--dataset_path`: Path to your dataset
- `--imb_type`: Type of imbalance ('exp' for exponential, 'step' for step)
- `--imb_factor`: Imbalance factor (0.01 for severe imbalance)
- `--MME_loss`: Enable D-LoRA loss function
- `--num_experts`: Number of experts (default: 3)
- `--mixrs`: Enable MixSSS data augmentation
- `--lora_r`: LoRA rank (default: 12)
- `--lora_alpha`: LoRA scaling factor (default: 24)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{WANG2025104623,
    title = {DMRS: Long-tailed remote sensing recognition via semantic-aware mixing and diversity experts},
    journal = {International Journal of Applied Earth Observation and Geoinformation},
    volume = {141},
    pages = {104623},
    year = {2025},
    issn = {1569-8432},
    doi = {https://doi.org/10.1016/j.jag.2025.104623},
    url = {https://www.sciencedirect.com/science/article/pii/S1569843225002705},
    author = {Yifan Wang and Fan Zhang and Qihao Zhao and Wei Hu and Fei Ma},
    keywords = {Long-tail distribution, Remote sensing, Diversity experts, Data augmentation, Foundation models},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please:
- Open an issue on GitHub
- Contact: [2024200827@buct.edu.cn]
