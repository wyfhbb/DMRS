import torch
import clip
import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

# Import your model classes here
from CLIPwithLoraModel import MultiExpertCLIP, get_model_with_lora

def calculate_model_stats(model, device, num_experts=3, is_mme=True):
    """
    Calculate and return parameter statistics for the model.
    
    Args:
        model: The model to analyze
        device: Device the model is on
        num_experts: Number of experts in multi-expert models
        is_mme: Whether this is a MultiExpertCLIP model
        
    Returns:
        A dictionary with parameter statistics
    """
    stats = {}
    
    # Total parameters
    stats['total_params'] = sum(p.numel() for p in model.parameters())
    
    # Trainable parameters
    stats['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Activation ratio
    stats['activation_ratio'] = stats['trainable_params'] / stats['total_params'] * 100
    
    # Count parameters by module type
    if is_mme:
        # For MultiExpertCLIP
        stats['text_encoder_params'] = sum(p.numel() for name, p in model.model.base_model.model.transformer.named_parameters())
        stats['vision_encoder_params'] = sum(p.numel() for name, p in model.model.base_model.model.visual.named_parameters())
        
        # LoRA parameters
        stats['lora_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora" in name.lower())
        stats['lora_A_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora_A" in name)
        stats['lora_B_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora_B" in name)
        
        # Expert-specific parameters
        expert_params = []
        for i in range(num_experts):
            expert_count = sum(p.numel() for name, p in model.named_parameters() 
                              if f"experts.{i}" in name)
            expert_params.append(expert_count)
        
        stats['expert_params'] = expert_params
        stats['avg_expert_params'] = np.mean(expert_params) if expert_params else 0
    else:
        # For standard CLIP+LoRA
        stats['text_encoder_params'] = sum(p.numel() for name, p in model.named_parameters() 
                                       if "transformer" in name and not "visual" in name)
        stats['vision_encoder_params'] = sum(p.numel() for name, p in model.named_parameters() 
                                        if "visual" in name)
        
        # LoRA parameters
        stats['lora_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora" in name.lower())
        stats['lora_A_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora_A" in name)
        stats['lora_B_params'] = sum(p.numel() for name, p in model.named_parameters() if "lora_B" in name)
    
    # Estimate FLOPs for CLIP ViT-B/32 (based on typical values)
    # These are rough estimates based on model size and operations
    batch_size = 1
    image_size = 224
    text_length = 77  # CLIP's default text length
    
    # Vision Transformer FLOPs (based on ViT-B/32 architecture)
    # This is a rough estimate, adjusting for your specific model may be needed
    patch_size = 32
    num_patches = (image_size // patch_size) ** 2 + 1  # +1 for class token
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    mlp_ratio = 4
    
    # Patch embedding + position embedding
    patch_embed_flops = batch_size * (image_size ** 2) * 3 * embed_dim
    
    # Transformer layers
    # Self-attention: 4 * N^2 * d (where N is sequence length, d is embed dim)
    # MLP: 2 * N * d * (4*d) (assuming MLP ratio of 4)
    attn_flops = 4 * num_patches ** 2 * embed_dim
    mlp_flops = 2 * num_patches * embed_dim * (embed_dim * mlp_ratio)
    layer_flops = attn_flops + mlp_flops
    transformer_flops = num_layers * layer_flops
    
    # Text encoder FLOPs (similar calculation)
    text_embedding_flops = batch_size * text_length * embed_dim
    text_transformer_flops = num_layers * (4 * text_length ** 2 * embed_dim + 2 * text_length * embed_dim * (embed_dim * mlp_ratio))
    
    # Total FLOPs estimate
    vision_flops = patch_embed_flops + transformer_flops
    text_flops = text_embedding_flops + text_transformer_flops
    
    # For multi-expert model, multiply by number of experts
    if is_mme:
        vision_flops *= num_experts
    
    stats['estimated_inference_flops'] = vision_flops + text_flops
    
    # For training, we add backward pass FLOPs
    # Only trainable parameters contribute to backward pass
    # Backward pass is roughly 2x the forward pass for trainable params
    trainable_ratio = stats['trainable_params'] / stats['total_params']
    backward_flops = stats['estimated_inference_flops'] * 2 * trainable_ratio
    stats['estimated_training_flops'] = stats['estimated_inference_flops'] + backward_flops
    
    return stats

def add_to_tensorboard(stats, log_dir="./logs"):
    """Add model statistics to TensorBoard"""
    # Create a unique run ID with timestamp
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"model_stats_{run_id}"))
    
    # Add parameters and FLOPS to TensorBoard
    writer.add_scalar('Model/TotalParams', stats['total_params'], 0)
    writer.add_scalar('Model/TrainableParams', stats['trainable_params'], 0)
    writer.add_scalar('Model/ActivationRatio', stats['activation_ratio'], 0)
    writer.add_scalar('Model/TextEncoderParams', stats['text_encoder_params'], 0)
    writer.add_scalar('Model/VisionEncoderParams', stats['vision_encoder_params'], 0)
    writer.add_scalar('Model/LoRAParams', stats.get('lora_params', 0), 0)
    
    if 'estimated_inference_flops' in stats:
        writer.add_scalar('Model/InferenceFLOPs', stats['estimated_inference_flops'], 0)
    if 'estimated_training_flops' in stats:
        writer.add_scalar('Model/TrainingFLOPs', stats['estimated_training_flops'], 0)
    
    # If expert data is available
    if 'expert_params' in stats and stats['expert_params']:
        for i, exp_params in enumerate(stats['expert_params']):
            writer.add_scalar(f'Model/Expert{i+1}Params', exp_params, 0)
    
    writer.close()
    return os.path.join(log_dir, f"model_stats_{run_id}")

def format_stats(stats):
    """Format statistics for printing and logging"""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL STATISTICS")
    lines.append("=" * 80)
    
    lines.append("\nPARAMETER COUNTS:")
    lines.append(f"Total parameters: {stats['total_params']:,}")
    lines.append(f"Trainable parameters: {stats['trainable_params']:,} " +
                f"({stats['activation_ratio']:.2f}% of total)")
    
    if 'lora_params' in stats:
        lines.append(f"LoRA parameters: {stats['lora_params']:,}")
    
    if 'lora_A_params' in stats:
        lines.append(f"LoRA A parameters: {stats['lora_A_params']:,}")
        lines.append(f"LoRA B parameters: {stats['lora_B_params']:,}")
    
    lines.append(f"Text encoder parameters: {stats['text_encoder_params']:,}")
    lines.append(f"Vision encoder parameters: {stats['vision_encoder_params']:,}")
    
    if 'expert_params' in stats and stats['expert_params']:
        lines.append("\nEXPERT PARAMETERS:")
        for i, exp_params in enumerate(stats['expert_params']):
            lines.append(f"Expert {i+1} parameters: {exp_params:,}")
        lines.append(f"Average parameters per expert: {int(stats['avg_expert_params']):,}")
    
    if 'estimated_inference_flops' in stats:
        lines.append("\nESTIMATED COMPUTATIONAL COST (FLOPS):")
        lines.append(f"Inference FLOPs: {stats['estimated_inference_flops']:,}")
        lines.append(f"Training FLOPs (per batch):")
        lines.append(f"  - Forward pass: {stats['estimated_inference_flops']:,}")
        if 'estimated_training_flops' in stats:
            backward_flops = stats['estimated_training_flops'] - stats['estimated_inference_flops']
            lines.append(f"  - Backward pass (est.): {backward_flops:,}")
            lines.append(f"  - Total training: {stats['estimated_training_flops']:,}")
    
    return lines

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate model statistics for CLIP+LoRA models")
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--num_experts', type=int, default=7, help='Number of experts for MultiExpertCLIP')
    parser.add_argument('--mme', action='store_true', default=True, help='Use MultiExpertCLIP model')
    parser.add_argument('--lora_r', type=int, default=12, help='Rank of LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=24, help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout probability for LoRA')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for calculation')
    parser.add_argument('--output_file', type=str, default='model_stats.txt', help='File to save the statistics')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save TensorBoard logs')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = args.device
    
    print(f"Calculating model statistics on {device}...")
    
    # Create model architecture object for model creation
    class ModelArgs:
        def __init__(self, lora_r, lora_alpha, lora_dropout, device):
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.device = device
    
    model_args = ModelArgs(args.lora_r, args.lora_alpha, args.lora_dropout, args.device)
    
    # Create model
    try:
        if args.mme:
            print("Creating MultiExpertCLIP model...")
            model = MultiExpertCLIP(num_experts=args.num_experts, device=device)
        else:
            print("Creating standard CLIP+LoRA model...")
            model = get_model_with_lora(model_args, device)
        
        model = model.to(device)
        
        # Load pre-trained weights if specified
        if args.model_path and os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Falling back to basic CLIP model for estimation...")
        model, _ = clip.load("ViT-B/32", device=device)
        args.mme = False
    
    # Calculate statistics
    stats = calculate_model_stats(model, device, args.num_experts, args.mme)
    
    # Format and print statistics
    output_lines = format_stats(stats)
    for line in output_lines:
        print(line)
    
    # Save to file
    with open(args.output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    print(f"\nStatistics saved to {args.output_file}")
    
    # Add to TensorBoard
    tb_log_dir = add_to_tensorboard(stats, args.log_dir)
    print(f"TensorBoard logs saved to {tb_log_dir}")
    print(f"View with: tensorboard --logdir={args.log_dir}")

if __name__ == "__main__":
    main()