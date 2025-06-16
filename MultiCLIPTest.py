import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple, Union
import clip
from peft import LoraConfig, get_peft_model

class ParallelMultiheadAttention(nn.Module):
    def __init__(self, original_mha, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            copy.deepcopy(original_mha) for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, *args, **kwargs):
        # 获取输入
        x = args[0]
        batch_size = x.shape[0] // self.num_experts
        
        outputs = []
        weights = []
        
        # 按专家数分割输入
        expert_inputs = x.chunk(self.num_experts, dim=0)
        
        for idx, expert in enumerate(self.experts):
            # 每个专家处理对应的输入部分
            out, weight = expert(expert_inputs[idx], *args[1:], **kwargs)
            outputs.append(out)
            weights.append(weight)
        
        # 在batch维度上拼接输出
        out_concat = torch.cat(outputs, dim=0)
        weights_concat = torch.cat(weights, dim=0) if weights[0] is not None else None
        
        return out_concat, weights_concat

class MultiExpertCLIP(nn.Module):
    def __init__(self, original_model, num_experts=3):
        super().__init__()
        self.model = modify_model_attention(original_model, num_experts)
        self.num_experts = num_experts
        
    def forward(self, image, text):
        # 扩展输入图像以适应多个专家
        # [batch_size, C, H, W] -> [batch_size * num_experts, C, H, W]
        expanded_image = torch.cat([image for _ in range(self.num_experts)], dim=0)
        
        # 同样扩展文本输入
        expanded_text = torch.cat([text for _ in range(self.num_experts)], dim=0)
        
        # 前向传播
        image_features, text_features = self.model(expanded_image, expanded_text)
        
        # 重组特征以分离专家输出
        # [batch_size * num_experts, feature_dim] -> [num_experts, batch_size, feature_dim]
        batch_size = image.shape[0]
        image_features = image_features.reshape(self.num_experts, batch_size, -1)
        text_features = text_features.reshape(self.num_experts, batch_size, -1)
        
        return image_features, text_features

def modify_model_attention(model: nn.Module, num_experts: int = 3) -> nn.Module:
    modified_model = copy.deepcopy(model)
    
    for block in modified_model.base_model.model.visual.transformer.resblocks:
        original_mha = block.attn
        block.attn = ParallelMultiheadAttention(original_mha, num_experts)
    
    return modified_model

if __name__ == "__main__":
    batch_size = 3
    num_experts = 3
    device = "cuda"

    base_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Freeze text encoder
    for name, param in base_model.named_parameters():
        if "visual" not in name:
            param.requires_grad = False

    # Find attention layers
    target_modules = []
    for name, module in base_model.visual.transformer.resblocks.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            target_modules.append(f'visual.transformer.resblocks.{name}.out_proj')

    base_model = base_model.float()

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias='none'
    )
    base_model = get_peft_model(base_model, config)
    
    # 创建多专家模型
    model = MultiExpertCLIP(base_model, num_experts=num_experts)
    
    # 测试输入
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    label = ["A noisy image", "A red image", "A blue image"]
    tokenized_classes = clip.tokenize(label).to(device)
    
    try:
        with torch.no_grad():
            # 前向传播
            image_features, text_features = model(dummy_input, tokenized_classes)
            
            # 计算每个专家的输出与文本特征的相似度
            expert_logits = []
            for expert_idx in range(num_experts):
                expert_image_features = image_features[expert_idx]  # [batch_size, feature_dim]
                expert_text_features = text_features[expert_idx]    # [batch_size, feature_dim]
                expert_logit = expert_image_features @ expert_text_features.t()
                expert_logits.append(expert_logit)
            
            expert_logits = torch.stack(expert_logits)  # [num_experts, batch_size, num_classes]
            
        print("Forward pass successful!")
        print(f"Image features shape (per expert): {image_features.shape}")
        print(f"Text features shape (per expert): {text_features.shape}")
        print(f"Logits shape (per expert): {expert_logits.shape}")
        
        # 打印每个专家的预测结果
        for expert_idx in range(num_experts):
            print(f"\nExpert {expert_idx + 1} predictions:")
            expert_probs = expert_logits[expert_idx].softmax(dim=-1)
            for batch_idx in range(batch_size):
                pred_class = torch.argmax(expert_probs[batch_idx]).item()
                confidence = expert_probs[batch_idx][pred_class].item()
                print(f"Image {batch_idx + 1}: Predicted '{label[pred_class]}' with confidence {confidence:.3f}")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")