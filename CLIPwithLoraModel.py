import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple, Union
import clip
from peft import LoraConfig, get_peft_model
import numpy as np
import os

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
        
        outputs = []
        weights = []
        
        batch_size = x.shape[1]
        
        blocks = batch_size // self.num_experts
        
        for idx, expert in enumerate(self.experts):
            # 每个专家处理对应的输入部分
            out, weight = expert(x, *args[1:], **kwargs)
            outputs.append(out[:,idx*blocks:(idx+1)*blocks])
            if weight is not None:
                weights.append(weight[:,idx*blocks:(idx+1)*blocks])
        
        # 在batch维度上拼接输出
        out_concat = torch.cat(outputs, dim=1)
        weights_concat = torch.cat(weights, dim=1) if len(weights) > 0 else None
        
        return out_concat, weights_concat

class MultiExpertCLIP(nn.Module):
    def __init__(self, num_experts=3,device="cuda"):
        super().__init__()
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
            r=12,
            lora_alpha=24,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias='none'
        )
        base_model = get_peft_model(base_model, config)
        
        for name, param in base_model.named_parameters():
            if "lora" in name or any(target in name for target in target_modules):
                param.requires_grad = True
        
        for block in base_model.base_model.model.visual.transformer.resblocks:
            original_mha = block.attn
            block.attn = ParallelMultiheadAttention(original_mha, num_experts)
        
        self.model = base_model
        self.num_experts = num_experts
        
        
    def forward(self, image, text):
        # 扩展输入图像以适应多个专家
        # [batch_size, C, H, W] -> [batch_size * num_experts, C, H, W]
        expanded_image = torch.cat([image for _ in range(self.num_experts)], dim=0)
        
        # 前向传播
        image_logits, text_logits = self.model(expanded_image, text)
        
        image_logits = image_logits.reshape(self.num_experts, -1, image_logits.shape[-1])
        # 使用.t()转置后两个维度
        text_logits = image_logits.permute(0, 2, 1)
        
        return image_logits, text_logits
    
    def feature(self, image, text):
        # 扩展输入图像以适应多个专家
        # [batch_size, C, H, W] -> [batch_size * num_experts, C, H, W]
        expanded_image = torch.cat([image for _ in range(self.num_experts)], dim=0)
        
        # 前向传播
        image_features = self.model.encode_image(expanded_image)
        text_features = self.model.encode_text(text)
        
        image_features = image_features.reshape(self.num_experts, -1, image_features.shape[-1])
        # 使用.t()转置后两个维度
        text_features = image_features.permute(0, 2, 1)
        
        return image_features, text_features


def get_model_with_lora(args, device):
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Freeze text encoder since we only want to fine-tune visual attention layers
    for name, param in model.named_parameters():
        if "visual" not in name:
            param.requires_grad = False

    # Dynamically find all attention layers in the visual transformer
    target_modules = []
    for name, module in model.visual.transformer.resblocks.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            target_modules.append(f'visual.transformer.resblocks.{name}.out_proj')

    # 修改 1: 确保模型参数使用 FP32
    model = model.float()
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias='none'
    )
    model = get_peft_model(model, config)

    # Ensure LoRA layers are trainable
    for name, param in model.named_parameters():
        if "lora" in name or any(target in name for target in target_modules):
            param.requires_grad = True
    return model

if __name__ == "__main__":
    import clip
    from Loss import MMELoss
    
    # 基本设置
    batch_size = 4
    num_experts = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型
    model = MultiExpertCLIP(num_experts=num_experts, device=device)
    model = model.to(device)
    
    # 创建损失函数
    # 假设我们有3个类别的不平衡数据集
    cls_num_list = [100, 50, 10]
    criterion = MMELoss(cls_num_list=cls_num_list, num_experts=num_experts)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 创建测试数据
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    texts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
    text_tokens = clip.tokenize(texts).to(device)
    targets = torch.randint(0, len(texts), (batch_size,)).to(device)
    
    print("初始化完成，开始测试训练...")
    print("\n模型结构:")
    print(model)
    
    # 训练循环
    for epoch in range(3):
        # 前向传播
        image_logits, _ = model(images, text_tokens)
        
        
        # 计算损失
        loss = criterion(experts_logits=image_logits, targets=targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算每个专家的准确率
        accuracies = []
        for i in range(num_experts):
            pred = torch.argmax(image_logits[i], dim=1)
            acc = (pred == targets).float().mean().item()
            accuracies.append(acc)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Loss: {loss.item():.4f}")
        
        # 打印每个专家的准确率
        for i, acc in enumerate(accuracies):
            print(f"Expert {i + 1} Accuracy: {acc:.4f}")
        
        # 打印参数梯度信息
        print("\n梯度信息:")
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:  # 只打印有效梯度
                    print(f"{name} grad norm: {grad_norm:.4f}")
        
        # 打印部分参数值，用于验证参数更新
        if epoch == 0 or epoch == 2:
            print("\n参数值示例:")
            for name, param in model.named_parameters():
                if "lora" in name:  # 只打印LoRA相关参数
                    print(f"{name} mean value: {param.mean().item():.4f}")
    
    print("\n测试完成!")
    
    # 保存一些测试结果
    test_results = {
        "final_loss": loss.item(),
        "expert_accuracies": accuracies,
        "model_state": model.state_dict()
    }
    
    # 可选：保存模型
    try:
        torch.save(test_results, "test_results.pth")
        print("测试结果已保存到 test_results.pth")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")