import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMELoss(nn.Module):
    def __init__(self, cls_num_list=None, num_experts=3,device='cuda'):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = len(cls_num_list)
        
        # 计算类别权重
        cls_num_list = np.array(cls_num_list)
        # 使用 log space 来提高数值稳定性
        self.register_buffer('log_prior', torch.FloatTensor(np.log(cls_num_list / np.sum(cls_num_list) + 1e-12)))
        
        self.log_prior = self.log_prior.to(device)
        
        # 为不同专家设置不同的λ值
        if num_experts == 3:
            self.lambda_values = [-0.5, 0, 1.5]  # 头部、中部、尾部专家
        else:
            self.lambda_values = np.linspace(-1, 2, num_experts).tolist()
        
    def forward(self, experts_logits, targets, test=False):
        """
        Args:
            expert_logits: List of expert outputs before softmax [num_experts, batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        loss = 0
        batch_size = experts_logits[0].size(0)
        
        if test:
            targets = F.one_hot(targets, self.num_classes)
            # long转为float
            targets = targets.float()
        
        # 计算每个专家的损失
        for i in range(self.num_experts):
            # 在log space计算重平衡权重
            log_weight = self.lambda_values[i] * self.log_prior
            
            
            # 对logits进行重平衡
            balanced_logits = experts_logits[i] + log_weight.unsqueeze(0)
            
            balanced_logits = F.log_softmax(balanced_logits, dim=1)
            
            # 使用交叉熵损失（包含了log_softmax的数值稳定实现）
            loss_i = F.cross_entropy(balanced_logits, targets)
            loss += loss_i

        return loss / self.num_experts

    def inference(self, experts_logits):
        """
        推理时的专家集成
        Args:
            expert_logits: List of expert outputs before softmax [num_experts, batch_size, num_classes]
        """
        probs = []
        # 对每个专家的输出进行重平衡并转换为概率
        for i in range(self.num_experts):
            # log_weight = self.lambda_values[i] * self.log_prior
            # balanced_logits = experts_logits[i] + log_weight.unsqueeze(0)
            # probs.append(F.softmax(balanced_logits, dim=1))
            probs.append(F.softmax(experts_logits[i], dim=1))
            
        # 平均所有专家的预测概率
        final_prob = torch.stack(probs).mean(0)
        return final_prob
    def expert_inference(self, experts_logits):
        """
        推理时的专家集成
        Args:
            expert_logits: List of expert outputs before softmax [num_experts, batch_size, num_classes]
        """
        probs = []
        # 对每个专家的输出进行重平衡并转换为概率
        for i in range(self.num_experts):
            # log_weight = self.lambda_values[i] * self.log_prior
            # balanced_logits = experts_logits[i] + log_weight.unsqueeze(0)
            # probs.append(F.softmax(balanced_logits, dim=1))
            probs.append(F.softmax(experts_logits[i], dim=1))
            
        return probs

if __name__ == "__main__":
    # 设置基本参数
    batch_size = 4
    num_experts = 3
    num_classes = 5
    feature_dim = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟类别分布（不平衡数据集）
    cls_num_list = [100, 50, 30, 20, 10]  # 示例类别分布
    
    # 创建一个简单的多专家模型
    class SimpleMultiExpertModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes, num_experts):
            super().__init__()
            self.num_experts = num_experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_classes)
                ) for _ in range(num_experts)
            ])
        
        def forward(self, x):
            return torch.stack([expert(x) for expert in self.experts])
    
    # 初始化模型、损失函数和优化器
    model = SimpleMultiExpertModel(feature_dim, 20, num_classes, num_experts).to(device)
    criterion = MMELoss(cls_num_list=cls_num_list, num_experts=num_experts)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 创建随机测试数据
    inputs = torch.randn(batch_size, feature_dim).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print("初始参数:")
    for name, param in model.named_parameters():
        print(f"{name}: grad={param.requires_grad}, shape={param.shape}")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(3):
        # 前向传播
        experts_logits = model(inputs)  # shape: [num_experts, batch_size, num_classes]
        
        # 计算损失
        loss = criterion(experts_logits=experts_logits, targets=targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算每个专家的准确率
        accuracies = []
        for i in range(num_experts):
            pred = torch.argmax(experts_logits[i], dim=1)
            acc = (pred == targets).float().mean().item()
            accuracies.append(acc)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Loss: {loss.item():.4f}")
        for i, acc in enumerate(accuracies):
            print(f"Expert {i + 1} Accuracy: {acc:.4f}")
        
        # 打印梯度信息
        print("\n梯度信息:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item():.4f}")
    
    print("\n测试完成!")
    
    