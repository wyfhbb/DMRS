import torch
import clip
import torchvision.transforms as transforms
from Load_Imbalance_Data import IMBALANCE_DATA, STRONG_WEAK_DATASET, ImbalanceDataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from MixRS import MixRS
import numpy as np
from Loss import MMELoss
from CLIPwithLoraModel import MultiExpertCLIP, get_model_with_lora
import logging
import datetime
from torchvision.transforms import AutoAugment,AutoAugmentPolicy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate CLIP with LoRA")
    parser.add_argument('--dataset_path', type=str, default='./NWPU-RESISC45', help='Path to the dataset')
    parser.add_argument('--imb_type', type=str, default='exp', help='Type of imbalance')
    parser.add_argument('--imb_factor', type=float, default=0.01, help='Imbalance factor')
    parser.add_argument('--mixrs', type=bool, default=True, help='Use MixRS')
    parser.add_argument('--MME_loss', type=bool, default=True, help='Use Mix Multi-Expert Loss')
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test dataset ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval of epochs to evaluatse the model')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=12, help='Rank of LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=24, help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout probability for LoRA')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/evaluation')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load the pre-trained model')
    return parser.parse_args()

def get_dataset(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset_imb_split = IMBALANCE_DATA(
        root_path=args.dataset_path,
        imb_type=args.imb_type,
        imb_factor=args.imb_factor,
        split_test=True,
        imbalance=True,
        test_ratio=args.test_ratio
    )
    dataset_cls_num_list = dataset_imb_split.get_cls_num_list()

    train_dataset = dataset_imb_split.train_dataset
    train_targets = dataset_imb_split.train_targets
    test_dataset = dataset_imb_split.test_dataset
    test_targets = dataset_imb_split.test_targets

    train_dataset = STRONG_WEAK_DATASET(train_dataset, train_targets, transform, transform_strong=None, transform_weak=None)
    test_dataset = STRONG_WEAK_DATASET(test_dataset, test_targets, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]), transform_strong=None, transform_weak=None)

    # Pre-tokenize the class names to avoid repeated CPU-GPU transfers
    classes = dataset_imb_split.classes
    tokenized_classes = clip.tokenize(classes).to(args.device if torch.cuda.is_available() else "cpu")

    return train_dataset, train_targets, test_dataset, test_targets, classes, tokenized_classes, dataset_cls_num_list


def train_one_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, writer, tokenized_classes, classes, scaler=None, mixrs=None, args=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", ncols=100):
        images = images.to(device).float()
        targets = targets.to(device).long()
        
        if args is not None and hasattr(args, 'mixrs') and args.mixrs and mixrs is not None:
            train_images, train_labels, mask, sk_ratio = mixrs(images, targets, num_classes=len(classes))
            train_images = train_images.to(device).float()
            train_labels = train_labels.to(device).float()
        else:
            train_images = images.to(device).float()
            train_labels = targets.to(device).long()
        

        optimizer.zero_grad()

        # Use all class texts
        tokenized_texts = tokenized_classes

        if scaler:
            with torch.amp.autocast_mode.autocast(device_type=device, dtype=torch.float32): 
                logits_per_image, _ = model(train_images, tokenized_texts)
                if args.MME_loss:
                    loss = loss_fn(experts_logits=logits_per_image, targets=train_labels)
                else:
                    loss = loss_fn(logits_per_image, train_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            logits_per_image, _ = model(train_images, tokenized_texts)

            # Calculate loss
            if args.MME_loss:
                loss = loss_fn(experts_logits=logits_per_image, targets=train_labels)
            else:
                loss = loss_fn(logits_per_image, train_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Get prediction results
        if args.MME_loss:
            logits_per_image = logits_per_image.mean(dim=0)
            pred = torch.argmax(logits_per_image, dim=1)
        else:
            pred = torch.argmax(logits_per_image, dim=1)
        
        if args.mixrs:
            correct += (pred == torch.argmax(train_labels, dim=1)).sum().item()
        else:
            correct += (pred == targets).sum().item()
        total += train_labels.size(0)

    avg_loss = running_loss / len(train_dataloader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    logging.info(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    

    # Write to tensorboard
    writer.add_scalar('Loss/train', avg_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', accuracy, epoch + 1)

def evaluate_model(model, test_dataloader, loss_fn, tokenized_classes, device, epoch, writer, classes, args, cls_num_list=None):
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_targets = []
    # 按照高中低分别记录，高频为训练集中样本数量大于100的，低频为小于20的，中频为20-100的
    # Initialize counters for frequency-based statistics
    high_correct = 0
    mid_correct = 0
    low_correct = 0
    high_total = 0
    mid_total = 0
    low_total = 0

    # Define frequency thresholds
    high_thresh = 80
    low_thresh = 20
    
    # Create frequency category mapping for each class
    if cls_num_list is not None:
        class_freq_map = {}
        for i, num_samples in enumerate(cls_num_list):
            if num_samples >= high_thresh:
                class_freq_map[i] = 'high'
            elif num_samples <= low_thresh:
                class_freq_map[i] = 'low'
            else:
                class_freq_map[i] = 'mid'

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(test_dataloader, desc="Evaluating", ncols=100):
            images = images.to(device)
            targets = targets.to(device).long()

            # Use all class texts
            tokenized_texts = tokenized_classes

            # Forward pass
            logits_per_image, _ = model(images, tokenized_texts)

            # Calculate loss
            if args.MME_loss:
                loss = loss_fn(experts_logits=logits_per_image, targets=targets, test=True)
            else:
                loss = loss_fn(logits_per_image, targets)
            running_loss += loss.item()

            # Get prediction results
            if args.MME_loss:
                logits_per_image = loss_fn.inference(logits_per_image)
            pred = torch.argmax(logits_per_image, dim=1)

            correct += (pred == targets).sum().item()
            total += targets.size(0)
            # Update frequency-based metrics
            if cls_num_list is not None:
                for p, t in zip(pred, targets):
                    freq_category = class_freq_map[t.item()]
                    is_correct = (p == t).item()
                    
                    if freq_category == 'high':
                        high_correct += is_correct
                        high_total += 1
                    elif freq_category == 'mid':
                        mid_correct += is_correct
                        mid_total += 1
                    else:  # low frequency
                        low_correct += is_correct
                        low_total += 1

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate overall metrics
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_dataloader)
    
    # Calculate frequency-based accuracies
    if cls_num_list is not None:
        high_acc = 100 * high_correct / high_total if high_total > 0 else 0
        mid_acc = 100 * mid_correct / mid_total if mid_total > 0 else 0
        low_acc = 100 * low_correct / low_total if low_total > 0 else 0
        
        print(f'\nFrequency-based accuracies:')
        print(f'High-frequency (>{high_thresh} samples): {high_acc:.2f}%')
        print(f'Mid-frequency ({low_thresh}-{high_thresh} samples): {mid_acc:.2f}%')
        print(f'Low-frequency (<{low_thresh} samples): {low_acc:.2f}%')
        
        logging.info(f'\nFrequency-based accuracies:')
        logging.info(f'High-frequency (>{high_thresh} samples): {high_acc:.2f}%')
        logging.info(f'Mid-frequency ({low_thresh}-{high_thresh} samples): {mid_acc:.2f}%')
        logging.info(f'Low-frequency (<{low_thresh} samples): {low_acc:.2f}%')
        
        # Add frequency-based metrics to tensorboard
        writer.add_scalar('Accuracy/high_freq', high_acc, epoch + 1)
        writer.add_scalar('Accuracy/mid_freq', mid_acc, epoch + 1)
        writer.add_scalar('Accuracy/low_freq', low_acc, epoch + 1)

    # Calculate accuracy
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_dataloader) 
    print(f'Accuracy on test set after Epoch {epoch+1}: {accuracy:.2f}%, Loss: {avg_loss:.4f}')
    logging.info(f'Accuracy on test set after Epoch {epoch+1}: {accuracy:.2f}%, Loss: {avg_loss:.4f}')

    # Calculate per-class accuracy
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=classes))
    logging.info(classification_report(all_targets, all_preds, target_names=classes))
    # Write to tensorboard
    writer.add_scalar('Accuracy/test', accuracy, epoch + 1)
    writer.add_scalar('Loss/test', avg_loss, epoch + 1)

    return accuracy

def main():
    args = parse_arguments()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 设置日志保存路径 时间 lr MME_loss mixrs num_experts
    log_file_path = f'./{args.log_dir}/training_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{args.lr}_MMELoss{args.MME_loss}_MixRS{args.mixrs}'
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    log_path = os.path.join(log_file_path, 'training.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 保存当前参数
    logging.info(f"Arguments: {args}")
    
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_file_path, f"run_{run_id}")
    writer = SummaryWriter(log_dir=log_dir)

    # Get dataset and classes
    train_dataset, train_targets, test_dataset, test_targets, classes, tokenized_classes, dataset_cls_num_list = get_dataset(args)

    # BalancedSampler DataLoader
    if args.mixrs:
        train_dataloader = ImbalanceDataLoader(train_dataset, train_targets, len(classes), args.batch_size, shuffle=True, num_workers=4, balanced=True)
        test_dataloader = ImbalanceDataLoader(test_dataset, test_targets, len(classes), args.batch_size, shuffle=False, num_workers=4, balanced=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.MME_loss:
        # Get the model with LoRA applied
        model = MultiExpertCLIP(num_experts=args.num_experts,device=device)
        model = model.to(device)  # Ensure model is on the correct device
    else:
        # Get the model with LoRA applied
        model = get_model_with_lora(args, device)
        model = model.to(device)  # Ensure model is on the correct device
    
    # 记录模型所有形状
    logging.info(f"Model: {model}")
    
    # Load pre-trained model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from {args.load_model}")
        logging.info(f"Model loaded from {args.load_model}")
    
    # 计算并记录模型参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")
    # logging.info(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")
    
    # 计算text_encoder的参数量
    # if args.MME_loss:
    #     text_encoder_params = sum(p.numel() for name, p in model.model.base_model.model.transformer.named_parameters())
    # else:
    #     text_encoder_params = sum(p.numel() for name, p in model.named_parameters() if "transformer" in name and not "visual" in name)
    
    # 计算image_encoder参数量
    # if args.MME_loss:
    #     vision_encoder_params = sum(p.numel() for name, p in model.model.base_model.model.visual.named_parameters())
    # else:
    #     vision_encoder_params = sum(p.numel() for name, p in model.named_parameters() if "visual" in name)
        
    # print(f"图像编码器参数量: {vision_encoder_params:,}, 文本编码器参数量: {text_encoder_params:,}")
    
    
    # 计算模型FLOPs - 分别计算推理和训练时的计算量
    # def calculate_flops(model, input_shape=(3, 224, 224)):
    #     try:
    #         from ptflops import get_model_complexity_info
    #         import copy
            
    #         # 为了计算推理时的FLOPs，创建一个包装类
    #         class InferenceModelWrapper(nn.Module):
    #             def __init__(self, model, device):
    #                 super().__init__()
    #                 self.model = model
    #                 self.device = device
    #                 self.dummy_text = clip.tokenize(["dog"]).to(device)
                
    #             def forward(self, x):
    #                 # 设置为评估模式
    #                 self.model.eval()
    #                 return self.model(x, self.dummy_text)[0]
            
    #         # 为了计算训练时的FLOPs，创建一个包装类
    #         class TrainingModelWrapper(nn.Module):
    #             def __init__(self, model, device):
    #                 super().__init__()
    #                 self.model = model
    #                 self.device = device
    #                 self.dummy_text = clip.tokenize(["dog"]).to(device)
                
    #             def forward(self, x):
    #                 self.model.train()
    #                 # 保存计算图来计算反向传播
    #                 return self.model(x, self.dummy_text)[0]
            
    #         # 1. 计算推理时的FLOPs
    #         model_copy = copy.deepcopy(model)
    #         inference_model = InferenceModelWrapper(model_copy, device)
    #         inference_model.eval()
            
    #         inference_macs, _ = get_model_complexity_info(
    #             inference_model, input_shape, as_strings=False,
    #             print_per_layer_stat=False, verbose=False
    #         )
            
    #         # 2. 计算训练时的FLOPs
    #         # 对于LoRA模型，我们需要计算：
    #         # - 前向传播的FLOPs
    #         # - 反向传播中只有LoRA参数需要计算梯度
    #         model_copy = copy.deepcopy(model)
    #         training_model = TrainingModelWrapper(model_copy, device)
    #         training_model.train()
            
    #         # 首先计算前向传播的FLOPs
    #         training_forward_macs, _ = get_model_complexity_info(
    #             training_model, input_shape, as_strings=False,
    #             print_per_layer_stat=False, verbose=False
    #         )
            
    #         # 计算LoRA的反向传播计算量
    #         # 只有可训练的参数(主要是LoRA参数)会参与反向传播
    #         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #         total_params = sum(p.numel() for p in model.parameters())
    #         lora_ratio = trainable_params / total_params
            
    #         # 估计反向传播的FLOPs为前向传播的两倍乘以LoRA参数比例
    #         training_backward_macs = 2 * training_forward_macs * lora_ratio
            
    #         # 总训练FLOPs = 前向传播 + 反向传播
    #         training_total_macs = training_forward_macs + training_backward_macs
            
    #         return inference_macs, training_total_macs
    #     except ImportError:
    #         print("警告: ptflops库未安装，无法计算FLOPs。请使用 pip install ptflops 安装。")
    #         logging.warning("ptflops库未安装，无法计算FLOPs。")
    #         return None, None
    #     except Exception as e:
    #         print(f"计算FLOPs时出错: {e}")
    #         logging.error(f"计算FLOPs时出错: {e}")
    #         return None, None
    
    # # 尝试计算FLOPs
    # inference_flops, training_flops = calculate_flops(model)
    
    # # 记录参数量和计算量信息
    # print(f"模型总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,}")
    # print(f"文本编码器参数量: {text_encoder_params:,}")
    # print(f"视觉编码器参数量: {vision_encoder_params:,}")
    # if inference_flops is not None:
    #     print(f"推理时FLOPs: {inference_flops:,}")
    # if training_flops is not None:
    #     print(f"训练时FLOPs: {training_flops:,}")
    
    # logging.info(f"模型总参数量: {total_params:,}")
    # logging.info(f"可训练参数量: {trainable_params:,}")
    # logging.info(f"文本编码器参数量: {text_encoder_params:,}")
    # logging.info(f"视觉编码器参数量: {vision_encoder_params:,}")
    # if inference_flops is not None:
    #     logging.info(f"推理时FLOPs: {inference_flops:,}")
    # if training_flops is not None:
    #     logging.info(f"训练时FLOPs: {training_flops:,}")
        
    # # 将信息写入TensorBoard
    # writer.add_scalar('Model/TotalParams', total_params, 0)
    # writer.add_scalar('Model/TrainableParams', trainable_params, 0)
    # writer.add_scalar('Model/TextEncoderParams', text_encoder_params, 0)
    # writer.add_scalar('Model/VisionEncoderParams', vision_encoder_params, 0)
    # if inference_flops is not None:
    #     writer.add_scalar('Model/InferenceFLOPs', inference_flops, 0)
    # if training_flops is not None:
    #     writer.add_scalar('Model/TrainingFLOPs', training_flops, 0)
    
    if args.mixrs:
        with torch.no_grad():
            if args.MME_loss:
                text_features = model.model.encode_text(tokenized_classes)
            else:
                text_features = model.encode_text(tokenized_classes)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            similarity_matrix = (100.0 * text_features @ text_features.T).cpu().numpy()
        np.fill_diagonal(similarity_matrix, 0)
        print(similarity_matrix)
        mixrs = MixRS(alpha=4, mask_num=7, scale_=2, scale=32, filp_or_similarity='similarity', Word_Similarity=similarity_matrix,device=device)
    else:
        mixrs = None
    
    if args.MME_loss:
        loss_fn = MMELoss(cls_num_list=dataset_cls_num_list[0], num_experts=args.num_experts,device=device)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Define optimizer: Only parameters that require gradients
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Define learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, pct_start=0.3, anneal_strategy='linear', cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5,  # 第一次重启的epoch数
    #     T_mult=2,  # 每次重启后周期的倍数
    #     eta_min=1e-6
    # )

    # Mixed precision training scaler
    scaler = torch.amp.GradScaler() if device == 'cuda' or device == 'cuda:0' or device == 'cuda:1' or device == 'cuda:2' else None

    # 添加最佳准确率追踪
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, writer, tokenized_classes, classes, scaler, mixrs, args)
        scheduler.step()

        # Evaluate the model at specified interval
        if (epoch + 1) % args.eval_interval == 0:
            current_accuracy = evaluate_model(model, test_dataloader, loss_fn, tokenized_classes, device, epoch, writer, classes, args, cls_num_list=dataset_cls_num_list[0])
            
            # 保存最佳模型
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                os.makedirs(log_file_path, exist_ok=True)
                best_model_path = os.path.join(log_file_path, f'best_model_acc.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"保存新的最佳模型，准确率: {best_accuracy:.2f}%")
                logging.info(f"保存新的最佳模型，准确率: {best_accuracy:.2f}%")

    # 最终评估
    final_accuracy = evaluate_model(model, test_dataloader, loss_fn, tokenized_classes, device, args.epochs, writer, classes, args, cls_num_list=dataset_cls_num_list[0])

    os.makedirs(log_file_path, exist_ok=True)
    final_model_path = os.path.join(log_file_path, f'final_model_epoch{args.epochs}_acc{final_accuracy:.2f}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"保存最终模型，准确率: {final_accuracy:.2f}%")
    logging.info(f"保存最终模型，准确率: {final_accuracy:.2f}%")

    # Close the tensorboard writer
    writer.close()

if __name__ == "__main__":
    main()
