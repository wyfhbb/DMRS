import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

class MixRS:
    def __init__(self, alpha=1.0, mask_num=7, scale_=2, scale=32, label_smoothing=0.05, filp_or_similarity='filp', Word_Similarity=None, device='cuda'):
        self.alpha = alpha
        self.mask_num = mask_num
        self.scale_ = scale_
        self.scale = scale
        self.label_smoothing = label_smoothing
        
        
        assert filp_or_similarity in ['filp', 'similarity'], "filp_or_similarity must be 'filp' or 'similarity'"
        self.filp_or_similarity = filp_or_similarity
        if self.filp_or_similarity == 'similarity':
            assert Word_Similarity is not None, "Word_Similarity must be provided when filp_or_similarity is 'similarity'"
            assert isinstance(Word_Similarity, (np.ndarray, torch.Tensor)), "Word_Similarity must be a numpy array or a torch tensor"
            # 直接转换为GPU tensor并保存
            if isinstance(Word_Similarity, np.ndarray):
                Word_Similarity = torch.from_numpy(Word_Similarity)
            self.Word_Similarity = Word_Similarity.to(device)  # 如果使用GPU
            # 预先将对角线置零
            self.Word_Similarity.fill_diagonal_(0)

    def one_hot(self, x, num_classes, on_value=1., off_value=0., device='cuda'):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

    def mixup_target(self, target, old_target, num_classes, lam, device='cuda'):
        off_value = self.label_smoothing / num_classes
        on_value = 1. - self.label_smoothing + off_value
        y1 = self.one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
        y2 = self.one_hot(old_target, num_classes, on_value=on_value, off_value=off_value, device=device)
        return y1 * lam + y2 * (1. - lam)

    def __call__(self, samples, targets, num_classes):
        batch_size = samples.size(0)
        token_count = self.mask_num ** 2
        
        # Generate flipped versions
        if self.filp_or_similarity == 'filp':
            # file版本是倒序
            flip_indices = torch.arange(batch_size-1, -1, -1)
            flip_samples = samples[flip_indices] 
            flip_targets = targets[flip_indices]
        if self.filp_or_similarity == 'similarity':
            # 直接在GPU上操作，避免CPU-GPU传输
            batch_similarities = self.Word_Similarity[targets][:, targets]
            # 直接在GPU上获取最相似样本的索引
            flip_indices = torch.argmax(batch_similarities, dim=1)
            flip_samples = samples[flip_indices]
            flip_targets = targets[flip_indices]
        
        # Generate mask using beta distribution
        ratio = 1-np.random.beta(self.alpha, self.alpha)
        mask_ratio = [ratio for _ in range(batch_size)]
        mask_count = [int(np.ceil(token_count * r)) for r in mask_ratio]
        mask_ratio = [count/token_count for count in mask_count]
        
        # Generate masks
        mask_idx = [np.random.permutation(token_count)[:count] for count in mask_count]
        mask = np.zeros((batch_size, token_count), dtype=int)
        for i in range(batch_size):
            mask[i][mask_idx[i]] = 1
            
        # Reshape and scale masks
        mask = [m.reshape((self.mask_num, self.mask_num)) for m in mask]
        mask_ = [m.repeat(self.scale_, axis=0).repeat(self.scale_, axis=1) for m in mask]
        mask = [m.repeat(self.scale, axis=0).repeat(self.scale, axis=1) for m in mask]
        
        # Convert mask to tensor and expand dimensions
        mask = torch.from_numpy(np.array(mask)).to(samples.device)
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        mask = mask[:, :, :samples.shape[2], :samples.shape[2]]
        
        
        # Mix samples using mask
        mixed_samples = samples * mask + flip_samples * (1-mask)
        
        # Mix targets
        mask_ratio = torch.Tensor(mask_ratio).to(samples.device)
        ratio = mask_ratio.unsqueeze(1).repeat(1, num_classes)
        mixed_targets = self.mixup_target(targets, flip_targets, num_classes, ratio, device=targets.device)
        
        return mixed_samples, mixed_targets, np.array(mask_), mask_ratio

    def visualize_mixing(self, original_samples, mixed_samples, original_targets, mixed_targets, mask, classes=None):
        def denormalize(x):
            return x.cpu().clone().numpy().transpose(0, 2, 3, 1)
        
        batch_size = len(original_samples)
        fig, axs = plt.subplots(batch_size, 4, figsize=(15, 3*batch_size))
        
        if batch_size == 1:
            axs = axs.reshape(1, -1)
            
        for i in range(batch_size):
            # Original image
            axs[i,0].imshow(denormalize(original_samples)[i])
            axs[i,0].set_title('Original')
            if classes:
                orig_label = classes[original_targets[i]]
                axs[i,0].set_xlabel(f'Class: {orig_label}')
            
            # Mask
            axs[i,1].imshow(mask[i], cmap='gray')
            axs[i,1].set_title('Mask')
            
            # Mixed image
            axs[i,2].imshow(denormalize(mixed_samples)[i])
            axs[i,2].set_title('Mixed')
            
            # Mixed target distribution
            if classes:
                axs[i,3].bar(range(len(mixed_targets[i])), mixed_targets[i].cpu())
                axs[i,3].set_title('Mixed Target Distribution')
                axs[i,3].set_xticks(range(len(mixed_targets[i])))
                # axs[i,3].set_xticklabels(classes, rotation=45, ha='right')
            print(f"mixed_targets[{i}]: {mixed_targets[i]}")
            
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    import argparse
    from Load_Imbalance_Data import IMBALANCE_DATA, STRONG_WEAK_DATASET
    import clip
    def get_dataset(args):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset_imb_split = IMBALANCE_DATA(
            root_path=args.dataset_path,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            split_test=True,
            imbalance=True,
            test_ratio=args.test_ratio
        )

        train_dataset = dataset_imb_split.train_dataset
        train_targets = dataset_imb_split.train_targets
        test_dataset = dataset_imb_split.test_dataset
        test_targets = dataset_imb_split.test_targets

        train_dataset = STRONG_WEAK_DATASET(train_dataset, train_targets, transform, transform_strong=None, transform_weak=None)
        test_dataset = STRONG_WEAK_DATASET(test_dataset, test_targets, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]), transform_strong=None, transform_weak=None)

        # Pre-tokenize the class names to avoid repeated CPU-GPU transfers
        classes = dataset_imb_split.classes
        tokenized_classes = clip.tokenize(classes).to(args.device if torch.cuda.is_available() else "cpu")

        return train_dataset, test_dataset, classes, tokenized_classes
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./Dataset/NWPU-RESISC45')
    parser.add_argument('--imb_type', type=str, default='exp')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    

    # Get datasets
    train_dataset, test_dataset, classes, tokenized_classes = get_dataset(args)
    
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    # 将文本转化为token
    text_token = clip.tokenize(classes).to(args.device)
    # 计算文本特征和相似度矩阵
    with torch.no_grad():
        # 提取文本特征
        text_features = model.encode_text(text_token)
        # 特征归一化
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # 计算相似度矩阵 
        similarity_matrix = (100.0 * text_features @ text_features.T).cpu().numpy()
    np.fill_diagonal(similarity_matrix, 0)
    print(similarity_matrix)
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize MixPro
    mixrs = MixRS(alpha=6, mask_num=14, scale_=1, scale=16, filp_or_similarity='similarity', Word_Similarity=similarity_matrix)
    
    # Get a batch and test MixPro
    batch = next(iter(train_loader))
    images = batch[0].to(args.device)  # Assuming the first element is images
    labels = batch[1].to(args.device)  # Assuming the second element is labels
    
    # Apply MixPro
    mixed_images, mixed_labels, mask, mask_ratio = mixrs(images, labels, num_classes=len(classes))
    print("classes:", classes)
    print("num_classes:", len(classes))
    
    # Visualize results
    fig = mixrs.visualize_mixing(images, mixed_images, labels, mixed_labels, mask, classes)
    plt.show()
    
    # Print information
    print("Original labels:", [classes[l] for l in labels.cpu().numpy()])
    print("Mixing ratios:", mask_ratio.cpu().numpy())

    # Save visualization
    fig.savefig('mixpro_visualization.png', bbox_inches='tight', dpi=300)