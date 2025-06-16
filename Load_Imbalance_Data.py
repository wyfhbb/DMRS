import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm
import random
import os

class IMBALANCE_DATA:
    def __init__(
        self,
        root_path, 
        imb_type='exp', 
        imb_factor=0.01, 
        rand_number=0, 
        imbalance=True, 
        split_test=False, 
        test_ratio=0.2
    ):
        self.dataset = torchvision.datasets.ImageFolder(root_path)
        self.classes = self.dataset.classes
        self.targets = self.dataset.targets
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.rand_number = rand_number
        self.imbalance = imbalance
        self.split_test = split_test
        self.test_ratio = test_ratio
        
        np.random.seed(rand_number)
        
        if self.split_test:
            # Set random seed for reproducibility
            generator = torch.Generator().manual_seed(rand_number)
            
            # Calculate dataset lengths
            train_length = int(len(self.dataset) * (1 - test_ratio))
            test_length = int(len(self.dataset) * test_ratio)
            
            # Split dataset
            train_subset, test_subset = torch.utils.data.random_split(
                self.dataset,
                [train_length, test_length],
                generator=generator
            )
            
            # Extract images and targets from subsets
            # 显示进度
            self.train_dataset = [self.dataset[i][0] for i in tqdm(train_subset.indices, desc="Loading train dataset", ncols=100)]
            self.test_dataset = [self.dataset[i][0] for i in tqdm(test_subset.indices, desc="Loading test dataset", ncols=100)]
            self.train_targets = [self.dataset.targets[i] for i in tqdm(train_subset.indices, desc="Loading train targets", ncols=100)]
            self.test_targets = [self.dataset.targets[i] for i in tqdm(test_subset.indices, desc="Loading test targets", ncols=100)]
        else:
            self.train_dataset = [item[0] for item in tqdm(self.dataset, desc="Loading train dataset", ncols=100)]
            self.train_targets = self.targets
            
        if self.imbalance:
            self.gen_imbalance_data()
            
    
    def get_img_num_per_cls(self):
        """Calculate number of samples per class based on imbalance settings"""
        img_max = len(self.train_targets) // len(self.classes)
        img_num_per_cls = []
        
        if self.imb_type == 'exp':
            for cls_idx in range(len(self.classes)):
                num = img_max * (self.imb_factor**(cls_idx / (len(self.classes) - 1)))
                img_num_per_cls.append(int(num))
        elif self.imb_type == 'step':
            half_cls = len(self.classes) // 2
            img_num_per_cls.extend([int(img_max)] * half_cls)
            img_num_per_cls.extend([int(img_max * self.imb_factor)] * (len(self.classes) - half_cls))
        else:
            img_num_per_cls.extend([int(img_max)] * len(self.classes))
            
        return img_num_per_cls

    def gen_imbalance_data(self):
        """Generate imbalanced dataset based on specified settings"""
        img_num_per_cls = self.get_img_num_per_cls()
        new_data = []
        new_targets = []
        
        targets_np = np.array(self.train_targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        for cls_idx, img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == cls_idx)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:img_num]
            for idx in selec_idx:
                new_data.append(self.train_dataset[idx])
                new_targets.append(self.train_targets[idx])
                
        self.train_dataset = new_data
        self.train_targets = new_targets
    
    def get_cls_num_list(self):
        """返回每个类别的样本数量"""
        train_cls_num_list = [0] * len(self.classes)
        test_cls_num_list = [0] * len(self.classes)
        if self.split_test:
            for target in self.train_targets:
                train_cls_num_list[target] += 1
            for target in self.test_targets:
                test_cls_num_list[target] += 1
        else:
            for target in self.train_targets:
                train_cls_num_list[target] += 1
        return train_cls_num_list, test_cls_num_list

class STRONG_WEAK_DATASET(torch.utils.data.Dataset):
    def __init__(self, dataset, targets, transform, transform_strong=None, transform_weak=None):
        self.dataset = dataset
        self.targets = targets
        self.transform = transform
        self.transform_strong = transform_strong
        self.transform_weak = transform_weak

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = self.dataset[index]
        target = self.targets[index]
        if self.transform_strong is not None and self.transform_weak is not None and self.transform is None:
            return self.transform_strong(img), self.transform_weak(img), target
        elif self.transform_strong is None and self.transform_weak is None and self.transform is not None:
            return self.transform(img), target
        else:
            raise ValueError("Strong and weak transforms must both be specified or both be None")
        
class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch
        
class ImbalanceDataLoader(DataLoader):
    def __init__(
        self, 
        dataset,
        targets,
        num_classes,
        batch_size, 
        shuffle=True, 
        num_workers=1,
        balanced=False,
        retain_epoch_size=True
    ):
        self.dataset = dataset
        self.targets = targets
        self.num_classes = num_classes
        
        # Create class number list
        cls_num_list = [0] * num_classes
        for label in self.targets:
            cls_num_list[label] += 1
        self.cls_num_list = cls_num_list
        
        # Setup balanced sampling if requested
        sampler = None
        if balanced:
            buckets = [[] for _ in range(num_classes)]
            for idx, label in enumerate(self.targets):
                buckets[label].append(idx)
            sampler = BalancedSampler(buckets, retain_epoch_size)
            shuffle = False  # Disable shuffle when using sampler
            
        # Initialize DataLoader with proper parameters
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler
        )


if __name__ == '__main__':

    # 设置数据集路径
    root = './Dataset/RSD46-WHU'

    dataset_imb_split = IMBALANCE_DATA(
        root_path=root,
        imb_type='exp',
        imb_factor=0.01,
        split_test=True,
        imbalance=True,
        test_ratio=0.2
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset_imb_split.train_dataset = STRONG_WEAK_DATASET(dataset_imb_split.train_dataset, dataset_imb_split.train_targets, transform, transform_strong=None, transform_weak=None)
    print("\n2. 分离测试集，生成不平衡训练集：")
    data_loader = ImbalanceDataLoader(
        dataset=dataset_imb_split.train_dataset,
        targets=dataset_imb_split.train_targets,
        num_classes=len(dataset_imb_split.classes),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        balanced=True,
        retain_epoch_size=True
    )
    # 统计每个batch各个类别出现的概率
    cls_num_list = [0] * len(dataset_imb_split.classes)
    for images, labels in data_loader:
        for label in labels:
            cls_num_list[label] += 1
    print("每个batch各个类别出现的概率：", cls_num_list)
    
    # 展示各类别数量
    print("各类别数量：", dataset_imb_split.get_cls_num_list())
    
    # 按照ImageNet的方式存放数据集与验证集测试集和标签，测试集和验证集相同
    address = './ImageNet/RSD46-WHU'
    # 创建地址
    os.makedirs(address, exist_ok=True)
    os.makedirs(os.path.join(address, 'train'), exist_ok=True)
    os.makedirs(os.path.join(address, 'val'), exist_ok=True)
    os.makedirs(os.path.join(address, 'test'), exist_ok=True)
    
    # 创建类别子目录
    for split in ['train', 'val', 'test']:
        for cls in dataset_imb_split.classes:
            os.makedirs(os.path.join(address, split, cls), exist_ok=True)

    # 生成训练集标签文件
    to_pil = transforms.ToPILImage()
    with open(os.path.join(address, 'train.txt'), 'w') as f_train:
        for idx, (img_tensor, target) in tqdm(enumerate(dataset_imb_split.train_dataset), 
                                      desc="Generating train labels", total=len(dataset_imb_split.train_targets)):
            # 转换Tensor到PIL图像
            img = to_pil(img_tensor.cpu().contiguous())  # 调整通道顺序
            cls_name = dataset_imb_split.classes[target]
            # 使用normpath统一路径格式后替换分隔符
            img_path = os.path.normpath(os.path.join('train', cls_name, f'train_{idx:05d}.jpg')).replace(os.sep, '/')
            img.save(os.path.join(address, img_path))  # 保存时仍使用系统分隔符
            f_train.write(f'{img_path} {target}\n')  # 写入时强制使用/

    # 生成验证集/测试集标签文件（两者相同）
    with open(os.path.join(address, 'val.txt'), 'w') as f_val, \
         open(os.path.join(address, 'test.txt'), 'w') as f_test:
        for idx, (img, target) in tqdm(enumerate(zip(dataset_imb_split.test_dataset, dataset_imb_split.test_targets)), 
                                      desc="Generating val/test labels", total=len(dataset_imb_split.test_targets)):
            # 转换Tensor到PIL图像

            cls_name = dataset_imb_split.classes[target]
            for split in ['val', 'test']:
                # 统一处理路径分隔符
                img_path = os.path.normpath(os.path.join(split, cls_name, f'test_{idx:05d}.jpg')).replace(os.sep, '/')
                # 保存到系统路径时使用原始路径
                img.save(os.path.join(address, *img_path.split('/')))  # 分割路径组件
                (f_val if split == 'val' else f_test).write(f'{img_path} {target}\n')
    
