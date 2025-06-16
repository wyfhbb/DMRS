import torch
import clip
import torchvision.transforms as transforms
from Load_Imbalance_Data import IMBALANCE_DATA, STRONG_WEAK_DATASET
from tqdm import tqdm



path = '/home/wyf/workspace/DMRS/dataset/Dataset/NWPU-RESISC45'

# transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_imb_split = IMBALANCE_DATA(
    root_path=path,
    imb_type='exp',
    imb_factor=0.01,
    split_test=True,
    imbalance=True,
    test_ratio=0.2
)
classes = dataset_imb_split.classes
test_dataset = dataset_imb_split.test_dataset
test_targets = dataset_imb_split.test_targets

test_dataset = STRONG_WEAK_DATASET(test_dataset, test_targets, transform, transform_strong=None, transform_weak=None)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(model)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 提前计算文本特征
text = clip.tokenize(classes).to(device)

correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    # 预先计算文本特征
    text_features = model.encode_text(text)
    
    # 添加进度条
    for images, targets in tqdm(test_dataloader, desc="Evaluating", ncols=100):
        images = images.to(device)
        targets = targets.to(device)
        
        # 计算图像特征
        image_features = model.encode_image(images)
        
        # 计算相似度
        logits_per_image = (100.0 * image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1)
        
        # 获取预测结果
        pred = torch.argmax(probs, dim=1)
        
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 计算准确率
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')

# 计算每个类别的准确率
from sklearn.metrics import classification_report
print("\nDetailed Classification Report:")
print(classification_report(all_targets, all_preds, target_names=classes))


