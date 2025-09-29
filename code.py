import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from IPython.display import FileLink
from PIL import Image
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from tqdm import tqdm

# Define the source and destination directories
source_dir = '/kaggle/input/pascal-voc-2007/VOCtrainval_06-Nov-2007'
destination_dir = '/kaggle/working/pascal_voc_2007'

# Print the source and destination for confirmation
print("Source directory:", source_dir)
print("Destination directory:", destination_dir)

# Copy the dataset to the new working directory
shutil.copytree(source_dir, destination_dir)

# Confirm the copy operation
print("Dataset copied to:", destination_dir)
# Define the source and destination directories
source_dir = '/kaggle/working/pascal_voc_2007/VOCdevkit/VOC2007'
destination_dir = '/kaggle/working/'

# Print the source and destination for confirmation
print("Source directory:", source_dir)
print("Destination directory:", destination_dir)

# Move the contents of VOC2007 to the working directory
for item in os.listdir(source_dir):
    s = os.path.join(source_dir, item)
    d = os.path.join(destination_dir, item)
    # Use shutil.move to move files and directories
    shutil.move(s, d)

# Confirm the move operation
print("Contents of VOC2007 moved to:", destination_dir)

# Optionally, remove the now empty VOCdevkit directory
shutil.rmtree('/kaggle/working/pascal_voc_2007/VOCdevkit')

# Print final confirmation
print("VOCdevkit directory removed.")
def collect_object_types(annotation_dir):
    object_counter = Counter()
    xml_files = [file for file in os.listdir(annotation_dir) if file.endswith('.xml')]  
    
    # Iterate through all XML files in the annotations directory
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(annotation_dir, xml_file))
        root = tree.getroot()
        
        # Extract object classes
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            object_counter[class_name] += 1
    
    print(f"Processed {len(xml_files)} XML files.") 
    return object_counter
annotations_path = '/kaggle/working/Annotations'
object_counts = collect_object_types(annotations_path)


df = pd.DataFrame(object_counts.items(), columns=['Object Type', 'Count'])
df = df.sort_values(by='Count', ascending=False)

print(df)
plt.figure(figsize=(12, 6))
plt.bar(df['Object Type'], df['Count'], color='skyblue')
plt.xticks(rotation=45)
plt.title('Distribution of Object Types in Pascal VOC Dataset')
plt.xlabel('Object Types')
plt.ylabel('Count')
plt.grid(axis='y')


plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(df['Count'], labels=df['Object Type'], autopct='%1.1f%%', startangle=140)
plt.title('Object Type Distribution (Pie Chart)')
plt.axis('equal')  
plt.show()
transform = transforms.Compose([
    transforms.Resize((600, 600)),  
    transforms.RandomHorizontalFlip(0.5),  
    transforms.RandomRotation(degrees=15),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

print("Transformations set up for Faster R-CNN:")
print(transform)
class PascalVOCDataset:
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        self.images = []
        self.annotations = []
        self._load_data()

    def _load_data(self):
        image_dir = os.path.join(self.root, 'JPEGImages')
        annotation_dir = os.path.join(self.root, 'Annotations')
        if self.image_set == 'train':
            image_set_file = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        elif self.image_set == 'val':
            image_set_file = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')
        elif self.image_set == 'trainval':
            image_set_file = os.path.join(self.root, 'ImageSets', 'Main', 'trainval.txt')
        elif self.image_set == 'test':
            image_set_file = os.path.join(self.root, 'ImageSets', 'Main', 'test.txt') 
        else:
            raise ValueError("Invalid image_set: Choose from 'train', 'val', 'trainval', or 'test'.")

        with open(image_set_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        for img_id in image_ids:
            self.images.append(os.path.join(image_dir, f'{img_id}.jpg'))
            self.annotations.append(os.path.join(annotation_dir, f'{img_id}.xml'))

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        annotation = self._load_annotation(self.annotations[index])
        if self.transform:
            image = self.transform(image)
        return image, annotation


    def _load_annotation(self, annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        boxes = []
        labels = [] 
        label_map = self.create_label_map()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            boxes.append([
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ])
            label = obj.find('name').text  
            if label in label_map:
                labels.append(label_map[label])  
            else:
                print(f"Warning: {label} not found in label map.")
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) 
        labels_tensor = torch.tensor(labels, dtype=torch.int64) 
        return {'boxes': boxes_tensor, 'labels': labels_tensor}  



    def __len__(self):
        return len(self.images)

    def create_label_map(self):
        return {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19,
        }
root_path = '/kaggle/working/' 
train_dataset = PascalVOCDataset(root=root_path, image_set='train', transform=transform)
val_dataset = PascalVOCDataset(root=root_path, image_set='val', transform=transform)

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split_ratio = 0.8  
split_idx = int(np.floor(split_ratio * num_train))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
test_subset = torch.utils.data.Subset(train_dataset, test_indices)

def create_image_set_file(image_set, subset, root_path):
    image_set_file_path = os.path.join(root_path, 'ImageSets', 'Main', f'{image_set}.txt')
    with open(image_set_file_path, 'w') as f:
        for img_index in subset.indices:  
            image_id = os.path.basename(train_dataset.images[img_index]).split('.')[0]
            f.write(f"{image_id}\n")
create_image_set_file('train', train_subset, root_path)
create_image_set_file('test', test_subset, root_path)

new_trainval_dataset = torch.utils.data.ConcatDataset([train_subset, val_dataset])
def create_trainval_txt(new_trainval_dataset, train_subset, val_dataset, root_path):
    image_set_file_path = os.path.join(root_path, 'ImageSets', 'Main', 'trainval.txt')
    with open(image_set_file_path, 'w') as f:
        for img_index in train_subset.indices:
            image_id = os.path.basename(train_dataset.images[img_index]).split('.')[0]
            f.write(f"{image_id}\n")
        for img_index in range(len(val_dataset)):
            image_id = os.path.basename(val_dataset.images[img_index]).split('.')[0]
            f.write(f"{image_id}\n")                                                            

print("Number of training samples:", len(train_subset))
print("Number of validation samples:", len(val_dataset))
print("Number of trainval samples:", len(new_trainval_dataset))
print("Number of test samples:", len(test_subset))

create_trainval_txt(new_trainval_dataset, train_subset, val_dataset, root_path)
ef custom_collate_fn(batch):
    images, targets = zip(*batch)
    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)
    padded_images = []
    for img in images:
        padding = (0, max_width - img.size(2), 0, max_height - img.size(1))
        padded_image = F.pad(img, padding) 
        padded_images.append(padded_image)
    return torch.stack(padded_images), targets

batch_size = 4  
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=custom_collate_fn)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=custom_collate_fn)
print(f"Number of batches in train loader: {len(train_loader)}")
print(f"Number of batches in validation loader: {len(val_loader)}")
print(f"Number of batches in test loader: {len(test_loader)}")
# Step 1: Load ResNet-50 Backbone without Pre-trained Weights
backbone = resnet50(pretrained=False)
# Step 2: Remove the fully connected layers 
backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))  
backbone.out_channels = 2048  # ResNet-50 outputs 2048 feature maps
# Step 3: Create Anchor Generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), 
    aspect_ratios=((0.5, 1.0, 2.0),) * 5  
# Step 4: Create ROI Align Pooler
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],  
    output_size=7,  
    sampling_ratio=2)
# Step 5: Build the Faster R-CNN Model
model = FasterRCNN(
    backbone,
    num_classes=21,  
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler)
# Move model to the device 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# Set model to evaluation mode
model.eval()
# Debugging: Print the model architecture to verify
print(model)

# Create a dummy input tensor with the shape [N, C, H, W]
dummy_input = torch.rand(1, 3, 800, 800).to(device)  # Example size; adjust as needed

# Forward pass through the model without targets
with torch.no_grad():
    outputs = model(dummy_input)  # Forward pass through the model

# Print the outputs
print("Outputs from dummy input:", outputs)
class_counts = np.array([5447, 1644, 1432, 634, 625, 599, 538, 425, 418, 
                         406, 398, 390, 389, 367, 356, 353, 331, 328, 310, 272])
total_count = sum(class_counts)
class_weights = total_count / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  
num_epochs = 15  
learning_rate = 0.005  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
model.to(device)
print(f"Total batches: {len(train_loader)}")  
for epoch in range(num_epochs):
    model.train() 
    total_loss = 0
    for images, targets in tqdm(train_loader):  
        original_sizes = [img.size for img in images]  
        images = [image.to(device) for image in images]  
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)  
        losses = sum(loss for loss in loss_dict.values())  
        weighted_loss = 0
        if 'loss_classifier' in loss_dict:
            predicted_labels = torch.cat([t['labels'] for t in targets])  
            weighted_loss += (loss_dict['loss_classifier'] * class_weights[predicted_labels]).mean()
        weighted_loss += loss_dict['loss_box_reg'] + loss_dict['loss_objectness']
        weighted_loss.backward()  
        optimizer.step()  
        total_loss += weighted_loss.item()  
        transformed_sizes = [image.shape for image in images]
    print(f'Epoch #{epoch + 1} Loss: {total_loss / len(train_loader)}')
    
model_dir = '/kaggle/working/model_checkpoints'
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), f'{model_dir}/training_voc.pth')
def calculate_precision_recall(predictions, targets, iou_threshold=0.1):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        target_boxes = target['boxes']
        matched = [False] * len(target_boxes)
        for p_box in pred_boxes:
            ious = [calculate_iou(p_box.cpu().numpy(), t_box.cpu().numpy()) for t_box in target_boxes]
            if max(ious) >= iou_threshold:
                true_positive += 1
                matched[ious.index(max(ious))] = True
            else:
                false_positive += 1
        false_negative += matched.count(False)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return precision, recall
def calculate_map(predictions, targets, iou_thresholds=[0.1]):
    average_precisions = []
    for threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(predictions, targets, iou_threshold=threshold)
        average_precisions.append(precision)
    mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    return mAP
    precision, recall = calculate_precision_recall(all_predictions, all_targets)
mAP = calculate_map(all_predictions, all_targets)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}')
