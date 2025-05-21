import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def elementwise_cosine_similarity(A, B):
    # 计算每个元素的范数
    A_norm = torch.norm(A, dim=0, keepdim=True)
    B_norm = torch.norm(B, dim=0, keepdim=True)

    # 避免除零错误，添加一个非常小的值
    epsilon = 1e-8
    A_norm = A_norm + epsilon
    B_norm = B_norm + epsilon

    # 归一化
    A_normalized = A / A_norm
    B_normalized = B / B_norm

    # 逐元素计算余弦相似度
    cosine_sim = A_normalized * B_normalized
    return cosine_sim


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x)


class ImageFeatureDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.base_path = '/data03/Quilt-1M/quilt_1m'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.base_path, self.image_paths[idx])
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, idx
        except:
            return None, idx


def extract_features_batch(df, model, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = ImageFeatureDataset(df['image_path'].values, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    features_list = []
    valid_indices = []

    with torch.no_grad():
        for batch, indices in tqdm(dataloader):
            if batch is None:
                continue

            batch = batch.to(device)
            features = model(batch)
            features = features.squeeze()
            features = features.cpu().numpy()

            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            features_list.append(features)
            valid_indices.extend(indices.numpy())

    if len(features_list) == 0:
        return None, []

    return np.vstack(features_list), valid_indices


def select_diverse_samples(df, target_size=10000, similarity_threshold=0.85):
    model = ResNet18FeatureExtractor()

    print("Extracting image features...")
    features_array, valid_indices = extract_features_batch(
        df,
        model,
        batch_size=32,
        num_workers=4
    )

    if features_array is None:
        return df.index[:min(len(df), target_size)]

    features_tensor = torch.tensor(features_array)
    selected_indices = [valid_indices[0]]
    current_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_tensor = features_tensor.to(device)

    print("Selecting diverse samples...")
    for i in tqdm(range(1, len(features_tensor))):
        if current_size >= target_size:
            break

        current_features = features_tensor[i].unsqueeze(0)
        selected_features = features_tensor[torch.tensor(selected_indices)]

        similarities = elementwise_cosine_similarity(
            current_features.T,
            selected_features.T
        )

        if not torch.any(similarities > similarity_threshold):
            selected_indices.append(valid_indices[i])
            current_size += 1

    return df.index[selected_indices]


def create_balanced_dataset(csv_path, train_output_path, test_output_path):
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)

    print("Cleaning data...")
    df = df[df['corrected_text'].notna()]
    df = df.drop_duplicates(subset=['image_path'])

    def safe_eval(x):
        try:
            if '(' in x:
                x = x[:x.find('(')]
            return eval(x)
        except:
            return []

    df['pathology'] = df['pathology'].apply(safe_eval)

    target_labels = [
        'Dermatopathology', 'Breast pathology', 'Gastrointestinal',
        'Pulmonary', 'Cytopathology', 'Genitourinary', 'Gynecologic',
        'Endocrine', 'Bone', 'Cardiac'
    ]

    def get_single_label(labels):
        target_matches = [label for label in labels if label in target_labels]
        return target_matches[0] if len(target_matches) == 1 else None

    df['single_label'] = df['pathology'].apply(get_single_label)
    df = df.dropna(subset=['single_label'])

    balanced_dfs = []
    for label in target_labels:
        class_df = df[df['single_label'] == label].copy()

        if label == 'Dermatopathology':
            class_df = class_df.drop_duplicates(subset=['corrected_text'])
            selected_indices = select_diverse_samples(
                class_df,
                target_size=10000,
                similarity_threshold=0.85
            )
            class_df = class_df.loc[selected_indices]
        elif label in ['Gastrointestinal', 'Pulmonary']:
            class_df = class_df.drop_duplicates(subset=['corrected_text'])

        balanced_dfs.append(class_df)
        print(f"{label}: {len(class_df)} samples")

    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df[['image_path', 'caption', 'corrected_text', 'single_label']]

    train_dfs = []
    test_dfs = []

    for label in target_labels:
        class_df = final_df[final_df['single_label'] == label]
        train_df, test_df = train_test_split(
            class_df,
            test_size=0.2,
            random_state=42
        )
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print("\nDataset Statistics:")
    print(f"Total samples: {len(final_df)}")
    print("\nTrain set distribution:")
    print(train_df['single_label'].value_counts())
    print("\nTest set distribution:")
    print(test_df['single_label'].value_counts())


if __name__ == "__main__":
    input_csv = "/AA/AA/AA.csv"
    train_output = "/AA/AA/AA.csv"
    test_output = "/AA/AA/AA.csv"
    create_balanced_dataset(input_csv, train_output, test_output)