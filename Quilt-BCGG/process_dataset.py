import pandas as pd
from pathlib import Path
import shutil
import logging
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import numpy as np


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def create_directory_structure(base_dir):
    """创建数据集目录结构"""
    categories = [
        'Gynecologic', 'Breast pathology', 'Cardiac', 'Gastrointestinal'
    ]
    splits = ['train', 'test']
    types = ['images', 'texts']

    for type_dir in types:
        for split in splits:
            for category in categories:
                path = base_dir / type_dir / split / category.replace(' ', '_')
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {path}")

    metadata_dir = base_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)


def process_image(image_path):
    """处理图像到模型所需格式"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),  # 保持比例缩放到256
            transforms.CenterCrop(224),  # 中心裁剪到224x224
        ])
        image = transform(image)
        return image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None


def process_dataset(csv_path, base_dir, split, source_image_dir):
    """处理数据集，对不同类别使用不同的采样策略"""
    df = pd.read_csv(csv_path)

    # 定义类别及其采样策略
    category_sampling = {
        'Gynecologic': 0.5,  # 使用50%数据
        'Breast pathology': 0.5,  # 使用50%数据
        'Cardiac': 1.0,  # 使用全部数据
        # 'Bone': 1.0,  # 使用全部数据
        'Gastrointestinal': 1.0  # 使用全部数据
    }

    # 保留选定的类别
    df = df[df['single_label'].isin(category_sampling.keys())]

    # 如果是训练集，对不同类别进行不同比例的采样
    if split == 'train':
        sampled_dfs = []
        for category, ratio in category_sampling.items():
            category_df = df[df['single_label'] == category]
            if ratio < 1.0:
                n_samples = int(len(category_df) * ratio)
                category_df = category_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(category_df)
        df = pd.concat(sampled_dfs, ignore_index=True)

        # 保存采样后的训练集元数据
        sampled_csv_path = Path(str(csv_path).replace('.csv', '_sampled.csv'))
        df.to_csv(sampled_csv_path, index=False)
        logging.info(f"Saved sampled training set metadata to: {sampled_csv_path}")

    total = len(df)
    success_count = 0
    failed_count = 0

    for _, row in tqdm(df.iterrows(), total=total, desc=f"Processing {split} set"):
        try:
            file_name = Path(row['image_path']).name
            base_name = Path(file_name).stem

            src_image_path = Path(source_image_dir) / row['image_path']
            category = row['single_label'].replace(' ', '_')

            dst_image_path = base_dir / 'images' / split / category / file_name
            dst_text_path = base_dir / 'texts' / split / category / f"{base_name}.txt"

            # 处理图像
            if src_image_path.is_file():
                processed_image = process_image(src_image_path)
                if processed_image:
                    processed_image.save(dst_image_path)
                    success_count += 1
            else:
                logging.warning(f"Source image not found: {src_image_path}")
                failed_count += 1
                continue

            # 处理文本
            text_content = row['corrected_text']
            if pd.notna(text_content):
                dst_text_path.write_text(str(text_content))
            else:
                logging.warning(f"Empty text content for: {base_name}")

        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
            failed_count += 1
            continue

    logging.info(f"{split} set processing completed:")
    logging.info(f"Total: {total}, Success: {success_count}, Failed: {failed_count}")

    return success_count, failed_count, df


def save_metadata(train_df, test_df, base_dir):
    """保存元数据文件"""
    metadata_dir = base_dir / 'metadata'
    train_df.to_csv(metadata_dir / 'train_metadata.csv', index=False)
    test_df.to_csv(metadata_dir / 'test_metadata.csv', index=False)
    logging.info("Saved metadata files")


def main():
    # 设置路径
    base_dir = Path("/AA/AA/AA")
    train_csv = Path("/AA/AA/AA.csv")
    test_csv = Path("/AA/AA/AA.csv")
    source_image_dir = Path("/AA/AA/AA")

    setup_logging()

    try:
        # 创建目录结构
        create_directory_structure(base_dir)

        # 处理训练集和测试集
        train_success, train_failed, train_df = process_dataset(
            train_csv, base_dir, 'train', source_image_dir
        )
        test_success, test_failed, test_df = process_dataset(
            test_csv, base_dir, 'test', source_image_dir
        )

        # 保存元数据
        save_metadata(train_df, test_df, base_dir)

        # 输出最终统计
        logging.info("\nFinal Statistics:")
        logging.info(f"Training set - Success: {train_success}, Failed: {train_failed}")
        logging.info(f"Test set - Success: {test_success}, Failed: {test_failed}")

        # 输出每个类别的样本数量统计
        logging.info("\nClass distribution in training set:")
        train_dist = train_df['single_label'].value_counts()
        for category, count in train_dist.items():
            logging.info(f"{category}: {count}")

        logging.info("\nClass distribution in test set:")
        test_dist = test_df['single_label'].value_counts()
        for category, count in test_dist.items():
            logging.info(f"{category}: {count}")

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()