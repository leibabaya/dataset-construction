import os
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Set
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import shutil
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self):
        # 需要移除的诊断相关表达
        self.diagnosis_patterns = [
            # 诊断结论
            r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*[^.]*',
            r'final\s+diagnosis\s*(?::|is)[^.]*',
            r'pathologic\s+diagnosis\s*(?::|is)[^.]*',

            # 明确的诊断描述
            r'(?:consistent|compatible)\s+with\s+[^.]*(?:melanoma|lymphoma|carcinoma)[^.]*',
            r'(?:demonstrates?|shows?|reveals?|represents?|exhibits?)\s+[^.]*(?:melanoma|lymphoma|carcinoma)[^.]*',

            # 特定疾病名称
            r'(?:malignant\s+)?melanoma\b',
            r'(?:cutaneous\s+)?lymphoma\b',
            r'(?:basal|squamous)\s+cell\s+carcinoma\b',
            r'mycosis\s+fungoides\b',

            # 结论性表述
            r'sections?\s+demonstrate\s+[^.]*(?:melanoma|lymphoma|carcinoma)[^.]*',
            r'histologic\s+examination\s+shows\s+[^.]*(?:melanoma|lymphoma|carcinoma)[^.]*',

            # 其他诊断相关词汇
            r'diagnosis\s+is\s+[^.]*',
            r'diagnostic\s+of\s+[^.]*',
            r'consistent\s+with\s+[^.]*',
            r'compatible\s+with\s+[^.]*',
            r'confirmatory\s+of\s+[^.]*',
            r'confirmed\s+as\s+[^.]*',
            r'confirming\s+[^.]*',
            r'diagnostic\s+impression[^.]*',
            r'impression[^.]*'
        ]

        # 编译正则表达式以提高性能
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.diagnosis_patterns]

    def clean_text(self, text: str) -> str:
        """移除文本中的诊断信息，保留形态学描述"""
        if pd.isna(text):
            return ""

        cleaned_text = text

        # 移除诊断相关表述
        for pattern in self.compiled_patterns:
            cleaned_text = pattern.sub('', cleaned_text)

        # 清理多余的空白字符
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s*\.\s*', '. ', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text


class MalignantAnalyzer:
    def __init__(self):
        self.malignant_types = {
            'melanoma': {
                'patterns': [
                    # 明确的诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:malignant\s+)?melanoma\b',
                    r'(?:consistent|compatible)\s+with\s+(?:malignant\s+)?melanoma\b',
                    r'final\s+diagnosis\s*(?::|is)\s*(?:malignant\s+)?melanoma\b',
                    r'pathologic\s+diagnosis\s*(?::|is)\s*(?:malignant\s+)?melanoma\b',
                    # 明确的描述
                    r'(?:demonstrates?|shows?|reveals?|represents?|exhibits?)\s+(?:malignant\s+)?melanoma\b',
                    # 特定类型
                    r'(?:invasive|metastatic|nodular|superficial\s+spreading)\s+(?:malignant\s+)?melanoma\b',
                    # 额外的明确表述
                    r'sections?\s+demonstrate\s+(?:malignant\s+)?melanoma\b',
                    r'histologic\s+examination\s+shows\s+(?:malignant\s+)?melanoma\b'
                ],
                'exclusion_patterns': [
                    # 排除不确定性
                    r'(?:no|without|negative\s+for)\s+(?:evidence\s+of\s+)?melanoma\b',
                    r'rule\s+out\s+melanoma\b',
                    r'suspicious\s+for\s+melanoma\b',
                    # 排除其他类型的明确诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:basal|squamous)\s+cell\s+carcinoma\b',
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*lymphoma\b'
                ]
            },
            'squamous_cell_carcinoma': {
                'patterns': [
                    # 明确的诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*squamous\s+cell\s+carcinoma\b',
                    r'(?:consistent|compatible)\s+with\s+squamous\s+cell\s+carcinoma\b',
                    r'final\s+diagnosis\s*(?::|is)\s*squamous\s+cell\s+carcinoma\b',
                    r'pathologic\s+diagnosis\s*(?::|is)\s*squamous\s+cell\s+carcinoma\b',
                    # 明确的描述
                    r'(?:demonstrates?|shows?|reveals?|represents?|exhibits?)\s+squamous\s+cell\s+carcinoma\b',
                    # 特定描述
                    r'(?:invasive|well\s+differentiated)\s+squamous\s+cell\s+carcinoma\b',
                    # 额外的明确表述
                    r'sections?\s+demonstrate\s+squamous\s+cell\s+carcinoma\b',
                    r'histologic\s+examination\s+shows\s+squamous\s+cell\s+carcinoma\b'
                ],
                'exclusion_patterns': [
                    r'(?:no|without|negative\s+for)\s+(?:evidence\s+of\s+)?(?:squamous|scc)\b',
                    r'rule\s+out\s+(?:squamous|scc)\b',
                    r'suspicious\s+for\s+(?:squamous|scc)\b',
                    # 排除其他类型的明确诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*melanoma\b',
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:basal\s+cell\s+carcinoma|lymphoma)\b'
                ]
            },
            'basal_cell_carcinoma': {
                'patterns': [
                    # 明确的诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*basal\s+cell\s+carcinoma\b',
                    r'(?:consistent|compatible)\s+with\s+basal\s+cell\s+carcinoma\b',
                    r'final\s+diagnosis\s*(?::|is)\s*basal\s+cell\s+carcinoma\b',
                    r'pathologic\s+diagnosis\s*(?::|is)\s*basal\s+cell\s+carcinoma\b',
                    # 明确的描述
                    r'(?:demonstrates?|shows?|reveals?|represents?|exhibits?)\s+basal\s+cell\s+carcinoma\b',
                    # 特定描述
                    r'(?:invasive|nodular|superficial)\s+basal\s+cell\s+carcinoma\b',
                    # 额外的明确表述
                    r'sections?\s+demonstrate\s+basal\s+cell\s+carcinoma\b',
                    r'histologic\s+examination\s+shows\s+basal\s+cell\s+carcinoma\b'
                ],
                'exclusion_patterns': [
                    r'(?:no|without|negative\s+for)\s+(?:evidence\s+of\s+)?(?:basal|bcc)\b',
                    r'rule\s+out\s+(?:basal|bcc)\b',
                    r'suspicious\s+for\s+(?:basal|bcc)\b',
                    # 排除其他类型的明确诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*melanoma\b',
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:squamous\s+cell\s+carcinoma|lymphoma)\b'
                ]
            },
            'lymphoma': {
                'patterns': [
                    # 明确的诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:cutaneous\s+)?lymphoma\b',
                    r'(?:consistent|compatible)\s+with\s+(?:cutaneous\s+)?lymphoma\b',
                    r'final\s+diagnosis\s*(?::|is)\s*(?:cutaneous\s+)?lymphoma\b',
                    r'pathologic\s+diagnosis\s*(?::|is)\s*(?:cutaneous\s+)?lymphoma\b',
                    # 明确的描述
                    r'(?:demonstrates?|shows?|reveals?|represents?|exhibits?)\s+(?:cutaneous\s+)?lymphoma\b',
                    # 特定描述
                    r'(?:cutaneous\s+t-cell|b-cell)\s+lymphoma\b',
                    r'primary\s+cutaneous\s+lymphoma\b',
                    r'mycosis\s+fungoides\b',
                    # 额外的明确表述
                    r'sections?\s+demonstrate\s+(?:cutaneous\s+)?lymphoma\b',
                    r'histologic\s+examination\s+shows\s+(?:cutaneous\s+)?lymphoma\b'
                ],
                'exclusion_patterns': [
                    r'(?:no|without|negative\s+for)\s+(?:evidence\s+of\s+)?lymphoma\b',
                    r'rule\s+out\s+lymphoma\b',
                    r'suspicious\s+for\s+lymphoma\b',
                    # 排除其他类型的明确诊断
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*melanoma\b',
                    r'(?:diagnosis|diagnostic|confirmed)\s*(?:of|as|:|is)\s*(?:basal|squamous)\s+cell\s+carcinoma\b'
                ]
            }
        }

    def analyze_text(self, text: str) -> List[str]:
        """分析文本并返回匹配的恶性肿瘤类型，确保严格互斥"""
        if pd.isna(text):
            return []

        text = text.lower()
        matched_types = []

        # 核心不确定性表达
        uncertainty_patterns = [
            r'versus\b',
            r'vs\.',
            r'differential\s+diagnosis\b',
            r'suspicious\s+for\b',
            r'rule\s+out\b',
            r'cannot\s+exclude\b',
            r'possible\b',
            r'probable\b'
        ]

        # 如果包含不确定性表达，直接返回空
        if any(re.search(pattern, text) for pattern in uncertainty_patterns):
            return []

        for tumor_type, patterns in self.malignant_types.items():
            # 首先检查是否有排除模式
            has_exclusion = any(re.search(pattern, text)
                                for pattern in patterns['exclusion_patterns'])

            if not has_exclusion:
                # 然后检查是否有明确的诊断模式
                if any(re.search(pattern, text) for pattern in patterns['patterns']):
                    matched_types.append(tumor_type)

        # 如果匹配到多个类型，返回空列表以确保互斥性
        return matched_types if len(matched_types) == 1 else []


def create_dermato_dataset():
    """创建皮肤病理学数据集"""
    # 删除已存在的数据集
    target_dir = Path("/AA/AA/AA")
    if target_dir.exists():
        logger.info("Removing existing dataset...")
        shutil.rmtree(target_dir)

    # 基础路径设置
    source_dir = Path("/AA/AA/AA")

    # 创建必要的目录
    for split in ['train', 'test']:
        for modal in ['images', 'texts']:
            for label in ['melanoma', 'lymphoma', 'squamous_cell_carcinoma', 'basal_cell_carcinoma']:
                (target_dir / modal / split / label).mkdir(parents=True, exist_ok=True)

    (target_dir / 'metadata').mkdir(parents=True, exist_ok=True)

    # 图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    # 1. 读取和分析数据
    logger.info("Loading and analyzing dataset...")
    analyzer = MalignantAnalyzer()
    text_cleaner = TextCleaner()
    df = pd.read_csv("/AA/AA/AA.csv")

    # 过滤掉corrected_text为空的样本
    df = df[df['corrected_text'].notna()]

    def clean_pathology(x):
        if pd.isna(x):
            return []
        try:
            match = re.search(r'\[(.*?)\]', str(x))
            if match:
                items = [item.strip().strip("'\"") for item in match.group(1).split(',')]
                return [item for item in items if item]
            return []
        except:
            return []

    df['pathology'] = df['pathology'].apply(clean_pathology)
    derm_df = df[df['pathology'].apply(lambda x: 'Dermatopathology' in x)].copy()
    logger.info(f"Found {len(derm_df)} dermatopathology samples")

    # 2. 分析肿瘤类型和清理文本
    dataset_info = []
    for idx, row in tqdm(derm_df.iterrows(), desc="Analyzing samples"):
        tumor_types = analyzer.analyze_text(row['corrected_text'])
        if len(tumor_types) == 1:
            # 清理文本，移除诊断信息
            cleaned_text = text_cleaner.clean_text(row['corrected_text'])

            # 确保清理后的文本不为空
            if cleaned_text.strip():
                dataset_info.append({
                    'image_path': row['image_path'],
                    'caption': row['caption'],
                    'corrected_text': cleaned_text,
                    'single_label': tumor_types[0]
                })

    logger.info(f"Found {len(dataset_info)} valid samples")

    # 输出每个类别的样本数
    label_counts = pd.DataFrame(dataset_info)['single_label'].value_counts()
    logger.info("\nSample distribution:")
    for label, count in label_counts.items():
        logger.info(f"{label}: {count}")

    # 3. 训练测试集划分
    train_data, test_data = train_test_split(
        dataset_info,
        test_size=0.2,
        stratify=[x['single_label'] for x in dataset_info],
        random_state=42
    )

    # 4. 处理和保存数据
    def process_split(data, split_name):
        processed_rows = []
        for item in tqdm(data, desc=f"Processing {split_name}"):
            try:
                # 源文件路径
                src_img_path = source_dir / item['image_path']

                # 目标文件路径
                label_dir = item['single_label']
                dst_img_path = target_dir / 'images' / split_name / label_dir / src_img_path.name
                dst_txt_path = target_dir / 'texts' / split_name / label_dir / f"{src_img_path.stem}.txt"

                # 处理图像
                with Image.open(src_img_path) as img:
                    img = img.convert('RGB')
                    img = transform(img)
                    img.save(dst_img_path)

                # 保存文本
                dst_txt_path.write_text(item['corrected_text'])

                # 记录相对路径 - 只保存文件名
                processed_rows.append({
                    'image_path': src_img_path.name,
                    'caption': item['caption'],
                    'corrected_text': item['corrected_text'],
                    'single_label': item['single_label']
                })

            except Exception as e:
                logger.error(f"Error processing {src_img_path}: {e}")
                continue

        return pd.DataFrame(processed_rows)

    # 处理训练集和测试集
    logger.info("\nProcessing training and test sets...")
    train_df = process_split(train_data, 'train')
    test_df = process_split(test_data, 'test')

    # 保存元数据 - 确保列的顺序正确
    columns = ['image_path', 'caption', 'corrected_text', 'single_label']
    train_df = train_df[columns]
    test_df = test_df[columns]

    train_df.to_csv(target_dir / 'metadata' / 'train_metadata.csv', index=False)
    test_df.to_csv(target_dir / 'metadata' / 'test_metadata.csv', index=False)

    # 5. 输出数据集统计信息
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total samples: {len(train_df) + len(test_df)}")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Testing samples: {len(test_df)}")

    logger.info("\nClass distribution in training set:")
    train_dist = train_df['single_label'].value_counts()
    for label, count in train_dist.items():
        logger.info(f"{label}: {count} ({count / len(train_df) * 100:.1f}%)")

    logger.info("\nClass distribution in testing set:")
    test_dist = test_df['single_label'].value_counts()
    for label, count in test_dist.items():
        logger.info(f"{label}: {count} ({count / len(test_df) * 100:.1f}%)")

    # 保存数据集统计信息
    stats = {
        'total_samples': len(train_df) + len(test_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_distribution': train_dist.to_dict(),
        'test_distribution': test_dist.to_dict()
    }

    with open(target_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)

    logger.info("\nDataset creation completed!")
    logger.info(f"Dataset saved to: {target_dir}")


if __name__ == "__main__":
    create_dermato_dataset()