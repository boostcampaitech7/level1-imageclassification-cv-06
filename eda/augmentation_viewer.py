import streamlit as st
import sys
import os
import pandas as pd
import random
import numpy as np
import torch
import json
from torch.utils.data import DataLoader

# 상위 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transforms import TransformSelector
from src.sketch_transforms import SketchTransformSelector
from src.dataset import CustomDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

# 설정 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    st.title("Data Augmentation Viewer")

    # 설정 파일 로드
    config = load_config('./configs/config.json')

    # 데이터셋 선택
    dataset_type = st.radio("Select dataset", ["Train", "Test"]) 
    is_train = dataset_type == "Train"

    # CSV 파일 로드
    csv_path = config['train_csv'] if is_train else config['test_csv']
    info_df = pd.read_csv(csv_path)

    # 데이터 디렉토리 선택
    data_dir = config['traindata_dir'] if is_train else config['testdata_dir']

    # 변환 타입 선택
    transform_type = st.selectbox("Select transform type", ["Original", "Sketch (0-1 scaling)"])

    # 변환 라이브러리 선택
    library_type = st.selectbox("Select transformation library", ["torchvision", "albumentations"])

    # TransformSelector 인스턴스 생성
    if transform_type == "Original":
        selector = TransformSelector(library_type)
    else:
        selector = SketchTransformSelector(library_type)

    # 데이터셋에 맞는 변환 가져오기
    transform = selector.get_transform(is_train=is_train)

    # CustomDataset 인스턴스 생성
    dataset = CustomDataset(data_dir, info_df, transform, is_inference=not is_train)

    # 이미지 인덱스 선택 (슬라이더와 숫자 입력 모두 사용)
    col1, col2 = st.columns([3, 1])
    with col1:
        image_index = st.slider("Select image index", 0, len(dataset) - 1, 0)
    with col2:
        image_index = st.number_input("Or enter index", min_value=0, max_value=len(dataset) - 1, value=image_index, step=1)

    # 원본 이미지 가져오기 및 표시
    original_image = dataset.get_original_image(image_index)

    # 증강된 이미지 가져오기
    if is_train:
        augmented, _ = dataset[image_index]
    else:
        augmented = dataset[image_index]
    augmented_image = augmented.permute(1, 2, 0).numpy()

    # 증강된 이미지를 0-1 범위로 변환
    normalized_image = (augmented_image - augmented_image.min()) / (augmented_image.max() - augmented_image.min())

    # 이미지들을 가로로 배치
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

    with col2:
        st.subheader("Augmented Image\n(Normalized, as fed to model)")
        st.image(augmented_image, use_column_width=True, clamp=True)

    with col3:
        st.subheader("Augmented Image\n(Scaled to 0-1 for visualization)")
        st.image(normalized_image, use_column_width=True)

if __name__ == "__main__":
    main()