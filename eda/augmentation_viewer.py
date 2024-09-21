import streamlit as st
import sys
import os
import pandas as pd
import random
import numpy as np
import torch
import json


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

set_seed(42)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    st.title("Data Augmentation Viewer")

    config = load_config('./configs/config.json')

    dataset_type = st.radio("Select dataset", ["Train", "Test"]) 
    is_train = dataset_type == "Train"

    csv_path = config['train_csv'] if is_train else config['test_csv']
    info_df = pd.read_csv(csv_path)

    data_dir = config['traindata_dir'] if is_train else config['testdata_dir']

    library_type = st.selectbox("Select transformation library", ["torchvision", "albumentations"])

    original_selector = TransformSelector(library_type)
    sketch_selector = SketchTransformSelector(library_type)

    original_transform = original_selector.get_transform(is_train=is_train)
    sketch_transform = sketch_selector.get_transform(is_train=is_train)

    original_dataset = CustomDataset(data_dir, info_df, original_transform, is_inference=not is_train)
    sketch_dataset = CustomDataset(data_dir, info_df, sketch_transform, is_inference=not is_train)

    col1, col2 = st.columns([3, 1])
    with col1:
        image_index = st.slider("Select image index", 0, len(original_dataset) - 1, 0)
    with col2:
        image_index = st.number_input("Or enter index", min_value=0, max_value=len(original_dataset) - 1, value=image_index, step=1)

    original_image = original_dataset.get_original_image(image_index)

    # 첫 번째 줄: 원본 이미지와 원본 변환 이미지
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

    with col2:
        st.subheader("Original Transform")
        if is_train:
            original_augmented, _ = original_dataset[image_index]
        else:
            original_augmented = original_dataset[image_index]
        
        original_augmented_image = original_augmented.permute(1, 2, 0).numpy()
        st.image(original_augmented_image, use_column_width=True, clamp=True)
        st.write(f"Shape: {original_augmented_image.shape}")
        st.write(f"Min: {original_augmented_image.min().item():.4f}, Max: {original_augmented_image.max().item():.4f}")
        st.write(f"Mean: {original_augmented_image.mean().item():.4f}, Std: {original_augmented_image.std().item():.4f}")

    # 두 번째 줄: 스케치 변환 이미지 3장
    st.subheader("Sketch Transforms")
    col1, col2, col3 = st.columns(3)

    for i, col in enumerate([col1, col2, col3]):
        with col:
            if is_train:
                sketch_augmented, _ = sketch_dataset[image_index]
            else:
                sketch_augmented = sketch_dataset[image_index]
            
            sketch_augmented_image = sketch_augmented.permute(1, 2, 0).numpy()
            st.image(sketch_augmented_image, use_column_width=True, clamp=True)
            st.write(f"Shape: {sketch_augmented_image.shape}")
            st.write(f"Min: {sketch_augmented_image.min().item():.4f}, Max: {sketch_augmented_image.max().item():.4f}")
            st.write(f"Mean: {sketch_augmented_image.mean().item():.4f}, Std: {sketch_augmented_image.std().item():.4f}")

if __name__ == "__main__":
    main()