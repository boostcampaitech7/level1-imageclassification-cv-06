import streamlit as st
import sys
import os
import pandas as pd
import random
import numpy as np
import torch
import json
import cv2
from torchcam.methods import GradCAM


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transforms import TransformSelector
from src.sketch_transforms import SketchTransformSelector
from src.dataset import CustomDataset
from src.models import ModelSelector

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

@st.cache_resource
def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_selector = ModelSelector(
        model_type=config['model']['type'],
        num_classes=500,
        model_name=config['model']['name'],
        pretrained=False
    )
    model = model_selector.get_model()
    model.load_state_dict(torch.load(os.path.join(config['result_path'], "best_model.pt"), map_location=device))
    model.to(device)
    model.eval()
    return model, device

def find_target_layer(model):
    #모델 구조 확인
    #print(model)
    target_layer = 'model.stages.3.blocks.2.conv_dw'  # ConvNeXt V2의 마지막 stage의 마지막 블록의 conv_dw 레이어
    return target_layer

def generate_gradcam(model, device, image, target_layer):
    model.eval()
    model.requires_grad_(True)
    cam_extractor = GradCAM(model, target_layer)
    
    image = image.unsqueeze(0).to(device)
    image.requires_grad_(True)
    
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    
    cam = cam_extractor(pred.item(), outputs)[0]
    cam = cam.cpu().detach().numpy()  # detach() 추가
    cam = np.mean(cam, axis=0)  # 여러 채널이 있는 경우 평균을 취합니다
    cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = np.uint8(255 * cam)
    
    # cam이 2D 배열인지 확인하고, 그렇지 않으면 2D로 변환
    if cam.ndim != 2:
        cam = np.mean(cam, axis=2)
    
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    
    # 입력 이미지 처리
    input_image = image.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))  # detach() 추가
    if input_image.shape[2] == 1:  # 1채널 이미지인 경우
        input_image = np.squeeze(input_image, axis=2)
        input_image = np.stack([input_image] * 3, axis=-1)
    else:  # 3채널 이미지인 경우
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        input_image = (input_image * 255).astype(np.uint8)
    
    # 오버레이 이미지 생성
    overlay = cam * 0.3 + input_image * 0.5
    overlay = np.uint8(overlay)
    
    model.requires_grad_(False)
    return overlay

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

    # Grad-CAM 시각화
    st.subheader("Grad-CAM Visualization")
    
    if st.button("Generate Grad-CAM"):
        with st.spinner("Generating Grad-CAM..."):
            model, device = load_model(config)
            target_layer = find_target_layer(model)

            original_overlay = generate_gradcam(model, device, original_augmented, target_layer)
            sketch_overlay = generate_gradcam(model, device, sketch_augmented, target_layer)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Grad-CAM Overlay")
            st.image(original_overlay, use_column_width=True)
        with col2:
            st.subheader("Sketch Grad-CAM Overlay")
            st.image(sketch_overlay, use_column_width=True)

if __name__ == "__main__":
    main()