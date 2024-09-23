import streamlit as st
import sys
import os
import pandas as pd
import random
import numpy as np
import torch
import json
import cv2
import time
from torchcam.methods import GradCAM
from tqdm import tqdm
from torch.utils.data import Dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transforms import TransformSelector
from src.sketch_transforms import SketchTransformSelector
# from src.dataset import CustomDataset
from src.models import ModelSelector


class CustomDataset(Dataset):
    def __init__(self, root_dir, info_df, transform, is_inference=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target

    def get_original_image(self, index):
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

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

@st.cache_resource
def load_all_datasets(config, library_type):
    train_info_df = pd.read_csv(config['train_csv'])
    test_info_df = pd.read_csv(config['test_csv'])

    original_selector = TransformSelector(library_type)
    sketch_selector = SketchTransformSelector(library_type)

    train_original_transform = original_selector.get_transform(is_train=True)
    train_sketch_transform = sketch_selector.get_transform(is_train=True)
    test_original_transform = original_selector.get_transform(is_train=False)
    test_sketch_transform = sketch_selector.get_transform(is_train=False)

    train_original_dataset = CustomDataset(config['traindata_dir'], train_info_df, train_original_transform, is_inference=False)
    train_sketch_dataset = CustomDataset(config['traindata_dir'], train_info_df, train_sketch_transform, is_inference=False)
    test_original_dataset = CustomDataset(config['testdata_dir'], test_info_df, test_original_transform, is_inference=True)
    test_sketch_dataset = CustomDataset(config['testdata_dir'], test_info_df, test_sketch_transform, is_inference=True)

    return {
        'train': {
            'original': train_original_dataset,
            'sketch': train_sketch_dataset,
            'info_df': train_info_df
        },
        'test': {
            'original': test_original_dataset,
            'sketch': test_sketch_dataset,
            'info_df': test_info_df
        }
    }

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

def get_misclassified_images(model, device, dataset, info_df, image_type):
    misclassified = []
    model.eval()
    
    # Streamlit 프로그레스 바 초기화
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_images = len(dataset)
    
    with torch.no_grad():
        for i in tqdm(range(total_images), desc=f"Processing {image_type} images"):
            data = dataset[i]
            
            # 데이터셋이 (이미지, 레이블) 튜플을 반환하는 경우
            if isinstance(data, tuple):
                image, _ = data
            else:
                image = data
            
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            
            _, pred = torch.max(output, 1)
            
            true_label = info_df.iloc[i]['target']
            
            if pred.item() != true_label:
                misclassified.append((i, pred.item(), true_label))
            
            # 프로그레스 바 업데이트
            progress = (i + 1) / total_images
            progress_bar.progress(progress)
            status_text.text(f"Processing {image_type} image {i+1}/{total_images}")
            
            # Streamlit이 업데이트를 표시할 시간을 주기 위해 잠시 대기
            time.sleep(0.01)
    
    # 프로그레스 바와 상태 텍스트 제거
    progress_bar.empty()
    status_text.empty()
    
    return misclassified

@st.cache_data
def cached_get_misclassified_images(_model, device, _dataset, info_df, image_type):
    return get_misclassified_images(_model, device, _dataset, info_df, image_type)

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
    
    # st.title("Data Augmentation Viewer")

    # config = load_config('./configs/config.json')

    # library_type = st.selectbox("Select transformation library", ["torchvision", "albumentations"])

    # all_datasets = load_all_datasets(config, library_type)

    # dataset_type = st.radio("Select dataset", ["Train", "Test"]) 
    
    # original_dataset = all_datasets[dataset_type.lower()]['original']
    # sketch_dataset = all_datasets[dataset_type.lower()]['sketch']
    # info_df = all_datasets[dataset_type.lower()]['info_df']

    # is_train = dataset_type == "Train"

    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     image_index = st.slider("Select image index", 0, len(original_dataset) - 1, 0)
    # with col2:
    #     image_index = st.number_input("Or enter index", min_value=0, max_value=len(original_dataset) - 1, value=image_index, step=1)

    # original_image = original_dataset.get_original_image(image_index)

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

    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            model, device = load_model(config)
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
        st.success("Model loaded successfully!")

    # Grad-CAM 시각화
    st.subheader("Grad-CAM Visualization")
    
    if st.button("Generate Grad-CAM"):
        if not st.session_state.model_loaded:
            st.warning("Please load the model first.")
        else:
            with st.spinner("Generating Grad-CAM..."):
                model = st.session_state.model
                device = st.session_state.device
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

    # 잘못 분류된 이미지 시각화
    st.subheader("Misclassified Images")
    
    image_type = st.radio("Select image type for misclassification analysis:", ("Original", "Sketch"))
    
    if 'misclassified' not in st.session_state:
        st.session_state.misclassified = None

    if st.button("Find Misclassified Images"):
        if not st.session_state.model_loaded:
            st.warning("Please load the model first.")
        else:
            with st.spinner(f"Finding misclassified {image_type.lower()} images..."):
                model = st.session_state.model
                device = st.session_state.device
                dataset = original_dataset if image_type == "Original" else sketch_dataset
                st.session_state.misclassified = cached_get_misclassified_images(model, device, dataset, info_df, image_type.lower())

    if st.session_state.misclassified is not None:
        misclassified = st.session_state.misclassified
        if misclassified:
            st.write(f"Found {len(misclassified)} misclassified {image_type.lower()} images.")
            selected_index = st.selectbox("Select a misclassified image:", 
                                          options=[f"Image {i}: Predicted {pred}, True {true}" 
                                                   for i, pred, true in misclassified],
                                          format_func=lambda x: x.split(":")[0])
            
            selected_image_index = int(selected_index.split("Image ")[1].split(":")[0])
            original_image = original_dataset.get_original_image(selected_image_index)
            selected_dataset = original_dataset if image_type == "Original" else sketch_dataset
            augmented_data = selected_dataset[selected_image_index]

            # 데이터셋이 (이미지, 레이블) 튜플을 반환하는 경우 처리
            if isinstance(augmented_data, tuple):
                augmented_image, _ = augmented_data
            else:
                augmented_image = augmented_data

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_column_width=True)
            with col2:
                st.subheader(f"{image_type} Augmented")
                st.image(augmented_image.permute(1, 2, 0).numpy(), use_column_width=True, clamp=True)

            # 예측 결과 표시
            for i, pred, true in misclassified:
                if i == selected_image_index:
                    st.write(f"{image_type} Prediction: {pred}")
                    st.write(f"True Label: {true}")
                    break
        else:
            st.write(f"No misclassified {image_type.lower()} images found.")

    # 모델 로드 상태 표시
    if st.session_state.model_loaded:
        st.sidebar.success("Model is loaded and ready to use.")
    else:
        st.sidebar.warning("Model is not loaded. Please load the model to use Grad-CAM and misclassification detection.")

if __name__ == "__main__":
    main()