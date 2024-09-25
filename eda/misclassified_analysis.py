import sys
import os
import pandas as pd
import torch
import json
import pickle
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.transforms import TransformSelector
from src.sketch_transforms import SketchTransformSelector
from src.dataset import CustomDataset
from src.models import ModelSelector

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

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

def load_dataset(config, library_type):
    info_df = pd.read_csv(config['train_csv'])
    
    sketch_selector = SketchTransformSelector(library_type)
    sketch_transform = sketch_selector.get_transform(is_train=True)
    
    dataset = CustomDataset(
        config['traindata_dir'], 
        info_df, 
        sketch_transform, 
        is_inference=False
    )
    
    return dataset, info_df

def get_misclassified_images(model, device, dataset, info_df):
    misclassified = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing images"):
            data = dataset[i]
            
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
    
    return misclassified

def main():
    config = load_config('./configs/config.json')
    model, device = load_model(config)
    
    # pkl 파일 로드
    with open('./eda/class_mapping.pkl', 'rb') as f:
        label_to_text = pickle.load(f)
    
    # 라벨을 텍스트로 변환하는 함수
    def label_to_text_func(label):
        return label_to_text.get(label, f"Unknown label: {label}")
    
    dataset, info_df = load_dataset(config, 'torchvision')
    misclassified = get_misclassified_images(model, device, dataset, info_df)
    
    # 결과를 DataFrame으로 변환
    results = []
    for idx, pred, true in misclassified:
        results.append({
            'Image Index': idx,
            'Predicted Class (Number)': pred,
            'Predicted Class (Text)': label_to_text_func(pred),
            'True Class (Number)': true,
            'True Class (Text)': label_to_text_func(true)
        })
    
    df = pd.DataFrame(results)
    
    # CSV 파일로 저장
    output_path = './eda/misclassified_train.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved misclassified train data to {output_path}")
    print(f"Total misclassified images: {len(misclassified)}")

if __name__ == "__main__":
    main()