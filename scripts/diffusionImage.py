import os
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# 매핑 파일을 로드하여 클래스 ID와 설명의 사전을 반환합니다.
def load_mapping(mapping_file):
    """
    :param mapping_file: 매핑 파일 경로
    :return: 클래스 ID와 설명의 사전
    """
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                class_id, description = parts
                mapping[class_id] = description
    return mapping

# 설정 로드
config = {
    "train_csv": "./data/train.csv",
    "test_csv": "./data/test.csv",
    "result_path": "./train_result",
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.002,
    "model": {
        "type": "timm",
        "name": "convnext_base",
        "pretrained": True
    },
    "transform": "torchvision",
    "sketch_data_dir": "./data/train",
    "converted_data_dir": "./data/converted_images",
    "new_train_csv": "./data/new_train.csv",  # 새 CSV 파일 경로
    "mapping_file": "./data/imagenet_synset_to_definition.txt"
}

# Stable Diffusion 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

def convert_sketch_to_image(input_image_path, output_image_path, prompt="Make the input sketch image more realistic"):
    """
    스케치 이미지를 Stable Diffusion으로 변환하고 결과를 저장합니다.
    :param input_image_path: 스케치 이미지 경로
    :param output_image_path: 변환된 이미지를 저장할 경로
    :param prompt: 이미지 변환을 위한 프롬프트
    """
    # 스케치 이미지 로드
    sketch_image = Image.open(input_image_path).convert("RGB")

    # 이미지 변환 수행
    with torch.no_grad():
        result_image = pipe(prompt=prompt, 
                            num_inference_steps=20,
                            ).images[0]

    # 변환된 이미지 저장
    print(input_image_path)
    result_image.save(output_image_path)
    print(f"Converted image saved at: {output_image_path}")

# 디렉터리 생성
os.makedirs(config['converted_data_dir'], exist_ok=True)

# 매핑 파일 로드
mapping = load_mapping(config['mapping_file'])
# CSV 파일에서 스케치 이미지 경로 읽기
train_info = pd.read_csv(config['train_csv'])
# 'class_name' 열을 기준으로 DataFrame 정렬
train_info = train_info.sort_values(by='image_path').reset_index(drop=True)

train_info['converted_image_path'] = ""

# 새 CSV 파일의 초기 데이터프레임 생성
new_train_info = train_info.copy()
new_train_info['converted_image_path'] = ""

# 스케치 이미지를 변환하여 저장하고 경로를 CSV에 업데이트
for i, row in train_info.iterrows():
    class_name = row['class_name']  # 클래스 이름
    input_image_path = os.path.join(config['sketch_data_dir'], row['image_path'])
    
    # 클래스별 폴더 생성
    class_folder = os.path.join(config['converted_data_dir'], class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    output_image_path = os.path.join(class_folder, f"converted_{os.path.basename(row['image_path'])}")
    
    # 프롬프트 정의
    prompt = "more detailed sketch of " + mapping[class_name] 
    print(prompt)
    
    convert_sketch_to_image(input_image_path, output_image_path, prompt=prompt)
    
    # 변환된 이미지 경로를 새 CSV 파일에 업데이트
    new_train_info.at[i, 'converted_image_path'] = output_image_path

# 새 CSV 파일 저장
new_train_info.to_csv(config['new_train_csv'], index=False)
