 # Boostcamp AI Tech 7 CV 06
 
## Sketch 이미지 데이터 분류
![image](https://github.com/user-attachments/assets/42997a1b-a4c3-4e5c-b67d-92b217dd277a)
### Description
Sketch이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.

원본 ImageNet Sketch 데이터셋은 50,889개의 이미지 데이터로 구성되어 있으며 1,000개의 객체에 대해 각각 대략 50개씩의 이미지를 가지고 있습니다. 일반적인 객체들의 핸드 드로잉 이미지로 구성되어 있으며, 실제 객체를 대표하는 다양한 스타일과 특징을 보여줍니다. 

이번 경진대회에서 제공되는 데이터셋은 원본데이터를 직접 검수하고 정제하여 1,000개의 클래스에서 정제 후 이미지 수량이 많은 상위 500개의 객체를 선정했으며 총 25,035개의 이미지 데이터가 활용됩니다. 해당 이미지 데이터는 15,021개의 학습데이터와 10,014개의 Private&Public 평가데이터로 나누어 구성했습니다.

스케치 이미지 분류 모델을 개발함으로써 비전 시스템의 범용성을 향상시키며 다양한 실제 어플리케이션에 적용할 수 있는 능력을 키울 수 있습니다.

## Project Structure
This project implements an image classification model using PyTorch and timm.
- `data/`: Contains train and test data
- `eda/` : Code related to EDA
  - `augmentation_viewer.py`: Streamlit code for EDA  
(Augmentation visualization,  
Grad-CAM visualization,  
Misclassification image visualization)
  - `class_mapping.pkl`: PKL file for mapping class label onto text
  - `imagenet_idx2class.txt`: Txt file with class by train data folder
- `src/`: Source code for the project
  - `dataset.py`: CustomDataset class
  - `loss.py`: Loss function
  - `models.py`: Model architecture
  - `sketch_transforms.py`: Customized Data transformations for sketch data
  - `trainer.py`: Training loop
  - `transforms.py`: Basic Data transformations
- `utils/`: Utility functions
- `configs/`: Configuration files
- `scripts/`: Project execution files
  - `diffusionImage.py`: Script to make augmented images using diffusion model
  - `inference.py`: Script to run inference
  - `moveFiles.py`: Script to move augmented images 
  - `train.py`: Script to train the model
  

## Usage

1. Prepare your data in the `data/` directory.
2. Adjust the configuration in `configs/config.json` if needed.
3. Run training:
   ```
   python scripts/train.py
   ```
4. Run inference:
   ```
   python scripts/inference.py
   ```

## Requirements

- pandas==2.1.4
- matplotlib==3.8.4
- seaborn==0.13.2
- Pillow==10.3.0
- numpy==1.26.3
- timm==0.9.16
- albumentations==1.4.4
- tqdm==4.66.1
- scikit-learn==1.4.2
- opencv-python==4.9.0.80

