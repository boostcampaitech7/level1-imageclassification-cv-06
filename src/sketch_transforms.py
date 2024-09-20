import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A

class SketchTorchvisionTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 0-1 스케일링
        common_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x)  # 이미 0-1 범위이므로 추가 스케일링 불필요
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환
        return self.transform(image)  # 설정된 변환을 적용

class SketchAlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정
        common_transforms = [
            A.Resize(224, 224),
            ToTensorV2(),
        ]
        
        if is_train:
            # 훈련용 변환
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)
        
        return transformed['image'] / 255.0  # 0-1 스케일링

class SketchTransformSelector:
    def __init__(self, transform_type: str):
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'torchvision':
            return SketchTorchvisionTransform(is_train=is_train)
        elif self.transform_type == 'albumentations':
            return SketchAlbumentationsTransform(is_train=is_train)