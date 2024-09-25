import torch
import numpy as np
import albumentations as A
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from albumentations.pytorch import ToTensorV2


class SketchTorchvisionTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 0-1 스케일링
        common_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Min-Max 정규화
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.7),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.AugMix(
                        severity=3,
                        mixture_width=2,
                        chain_depth=-1,
                        alpha=0.5,
                        all_ops=False,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=255
                    ),
                    # transforms.RandomRotation(15),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
            # A.LongestMaxSize(max_size=224),  # 가장 긴 변의 크기를 224로 조정
            # A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(255, 255, 255)),  # 필요한 경우 패딩 추가
            # A.CenterCrop(height=224, width=224),  # 중앙에서 224x224 크기로 자르기
            A.Resize(224, 224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Min-Max 정규화
            ToTensorV2(),
        ]
        
        if is_train:
            # 훈련용 변환
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.7),
                    A.VerticalFlip(p=0.3),
                    # A.Rotate(limit=5, p=0.5),
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),
                    # A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
                    # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
                    # A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    # A.Morphological(p=0.5, scale=(2, 3), operation='dilation'),
                    # A.Morphological(p=0.5, scale=(2, 3), operation='erosion'),
                    # A.Perspective(scale=(0.05, 0.1), p=0.5),
                    # A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    # A.CoarseDropout(max_holes=4, max_height=10, max_width=10, fill_value=255, p=0.5),
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
        
        return transformed['image']

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