import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import random
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np

# UniformAug 구현을 위한 함수 정의
def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Invert(img, _):
    return ImageOps.invert(img)

def Rotate(img, magnitude):
    return img.rotate(magnitude)

def Posterize(img, magnitude):
    return ImageOps.posterize(img, magnitude)

def Solarize(img, magnitude):
    return ImageOps.solarize(img, magnitude)

def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np[img_np < threshold] = 255 - img_np[img_np < threshold]
    img_np = img_np.astype(np.uint8)
    return Image.fromarray(img_np)

def Color(img, magnitude):
    return ImageEnhance.Color(img).enhance(magnitude)

def Contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(magnitude)

def Brightness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(magnitude)

def Sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(magnitude)

def ShearX(img, magnitude):
    return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))

def ShearY(img, magnitude):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))

def CutoutAbs(img, size):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - size / 2.))
    y0 = int(max(0, y0 - size / 2.))
    x1 = int(min(w, x0 + size))
    y1 = int(min(h, y0 + size))
    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def TranslateXabs(img, pixels):
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))

def TranslateYabs(img, pixels):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))

def TranslateX(img, magnitude):
    return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0))

def TranslateY(img, magnitude):  # self 제거
    # numpy 배열을 PIL 이미지로 변환
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]))

def Posterize2(img, magnitude):
    return ImageOps.posterize(img, int(magnitude))

def augment_list(for_autoaug=True):
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),
            (Posterize2, 0, 4),
            (TranslateXabs, 0., 45),
            (TranslateYabs, 0., 45),
        ]
    else:
        l += [
            (TranslateX, 0., 0.33),
            (TranslateY, 0., 0.33),
        ]

    return l

# UniformAugment 클래스 정의
class UniformAugment:
    def __init__(self, ops_num=2):
        self._augment_list = augment_list(for_autoaug=False)
        self._ops_num = ops_num
    
    def __call__(self, img):
        ops = random.choices(self._augment_list, k=self._ops_num)
        for op in ops:
            augment_fn, low, high = op
            probability = random.random()
            if random.random() < probability:
                img = augment_fn(img.copy(), random.uniform(low, high))
        return img



class UniformTorchvisionTransform:
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
                [   UniformAugment(ops_num=2),  # UniformAugment를 초기화
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

class UniformAlbumentationsTransform:
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
                [   UniformAugment(),
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

class UniformTransformSelector:
    def __init__(self, transform_type: str):
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'torchvision':
            return UniformTorchvisionTransform(is_train=is_train)
        elif self.transform_type == 'albumentations':
            return UniformAlbumentationsTransform(is_train=is_train)