import torch
import torch.nn as nn
import timm
from torchvision import models


class SimpleCNN(nn.Module):
    """
    간단한 CNN 아키텍처를 정의하는 클래스.
    """
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 순전파 함수 정의
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)

class TorchvisionModel(nn.Module):
    """
    Torchvision에서 제공하는 사전 훈련된 모델을 사용하는 클래스.
    """
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool
    ):
        super(TorchvisionModel, self).__init__()
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # 모델의 최종 분류기 부분을 사용자 정의 클래스 수에 맞게 조정
        if hasattr(self.model, 'fc'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(self.model, 'classifier'):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class StackedModel(nn.Module):
    def __init__(self):
        super(StackedModel, self).__init__()
        # 각 모델을 초기화합니다.
        self.model1 = timm.create_model('convnext_large', pretrained=True)  
        self.model2 = timm.create_model('vit_base_patch16_224', pretrained=True)  
        self.model3 = timm.create_model('resnet152', pretrained=True)  

        # 각 모델의 출력 차원
        self.convnext_output_dim = 1000
        self.vit_output_dim = 1000
        self.resnet_output_dim = 1000

        # 최종 출력층
        self.fc = nn.Linear(self.convnext_output_dim + self.vit_output_dim + self.resnet_output_dim, 500)  # 예: 500 클래스 분류

    def forward(self, x):
        model1_out = self.model1(x)  # convnext 모델의 출력
        model2_out = self.model2(x)  # vit 모델의 출력
        model3_out = self.model3(x)  # resnet 모델의 출력

        # 모델 출력 연결
        out = torch.cat((model1_out, model2_out, model3_out), dim=1)
        out = self.fc(out)  # 최종 출력
        return out

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'stacked':
            self.model = StackedModel(num_classes=num_classes)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        # 생성된 모델 객체 반환
        return self.model