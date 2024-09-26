import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import numpy as np
import random
import optuna  # Optuna 임포트 추가

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.sketch_transforms import SketchTransformSelector
from src.Uni_Aug_transforms import UniformTransformSelector
from src.models import ModelSelector
from src.loss import Loss
from src.trainer import Trainer

# Random seed 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 시드 고정
set_seed(42)

# 설정 파일 로드
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def objective(trial):
    # 하이퍼파라미터 정의
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)  # 0.00001 ~ 0.1
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)  # 0.00001 ~ 0.01

    # 설정 파일 로드
    config = load_config('./configs/config.json')

    # 학습에 사용할 장비를 선택.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_info = pd.read_csv(config['train_csv'])

    # 총 class의 수를 측정.
    num_classes = len(train_info['target'].unique())

    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    train_df, val_df = train_test_split(
        train_info,
        test_size=0.2,
        stratify=train_info['target']
    )

    # 학습에 사용할 Transform을 선언.
    transform_selector = SketchTransformSelector(
        transform_type=config['transform']
    )
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # 학습에 사용할 Dataset을 선언.
    train_dataset = CustomDataset(
        root_dir=config['traindata_dir'],
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=config['traindata_dir'],
        info_df=val_df,
        transform=val_transform
    )

    # 학습에 사용할 DataLoader를 선언.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=8
    )

    # 학습에 사용할 Model을 선언.
    model_selector = ModelSelector(
        model_type=config['model']['type'],
        num_classes=num_classes,
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained']
    )
    model = model_selector.get_model()

    # 선언된 모델을 학습에 사용할 장비로 셋팅.
    model.to(device)

    # 학습에 사용할 optimizer를 선언하고, learning rate를 지정
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,  # Optuna에서 튜닝한 learning rate 사용
        weight_decay=weight_decay  # Optuna에서 튜닝한 weight decay 사용
    )

    # 스케줄러 초기화
    scheduler_step_size = 30  # 매 30step마다 학습률 감소
    scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소

    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)

    # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
    epochs_per_lr_decay = 2
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)

    # 학습에 사용할 Loss를 선언.
    loss_fn = Loss()

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=config['epochs'],
        result_path=config['result_path']
    )

    # 모델 학습.
    trainer.train()

    # 검증 정확도 반환 (최적화 목표)
    loss, val_accuracy = trainer.validate()  # validate 메서드가 검증 손실과 정확도를 반환한다고 가정

    return loss  # 정확도를 반환

def main():
    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize')  # 정확도 최대화
    study.optimize(objective, n_trials=10)  # 3번의 시도

    # 최적의 하이퍼파라미터 출력
    print("Best hyperparameters: ", study.best_params)

if __name__ == "__main__":
    main()