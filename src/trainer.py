import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model # 훈련할 모델
        self.device = device # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader # 훈련 데이터 로더
        self.val_loader = val_loader # 검증 데이터 로더
        self.optimizer = optimizer # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn # 손실 함수
        self.epochs = epochs # 총 훈련 에폭 수
        self.result_path = result_path # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch+1}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch+1} epoch result. Loss = {loss:.4f}")

    def train_epoch(self, acculation_steps=4) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        scaler = GradScaler() # AMP(Automatic Mixed Precision) 사용

        self.optimizer.zero_grad() # Gradient를 0으로 초기화

        for i, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # forward 단계에서 Mixed Precision을 사용하여 연산 속도 향상
            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            # 여러 배치에서 계산된 Gradient를 누적하기 전 손실 값을 정규화
            loss = loss / acculation_steps
            scaler.scale(loss).backward()
            total_loss += loss.item() # 손실 값을 누적

            # 훈련 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # 누적된 Gradient를 사용하여 모델 파라미터 업데이트 - Gradient Accumulation 적용
            if (i+1) % acculation_steps == 0 or (i+1) == len(self.train_loader):
                scaler.step(self.optimizer)
                scaler.update() # Scaling Factor를 업데이트
                self.optimizer.zero_grad() # Gradient를 0으로 초기화    
                self.scheduler.step()
                
            progress_bar.set_postfix(loss=loss.item())

        train_accuracy = correct_predictions / total_samples
        return total_loss / len(self.train_loader), train_accuracy

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                # validation loss
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # validation acc
                _, predicted = torch.max(outputs, 1)  # 예측된 클래스
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                progress_bar.set_postfix(loss=loss.item(), correct_predictions=correct_predictions, total_samples=total_samples)
                
        accuracy = correct_predictions / total_samples
        return total_loss / len(self.val_loader), accuracy

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()  # 훈련 정확도 포함
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},\n\t Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()
