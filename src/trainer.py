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
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
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
        scaler = GradScaler()

        self.optimizer.zero_grad() 

        for i, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            loss = loss / acculation_steps 
            scaler.scale(loss).backward()
            total_loss += loss.item() 

            # 훈련 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if (i+1) % acculation_steps == 0 or (i+1) == len(self.train_loader):
                scaler.step(self.optimizer)
                scaler.update() 
                self.optimizer.zero_grad()
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
