import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)
 
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        smoothed_labels = one_hot * self.confidence + self.smoothing / self.num_classes
        loss = -torch.sum(smoothed_labels * F.log_softmax(outputs, dim=1), dim=1)
        return loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output1, output2, targets):
        # Calculate the Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Contrastive loss formula
        loss = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss + self.loss_fn(output1, targets)
