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
