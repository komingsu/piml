import numpy as np
import torch
import torch.nn as nn

class MultiTaskWeightedLoss(torch.nn.Module):
    """
    Multi Scale strategy에서 각 스케일에 대해 weight를 자동으로 설정
    관련된 논문은 "https://arxiv.org/pdf/1805.06334.pdf"에서 참고할 수 있다.
    
    Inputs:
        multi-scale의 개수
        
    Outputs:
        scalar값
    """
    def __init__( 
        self,
        num_losses, # num_losses (int): The number of multi-task losses
    )
    super(MultiTaskWeightedLoss, self).__init__()
    if num_losses <= 0:
        raise ValueError("the value of num_losses should be positive, but got {}".format(num_losses))
    self.num_losses = num_losses
    self.params = nn.parameter.Parameter(torch.ones(3))
    
    def forward(self,losses):
        loss_sum = 0
        params = torch.pow(self.params, 2)
        for i in range(self.num_losses):
            weighted_loss = 0.5 * torch.div(losses[i], params[i]) + torch.log1p(params[i])
            loss_sum += weighted_loss
        return loss_sum
    
def l1_reg(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss