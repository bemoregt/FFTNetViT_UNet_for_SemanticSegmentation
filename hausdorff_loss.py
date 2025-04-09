import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt

# 경계 감지 함수
def get_boundary(mask, kernel_size=3):
    """
    마스크에서 경계를 추출하는 함수
    """
    dilated = ndimage.binary_dilation(mask, structure=np.ones((kernel_size, kernel_size)))
    eroded = ndimage.binary_erosion(mask, structure=np.ones((kernel_size, kernel_size)))
    return dilated & (~eroded)

# 하우스도르프 거리 손실 함수 구현
class HausdorffDTLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, logits, targets):
        """
        logits: 모델의 예측 출력 [B, 1, H, W]
        targets: 정답 마스크 [B, 1, H, W]
        """
        # 시그모이드 함수를 통한 확률 값으로 변환
        probs = torch.sigmoid(logits)
        
        # 바이너리 마스크로 변환 (임계값 0.5)
        pred_mask = (probs > 0.5).float()
        target_mask = targets.float()
        
        # 배치 크기
        batch_size = pred_mask.size(0)
        
        # 거리 변환을 위해 CPU로 이동
        pred_mask_cpu = pred_mask.detach().cpu().numpy()
        target_mask_cpu = target_mask.detach().cpu().numpy()
        
        # 거리 맵 계산을 위한 배열
        pred_dt = np.zeros_like(pred_mask_cpu)
        target_dt = np.zeros_like(target_mask_cpu)
        
        # 배치별로 거리 변환 계산
        for b in range(batch_size):
            # 예측 마스크의 경계로부터 거리 계산
            pred_dt[b, 0] = distance_transform_edt(pred_mask_cpu[b, 0] == 0) * (target_mask_cpu[b, 0] != 0).astype(np.float32)
            pred_dt[b, 0] += distance_transform_edt(pred_mask_cpu[b, 0] != 0) * (target_mask_cpu[b, 0] == 0).astype(np.float32)
            
            # 정답 마스크의 경계로부터 거리 계산
            target_dt[b, 0] = distance_transform_edt(target_mask_cpu[b, 0] == 0) * (pred_mask_cpu[b, 0] != 0).astype(np.float32)
            target_dt[b, 0] += distance_transform_edt(target_mask_cpu[b, 0] != 0) * (pred_mask_cpu[b, 0] == 0).astype(np.float32)
        
        # 텐서로 다시 변환하여 장치로 이동
        pred_dt = torch.from_numpy(pred_dt).to(logits.device)
        target_dt = torch.from_numpy(target_dt).to(logits.device)
        
        # 하우스도르프 거리 계산
        pred_loss = (pred_dt ** self.alpha).mean()
        target_loss = (target_dt ** self.alpha).mean()
        
        # 양방향 하우스도르프 거리 평균
        hausdorff_loss = (pred_loss + target_loss) / 2.0
        
        return hausdorff_loss

# 경계 중심 하우스도르프 손실 함수 (계산 효율성 향상)
class BoundaryHausdorffLoss(nn.Module):
    def __init__(self, alpha=2.0, kernel_size=3):
        super(BoundaryHausdorffLoss, self).__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        
    def forward(self, logits, targets):
        # 시그모이드 함수를 통한 확률 값으로 변환
        probs = torch.sigmoid(logits)
        
        # 바이너리 마스크로 변환 (임계값 0.5)
        pred_mask = (probs > 0.5).float()
        target_mask = targets.float()
        
        # 배치 크기
        batch_size = pred_mask.size(0)
        
        # 경계만 추출하기 위해 CPU로 이동
        pred_mask_cpu = pred_mask.detach().cpu().numpy()
        target_mask_cpu = target_mask.detach().cpu().numpy()
        
        pred_boundary = np.zeros_like(pred_mask_cpu)
        target_boundary = np.zeros_like(target_mask_cpu)
        pred_dt = np.zeros_like(pred_mask_cpu)
        target_dt = np.zeros_like(target_mask_cpu)
        
        for b in range(batch_size):
            # 경계 추출
            pred_boundary[b, 0] = get_boundary(pred_mask_cpu[b, 0], self.kernel_size)
            target_boundary[b, 0] = get_boundary(target_mask_cpu[b, 0], self.kernel_size)
            
            # 경계로부터의 거리 계산
            pred_dt[b, 0] = distance_transform_edt(~pred_boundary[b, 0]) * target_boundary[b, 0]
            target_dt[b, 0] = distance_transform_edt(~target_boundary[b, 0]) * pred_boundary[b, 0]
        
        # 경계가 없는 경우 처리
        pred_dt = np.where(pred_dt == 0, 0, pred_dt)
        target_dt = np.where(target_dt == 0, 0, target_dt)
        
        # 텐서로 변환
        pred_dt = torch.from_numpy(pred_dt).to(logits.device)
        target_dt = torch.from_numpy(target_dt).to(logits.device)
        
        # 평균 하우스도르프 거리 계산
        pred_boundary_sum = torch.sum(torch.from_numpy(target_boundary).to(logits.device)) + 1e-6
        target_boundary_sum = torch.sum(torch.from_numpy(pred_boundary).to(logits.device)) + 1e-6
        
        pred_loss = torch.sum(pred_dt ** self.alpha) / pred_boundary_sum
        target_loss = torch.sum(target_dt ** self.alpha) / target_boundary_sum
        
        hausdorff_loss = (pred_loss + target_loss) / 2.0
        
        return hausdorff_loss
