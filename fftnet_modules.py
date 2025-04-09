import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadSpectralAttention(nn.Module):
    """
    다중 헤드 스펙트럴 어텐션 모듈.
    주파수 도메인에서 어텐션을 계산하는 메커니즘을 구현합니다.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0., adaptive_spectral=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.adaptive_spectral = adaptive_spectral

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.spectral_proj = nn.Linear(dim, dim)
        
        if adaptive_spectral:
            # 적응형 주파수 필터링을 위한 파라미터
            self.freq_filter = nn.Parameter(torch.ones(1, num_heads, 1, 1))
            
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape  # 배치 크기, 시퀀스 길이, 채널 수
        
        # 표준 QKV 프로젝션
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C//H)
        
        # 주파수 도메인 변환
        # FFT 연산을 위해 실수 부분만 사용하도록 조정
        k_fft = torch.fft.rfft(k, dim=2, norm="ortho")
        
        if self.adaptive_spectral:
            # 적응형 주파수 필터링 적용
            k_fft = k_fft * self.freq_filter
        
        # 역 FFT로 시간 도메인으로 복원
        k_filtered = torch.fft.irfft(k_fft, n=N, dim=2, norm="ortho")
        
        # 어텐션 행렬 계산
        attn = (q @ k_filtered.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 어텐션 적용 및 출력 프로젝션
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerEncoderBlock(nn.Module):
    """
    FFTNet 트랜스포머 인코더 블록.
    주파수 도메인 어텐션과 표준 MLP 레이어를 결합합니다.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adaptive_spectral=True):
        super().__init__()
        
        # 첫 번째 정규화 레이어와 주파수 어텐션
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSpectralAttention(
            dim=dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            dropout=dropout,
            adaptive_spectral=adaptive_spectral
        )
        
        # 두 번째 정규화 레이어와 MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            dropout=dropout
        )
        
    def forward(self, x):
        # 잔차 연결과 함께 어텐션 적용
        x = x + self.attn(self.norm1(x))
        # 잔차 연결과 함께 MLP 적용
        x = x + self.mlp(self.norm2(x))
        return x

class MLP(nn.Module):
    """
    트랜스포머 블록에서 사용되는 다층 퍼셉트론입니다.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x