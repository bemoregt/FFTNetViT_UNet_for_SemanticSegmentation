import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fftnet_modules import TransformerEncoderBlock
import numpy as np
import math

class PatchEmbed(nn.Module):
    """
    2D 이미지를 패치 임베딩으로 변환하는 모듈입니다.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 패치 임베딩을 위한 합성곱 레이어
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # 합성곱을 통한 패치 임베딩
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # 패치 임베딩을 1차원으로 변환
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class FFTNetViT(nn.Module):
    """
    FFTNetViT 모델: 주파수 도메인 처리를 통합한 Vision Transformer 구현.
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000,
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        dropout=0.1,
        adaptive_spectral=True,
        norm_layer=nn.LayerNorm,
        return_intermediate=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate
        
        # 패치 임베딩 레이어
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # 위치 임베딩 및 클래스 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                dropout=dropout,
                adaptive_spectral=adaptive_spectral
            )
            for i in range(depth)
        ])
        
        # 최종 분류 헤드
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 초기화
        self._init_weights()
        
    def _init_weights(self):
        # 위치 임베딩 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 모듈별 초기화
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # 이미지를 패치 임베딩으로 변환
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # 위치 임베딩 추가
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 중간 특징 맵 저장을 위한 리스트
        features = []
        
        # 트랜스포머 블록 통과
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.return_intermediate:
                features.append(x)
        
        # 최종 정규화
        x = self.norm(x)
        
        # 중간 특징 맵 반환 또는 클래스 토큰만 반환
        if self.return_intermediate:
            return features
        else:
            return x[:, 0]  # 클래스 토큰 (CLS) 반환
    
    def forward(self, x):
        # 특징 추출
        x = self.forward_features(x)
        
        # 중간 특징 맵 반환 또는 분류 수행
        if self.return_intermediate:
            return x
        else:
            return self.head(x)