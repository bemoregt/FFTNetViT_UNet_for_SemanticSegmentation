import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile
import shutil
from PIL import Image
import random

# FFTNet 관련 모듈 임포트
from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock
from fftnet_vit import FFTNetViT

# 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 학습 파라미터 설정
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-4
image_size = 384  # 세그멘테이션을 위해 더 큰 이미지 크기

# 장치 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# FFTNetViT 인코더와 UNet 디코더를 결합한 세그멘테이션 모델 정의
class FFTNetViTUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, num_classes=1,
                 embed_dim=512, depth=8, mlp_ratio=4.0, dropout=0.1, num_heads=8, 
                 adaptive_spectral=True):
        super(FFTNetViTUNet, self).__init__()
        
        # FFTNetViT 인코더
        self.encoder = FFTNetViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_heads=num_heads,
            adaptive_spectral=adaptive_spectral,
            return_intermediate=True  # 중간 특징 맵을 반환하도록 수정 필요
        )
        
        # 패치 수 계산
        self.num_patches = (img_size // patch_size) ** 2
        self.latent_size = img_size // patch_size
        
        # 디코더 레이어
        self.decoder_channels = [embed_dim, 256, 128, 64, 32]
        
        # 디코더 업샘플링 레이어
        self.up_blocks = nn.ModuleList()
        for i in range(len(self.decoder_channels) - 1):
            self.up_blocks.append(
                UNetUpBlock(
                    in_channels=self.decoder_channels[i],
                    out_channels=self.decoder_channels[i+1],
                    scale_factor=2
                )
            )
        
        # 최종 출력 레이어
        self.final_conv = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
        
    def forward(self, x):
        # 인코더 통과 (중간 특징 맵 획득)
        features = self.encoder(x)
        
        # 마지막 특징 맵 추출
        # 1차원 출력(B, num_patches + 1, embed_dim)을 2D로 재구성
        # 첫 번째 토큰(CLS 토큰)은 제외
        batch_size = x.shape[0]
        latent = features[-1][:, 1:, :]  # CLS 토큰 제외
        latent = latent.reshape(batch_size, self.latent_size, self.latent_size, -1)
        latent = latent.permute(0, 3, 1, 2)  # B, C, H, W 형태로 변환
        
        # 디코더 통과
        x = latent
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # 최종 출력 레이어 통과
        out = self.final_conv(x)
        
        return out

# UNet 스타일 업샘플링 블록
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# Oxford-IIIT Pet Dataset 다운로드 및 준비
def download_pet_dataset():
    dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    dataset_path = "data/oxford-iiit-pet"
    os.makedirs(dataset_path, exist_ok=True)
    
    # 이미지 다운로드 및 압축 해제
    images_tar_path = os.path.join(dataset_path, "images.tar.gz")
    if not os.path.exists(images_tar_path):
        print("Downloading Oxford-IIIT Pet Dataset images...")
        response = requests.get(dataset_url, stream=True)
        with open(images_tar_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print("Extracting images...")
        import tarfile
        with tarfile.open(images_tar_path, "r:gz") as tar:
            tar.extractall(dataset_path)
    
    # 어노테이션 다운로드 및 압축 해제
    annotations_tar_path = os.path.join(dataset_path, "annotations.tar.gz")
    if not os.path.exists(annotations_tar_path):
        print("Downloading Oxford-IIIT Pet Dataset annotations...")
        response = requests.get(annotations_url, stream=True)
        with open(annotations_tar_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print("Extracting annotations...")
        import tarfile
        with tarfile.open(annotations_tar_path, "r:gz") as tar:
            tar.extractall(dataset_path)
    
    print("Oxford-IIIT Pet Dataset ready.")
    return dataset_path

# 세그멘테이션 데이터셋 클래스
class PetSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set="trainval", transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        
        # 이미지 및 마스크 파일 목록 로드
        with open(os.path.join(root_dir, "annotations", f"{image_set}.txt"), "r") as f:
            self.file_list = [line.strip().split()[0] for line in f.readlines()]
        
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        
        # 이미지 및 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # 데이터 변환 적용
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # 마스크를 3개 클래스(배경, 경계, 전경)에서 1개 채널(전경/배경)으로 변환
        mask = (mask > 1).float()  # 1은 배경, 2는 경계, 3은 전경
        
        return image, mask

# 데이터 증강 및 전처리
class SegmentationTransforms:
    def __init__(self, img_size=256):
        self.img_size = img_size
    
    def __call__(self, image, mask):
        # 리사이징
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # 무작위 수평 뒤집기
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 무작위 회전
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # 색상 지터링 (Color Jittering)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)
        
        # 가우시안 블러
        if random.random() > 0.7:
            from PIL import ImageFilter
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        
        # 무작위 확대/축소
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = int(self.img_size * scale)
            image = image.resize((new_size, new_size), Image.BILINEAR)
            mask = mask.resize((new_size, new_size), Image.NEAREST)
            
            # 중앙 크롭으로 원래 크기로 복원
            if new_size > self.img_size:
                left = (new_size - self.img_size) // 2
                top = (new_size - self.img_size) // 2
                right = left + self.img_size
                bottom = top + self.img_size
                image = image.crop((left, top, right, bottom))
                mask = mask.crop((left, top, right, bottom))
            else:
                # 패딩으로 원래 크기로 복원
                padded_image = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
                padded_mask = Image.new("L", (self.img_size, self.img_size), 0)
                paste_x = (self.img_size - new_size) // 2
                paste_y = (self.img_size - new_size) // 2
                padded_image.paste(image, (paste_x, paste_y))
                padded_mask.paste(mask, (paste_x, paste_y))
                image = padded_image
                mask = padded_mask
        
        # 텐서로 변환
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)
        
        # 정규화
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask

# Dice 계수 손실 함수
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = targets.size(0)
        
        # 차원 통합
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Dice 계수 계산
        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# 훈련 함수
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    train_loader_tqdm = tqdm(train_loader, desc="Training")
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item() * images.size(0)
        
        # tqdm 진행 상태 업데이트
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    # 에폭 평균 손실 계산
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return epoch_loss

# IoU 계산 함수
def calculate_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# 검증 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    ious = []
    
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        for images, masks in val_loader_tqdm:
            images, masks = images.to(device), masks.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # IoU 계산
            batch_iou = calculate_iou(outputs, masks)
            ious.extend(batch_iou.cpu().numpy())
            
            # 통계 업데이트
            running_loss += loss.item() * images.size(0)
            
            # tqdm 진행 상태 업데이트
            val_loader_tqdm.set_postfix(loss=loss.item())
    
    # 에폭 평균 손실 및 IoU 계산
    epoch_loss = running_loss / len(val_loader.dataset)
    mean_iou = np.mean(ious)
    
    return epoch_loss, mean_iou

# 예측 시각화 함수
def visualize_predictions(model, val_loader, device, num_samples=4):
    model.eval()
    images, masks, preds = [], [], []
    
    with torch.no_grad():
        for img, mask in val_loader:
            if len(images) >= num_samples:
                break
            
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred = torch.sigmoid(pred) > 0.5
            
            # 시각화를 위해 CPU로 이동 및 변환
            images.extend(img.cpu())
            masks.extend(mask.cpu())
            preds.extend(pred.cpu())
            
            if len(images) >= num_samples:
                break
    
    # 결과 시각화
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # 이미지 표시 (정규화 해제)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # 원본 이미지
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        
        # 실제 마스크
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # 예측 마스크
        axes[i, 2].imshow(preds[i].squeeze().numpy(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig("results/segmentation_predictions.png")
    plt.close()

# 메인 함수
def main():
    # Oxford-IIIT Pet Dataset 다운로드 및 준비
    dataset_path = download_pet_dataset()
    
    # 데이터셋 생성
    transform = SegmentationTransforms(img_size=image_size)
    
    train_dataset = PetSegmentationDataset(
        root_dir=dataset_path,
        image_set="trainval",
        transform=transform
    )
    
    # 데이터셋을 훈련과 검증으로 분할 (80:20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 모델 초기화
    model = FFTNetViTUNet(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        num_classes=1,  # 전경/배경 분할
        embed_dim=256,
        depth=16,
        mlp_ratio=4.0,
        dropout=0.1,
        num_heads=16,
        adaptive_spectral=True
    )
    
    # 모델을 선택한 장치로 이동
    model = model.to(device)
    
    # 손실 함수 정의
    criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    
    def combined_loss(outputs, targets):
        bce_loss = criterion(outputs, targets.float())
        dc_loss = dice_loss(outputs, targets.float())
        return 0.5 * bce_loss + 0.5 * dc_loss
    
    # 옵티마이저 정의
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 및 검증 손실, IoU를 저장할 리스트
    train_losses = []
    val_losses = []
    val_ious = []
    
    # 최고 검증 IoU와 해당 에폭 저장
    best_val_iou = 0.0
    best_epoch = 0
    
    # 학습 루프
    print("Starting training for FFTNet-UNet on Oxford-IIIT Pet Dataset...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, combined_loss, optimizer, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss, val_iou = validate(model, val_loader, combined_loss, device)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # 학습률 업데이트
        scheduler.step(val_loss)
        
        # 결과 출력
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # 최고 성능 모델 저장
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/fftnet_unet_pet_best.pth')
            print(f"New best model saved with IoU: {val_iou:.4f}")
        
        # 현재 에폭 모델 저장 (주기적으로 저장하기 위해 10 에폭마다)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/fftnet_unet_pet_epoch_{epoch+1}.pth')
            # 현재 예측 결과 시각화
            visualize_predictions(model, val_loader, device)
            
        # 학습 지표를 파일에 저장
        with open('results/fftnet_unet_pet_metrics.csv', 'a') as f:
            if epoch == 0:
                f.write('Epoch,Train Loss,Val Loss,Val IoU\n')
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_iou:.4f}\n")

    # 최종 모델 저장
    torch.save(model.state_dict(), 'models/fftnet_unet_pet_final.pth')
    print(f"Training completed. Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    # 최종 예측 결과 시각화
    visualize_predictions(model, val_loader, device, num_samples=8)

    # 학습 결과 시각화
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # IoU 그래프
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/fftnet_unet_pet_training_curves.png')
    plt.show()

# 필요한 FFTNetViT 수정 - 중간 특징 맵 반환을 위한 커스텀 버전
class CustomFFTNetViT(FFTNetViT):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
    
    def forward(self, x):
        # 기존 FFTNetViT의 forward 메소드를 중간 특징 맵을 반환하도록 수정
        # 이 부분은 실제 FFTNetViT 구현에 따라 조정 필요
        
        if not self.return_intermediate:
            return super().forward(x)
        
        # 중간 특징 맵 수집 로직 구현 (예시)
        features = []
        
        # 패치 임베딩
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        # 트랜스포머 블록 통과
        for blk in self.blocks:
            x = blk(x)
            features.append(x)
        
        return features

# FFTNetViT 모듈 업데이트
FFTNetViT = CustomFFTNetViT

# 메인 실행 부분
if __name__ == '__main__':
    main()