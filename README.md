# FFTNetViT-UNet for Semantic Segmentation

FFTNetViT-UNet은 주파수 도메인 기반 Vision Transformer와 UNet 디코더를 결합한 의미론적 이미지 세그멘테이션 모델입니다. 이 모델은 푸리에 변환(FFT)을 활용하여 주파수 영역에서의 특징을 효과적으로 포착하고, UNet 구조를 통해 고해상도 세그멘테이션 출력을 생성합니다.

## 모델 구조

![모델 구조](https://mermaid.ink/img/pako:eNqNkk9v2zAMxb-K4FMHbEma_1hOyQY0GNYMaNceih0KWqITLbLlSXKxIsh3HyUlWRds2E4U-fP7kRRF3ShpFbOsT63-1iC2LT7_evE0S3FBB6c1QosCDxJ5RzkNLgK7uI1Tyo1u9S2xH7uO9qnTOhjIXHq0MZLT3QnVuQC94GWEk3Y5uPZSR-6IXRwdSk88cY-iXm4bTjGH0aOvg6PYgfDOJt-TjH9mFR8CPzG7oj2_mAOhWFy7vCHXnCjCXdkaNXSoZ1Lk8LdKfsFv29vvpxGGPxwnw3ZMsS_hQJGbxuUfMzuaZxXs0qnXYTqW1TDNd6c8lEcTaRdlR3y79yTMG9gvX0nRbHw0nv4--vk8k38Jz3gVP3hUWMHDCsR1uy0ceCQ-GBuSw2yeFbf61FZnpgQaPMqyh4FLlrlG1_R-fxmC8bQeWOGllV3dX66XT-uSZY0IQZrwR9ywznTwRDzLmm5w2lqPdxqfmDU9QWudTEVbM1t9_gEPDcrQ?type=png)

### 주요 컴포넌트

1. **FFTNetViT 인코더**:
   - 이미지를 패치로 분할하고 임베딩
   - 클래스 토큰과 위치 임베딩 적용
   - 주파수 도메인 기반 멀티헤드 어텐션 적용
   - 적응형 주파수 필터링을 통한 특징 증강

2. **UNet 디코더**:
   - 점진적인 업샘플링을 통한 특징 맵 확장
   - 고해상도 세그멘테이션 마스크 생성
   - 잔차 연결을 통한 정보 보존

3. **하이브리드 손실 함수**:
   - BCE(Binary Cross Entropy) 손실과 Dice 계수 손실의 조합
   - 클래스 불균형 문제 완화

## 주요 특징

- **주파수 도메인 처리**: 푸리에 변환을 활용하여 공간 도메인과 주파수 도메인의 정보를 모두 활용
- **적응형 주파수 필터링**: 학습 가능한 파라미터로 중요한 주파수 대역을 강조
- **고해상도 출력**: UNet 아키텍처를 통한 섬세한 경계 보존
- **효율적인 학습**: 학습률 스케줄러와 데이터 증강을 통한 성능 최적화

## 데이터셋

모델은 [Oxford-IIIT Pet 데이터셋](https://www.robots.ox.ac.uk/~vgg/data/pets/)에서 훈련되었습니다. 이 데이터셋은:
- 37개 품종의 고양이와 개 이미지 데이터셋
- 각 품종별 약 200개의 이미지
- 픽셀 단위 세그멘테이션 어노테이션 포함

## 구현 및 사용 방법

### 필요 패키지

```
torch
torchvision
numpy
matplotlib
pillow
tqdm
einops
```

### 학습 실행

```bash
python train_FFTNetViT_Unet_seg.py
```

학습 시 데이터셋이 자동으로 다운로드되며, 다음과 같은 디렉토리 구조가 생성됩니다:
```
├── data/              # 데이터셋 저장 디렉토리
│   └── oxford-iiit-pet/
├── models/            # 훈련된 모델 저장 디렉토리 
│   ├── fftnet_unet_pet_best.pth
│   └── fftnet_unet_pet_final.pth
└── results/           # 시각화 결과 및 평가 메트릭 저장
    ├── segmentation_predictions.png
    └── fftnet_unet_pet_training_curves.png
```

### 추론 GUI 실행

```bash
python inference_gui.py
```

GUI 인터페이스를 통해 새로운 이미지에 대한 세그멘테이션을 수행할 수 있습니다.

## 성능 평가

모델은 Oxford-IIIT Pet 데이터셋에서 다음과 같은 성능을 보여줍니다:

- **IoU (Intersection over Union)**: ~0.85
- **Dice 계수**: ~0.91

### 추론 결과 예시

다음은 FFTNetViT-UNet 모델의 세그멘테이션 결과 예시입니다:

![추론 결과 예시](ScrShot%206.png)

## 참고 문헌

- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

## 라이센스

MIT License

## 연락처

- 개발자: Gromit Park
- GitHub: [@bemoregt](https://github.com/bemoregt)
