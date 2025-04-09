# inference_gui.py
import os
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, Button, Label, Frame
from PIL import Image, ImageTk
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 모델 정의 및 관련 모듈 임포트
from train_FFTNetViT_Unet_seg import FFTNetViTUNet, CustomFFTNetViT

# CustomFFTNetViT를 FFTNetViT로 덮어쓰는 부분이 필요하면 추가
FFTNetViT = CustomFFTNetViT

class SegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FFTNet-UNet 세그멘테이션 데모")
        self.root.geometry("1200x700")
        
        # 모델 로드
        self.load_model()
        
        # GUI 레이아웃 설정
        self.setup_gui()
        
        # 초기 이미지
        self.current_image = None
        self.current_segmentation = None
        
    def load_model(self):
        # 장치 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # 모델 초기화
        self.model = FFTNetViTUNet(
            img_size=384,
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
        
        # 저장된 모델 가중치 로드
        model_path = 'models/fftnet_unet_pet_best.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"모델 로딩 성공: {model_path}")
        else:
            print(f"경고: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def setup_gui(self):
        # 상단 프레임 - 버튼
        self.top_frame = Frame(self.root)
        self.top_frame.pack(pady=10)
        
        self.load_btn = Button(self.top_frame, text="이미지 로드", command=self.load_image, width=15, height=2)
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        self.segment_btn = Button(self.top_frame, text="세그멘테이션 실행", command=self.segment_image, width=15, height=2)
        self.segment_btn.pack(side=tk.LEFT, padx=10)
        
        # 중앙 프레임 - 이미지 표시
        self.image_frame = Frame(self.root)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 원본 이미지 표시 영역
        self.original_frame = Frame(self.image_frame, width=550, height=550, bd=2, relief=tk.SUNKEN)
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.original_frame.pack_propagate(0)
        
        self.original_label = Label(self.original_frame, text="원본 이미지")
        self.original_label.pack(pady=5)
        
        self.original_image_label = Label(self.original_frame)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        # 세그멘테이션 결과 표시 영역
        self.segmented_frame = Frame(self.image_frame, width=550, height=550, bd=2, relief=tk.SUNKEN)
        self.segmented_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.segmented_frame.pack_propagate(0)
        
        self.segmented_label = Label(self.segmented_frame, text="세그멘테이션 결과")
        self.segmented_label.pack(pady=5)
        
        self.segmented_image_label = Label(self.segmented_frame)
        self.segmented_image_label.pack(fill=tk.BOTH, expand=True)
        
        # 상태 메시지 표시
        self.status_label = Label(self.root, text="시작하려면 이미지를 로드하세요.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        # 이미지 파일 선택
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            # 이미지 로드 및 표시
            self.current_image = Image.open(file_path).convert("RGB")
            self.display_image(self.current_image, self.original_image_label)
            self.status_label.config(text=f"이미지 로드됨: {os.path.basename(file_path)}")
            
            # 세그멘테이션 버튼 활성화
            self.segment_btn.config(state=tk.NORMAL)
            
            # 세그멘테이션 결과 초기화
            self.segmented_image_label.config(image="")
            self.current_segmentation = None
    
    def segment_image(self):
        if self.current_image is None:
            self.status_label.config(text="이미지를 먼저 로드하세요.")
            return
        
        self.status_label.config(text="세그멘테이션 처리 중...")
        self.root.update()
        
        try:
            # 이미지 전처리
            img = self.preprocess_image(self.current_image)
            
            # 추론
            with torch.no_grad():
                output = self.model(img)
                mask = torch.sigmoid(output) > 0.5
            
            # 결과 시각화
            segmentation_result = self.visualize_result(img, mask)
            self.current_segmentation = segmentation_result
            
            # 결과 표시
            self.display_image(segmentation_result, self.segmented_image_label)
            self.status_label.config(text="세그멘테이션 완료")
            
        except Exception as e:
            self.status_label.config(text=f"오류 발생: {str(e)}")
            print(f"세그멘테이션 오류: {str(e)}")
    
    def preprocess_image(self, image):
        # img_size=384로 설정되어 있으므로, 동일한 크기로 리사이즈
        img_resized = image.resize((384, 384), Image.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)
        img_normalized = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_batch = img_normalized.unsqueeze(0).to(self.device)
        return img_batch
    
    def visualize_result(self, image_tensor, mask_tensor):
        # 원본 이미지와 세그멘테이션 마스크 시각화
        img = image_tensor.squeeze().cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        mask = mask_tensor.squeeze().cpu().numpy()
        
        # Matplotlib 그림 생성
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # 원본 이미지
        ax[0].imshow(img)
        ax[0].set_title("원본 이미지")
        ax[0].axis("off")
        
        # 마스크 오버레이
        ax[1].imshow(img)
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [1, 0, 0, 0.5]  # 빨간색 반투명 마스크
        ax[1].imshow(mask_rgba)
        ax[1].set_title("세그멘테이션 결과")
        ax[1].axis("off")
        
        fig.tight_layout()
        
        # Matplotlib 그림을 PIL 이미지로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close(fig)
        
        result_image = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
        return result_image
    
    def display_image(self, image, label_widget):
        # 이미지를 라벨 위젯에 표시
        if image is None:
            return
        
        # 이미지 크기 조정 (라벨 크기에 맞게)
        width = label_widget.winfo_width() or 500
        height = label_widget.winfo_height() or 500
        
        # 비율 유지하며 리사이징
        img_ratio = image.width / image.height
        if width / height > img_ratio:
            new_width = int(height * img_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / img_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # PIL 이미지를 Tkinter PhotoImage로 변환
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # 이미지 표시 및 참조 유지(GC 방지)
        label_widget.config(image=tk_image)
        label_widget.image = tk_image

# 메인 실행 함수
def main():
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()