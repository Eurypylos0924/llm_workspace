"""MNIST 숫자 레이블을 조건으로 이미지를 생성하는 Conditional GAN 모델."""
import torch
from torch import nn

LATENT_DIM = 100
NUM_CLASSES = 10
IMAGE_DIM = 28 * 28


class ConditionalGenerator(nn.Module):
    """노이즈와 숫자 레이블을 받아 28x28 이미지를 생성한다."""
    def __init__(self):
        super().__init__()
        # 숫자 0~9를 10차원 학습 가능한 벡터로 변환한다.
        self.embedding = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        # 노이즈 100차원과 숫자 임베딩 10차원을 결합하여 입력한다.
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMAGE_DIM),
            # MNIST 학습 이미지와 동일하게 픽셀값을 [-1, 1]로 제한한다.
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 숫자 레이블을 학습 가능한 벡터로 변환한다.
        label_features = self.embedding(labels)
        # 노이즈와 숫자 정보를 이어 붙여 Generator에 입력한다.
        combined = torch.cat([noise, label_features], dim=1)
        # 결과는 (배치크기, 784) 형태이다.
        return self.model(combined)


class ConditionalDiscriminator(nn.Module):
    """이미지와 지정 숫자 레이블의 일치 여부까지 판별한다."""
    def __init__(self):
        super().__init__()
        # 숫자 레이블을 이미지 픽셀 수와 같은 784차원으로 임베딩한다.
        self.embedding = nn.Embedding(NUM_CLASSES, IMAGE_DIM)
        # 이미지 784차원과 레이블 임베딩 784차원을 함께 판별한다.
        self.model = nn.Sequential(
            nn.Linear(IMAGE_DIM * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            # BCEWithLogitsLoss를 사용하므로 Sigmoid는 추가하지 않는다.
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 숫자 레이블을 784차원 임베딩으로 변환한다.
        label_features = self.embedding(labels)
        # 실제/가짜 이미지와 레이블 특징을 결합한다.
        combined = torch.cat([images, label_features], dim=1)
        # 결과는 확률이 아니라 logit이다.
        return self.model(combined)
