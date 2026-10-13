"""실행: python -m training.train_cgan --epochs 30"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from training.model import ConditionalGenerator, ConditionalDiscriminator, LATENT_DIM

ROOT = Path(__file__).resolve().parents[1]


def choose_device():
    # CUDA, Apple MPS, CPU 순서로 사용 가능한 학습 장치를 선택한다.
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(epochs: int, batch_size: int, learning_rate: float):
    # 동일한 설정에서 가능한 한 비슷한 결과를 재현하도록 난수 시드를 지정한다.
    torch.manual_seed(42)
    device = choose_device()
    print(f"학습 장치: {device}")
    # Tanh 출력 범위에 맞춰 실제 MNIST 이미지를 [-1, 1]로 정규화한다.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # 최초 실행 시 MNIST 데이터를 다운로드하고 이후에는 로컬 데이터를 재사용한다.
    dataset = datasets.MNIST(root=str(ROOT / "data"), train=True, download=True, transform=transform)
    # Windows에서도 실행하기 쉽도록 num_workers=0으로 설정한다.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 숫자 조건을 입력받는 Generator와 Discriminator를 초기화한다.
    generator = ConditionalGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)
    # BCEWithLogitsLoss는 내부적으로 Sigmoid를 처리한다.
    criterion = nn.BCEWithLogitsLoss()
    # 각 모델의 파라미터를 서로 독립적으로 갱신하도록 옵티마이저를 분리한다.
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    # 체크포인트와 샘플 이미지를 저장할 폴더를 준비한다.
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        sum_g, sum_d = 0.0, 0.0
        for real_images, labels in loader:
            # 실제 이미지의 28x28 픽셀을 784차원 벡터로 변환한다.
            real_images = real_images.to(device).reshape(-1, 784)
            labels = labels.to(device)
            count = real_images.size(0)
            real_targets = torch.ones(count, 1, device=device)
            fake_targets = torch.zeros(count, 1, device=device)
            # 첫 번째 단계에서는 Discriminator의 기울기를 초기화한다.
            optimizer_d.zero_grad()
            # 실제 이미지와 해당 숫자 레이블의 조합을 진짜(1)로 학습한다.
            real_logits = discriminator(real_images, labels)
            real_loss = criterion(real_logits, real_targets)
            # 현재 배치에 대응하는 무작위 노이즈를 생성한다.
            noise = torch.randn(count, LATENT_DIM, device=device)
            fake_images = generator(noise, labels)
            # detach()를 사용해 판별자 학습 시 Generator로 기울기가 흐르지 않도록 한다.
            fake_logits = discriminator(fake_images.detach(), labels)
            fake_loss = criterion(fake_logits, fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()
            # 두 번째 단계에서는 Generator의 기울기를 초기화한다.
            optimizer_g.zero_grad()
            # Generator 업데이트 중에는 판별자 파라미터의 기울기 계산을 생략한다.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            # 새로운 노이즈를 사용해 지정된 숫자에 해당하는 이미지를 생성한다.
            noise = torch.randn(count, LATENT_DIM, device=device)
            generated = generator(noise, labels)
            # 가짜 이미지가 진짜(1)로 판별되도록 Non-saturating GAN 손실을 계산한다.
            g_logits = discriminator(generated, labels)
            g_loss = criterion(g_logits, real_targets)
            g_loss.backward()
            optimizer_g.step()
            # 다음 배치의 판별자 학습을 위해 파라미터 기울기를 다시 활성화한다.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            sum_g += g_loss.item()
            sum_d += d_loss.item()
        print(f"Epoch {epoch:03d}/{epochs:03d} | D Loss: {sum_d / len(loader):.4f} | G Loss: {sum_g / len(loader):.4f}")
        # 학습 중간 결과를 저장해 숫자별 생성 상태를 점검한다.
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            generator.eval()
            with torch.inference_mode():
                # 0~9 숫자를 각각 한 장씩 생성한다.
                sample_labels = torch.arange(10, device=device)
                sample_noise = torch.randn(10, LATENT_DIM, device=device)
                sample_images = generator(sample_noise, sample_labels).reshape(-1, 1, 28, 28)
                save_image((sample_images.cpu() + 1) / 2, model_dir / f"sample_epoch_{epoch:03d}.png", nrow=5)
            # 서버에서 로딩할 Generator의 가중치와 모델 설정을 저장한다.
            torch.save({"generator_state_dict": generator.state_dict(), "latent_dim": LATENT_DIM, "epochs": epoch}, model_dir / "cgan_generator.pth")
    print(f"학습 완료: {model_dir / 'cgan_generator.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Conditional GAN 학습")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, lr은 모두 0보다 커야 한다.")
    train(args.epochs, args.batch_size, args.lr)
