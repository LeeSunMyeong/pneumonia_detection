import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.data_loader import PneumoniaDataset
from models.model import get_model

import os

def train():
    # 하이퍼파라미터 설정
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 변환 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 데이터셋 및 데이터로더 생성
    csv_file = 'path_to_dataset/stage_2_train_labels.csv'
    images_dir = 'path_to_dataset/stage_2_train_images'

    # 전체 데이터 로드
    full_df = pd.read_csv(csv_file)
    full_df = full_df[['patientId', 'Target']]

    # 훈련/검증 데이터 분할
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    # 데이터셋 생성
    train_dataset = PneumoniaDataset(csv_file=None, images_dir=images_dir, transform=train_transform)
    train_dataset.df = train_df.reset_index(drop=True)

    val_dataset = PneumoniaDataset(csv_file=None, images_dir=images_dir, transform=val_transform)
    val_dataset.df = val_df.reset_index(drop=True)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 로드
    model = get_model().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습률 스케줄러 (옵션)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 학습 시작
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            # 통계 업데이트
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # 검증 단계
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_images.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%\n')

    # 모델 저장
    torch.save(model.state_dict(), 'models/pneumonia_model.pth')
    print('Model saved.')

if __name__ == '__main__':
    train()
