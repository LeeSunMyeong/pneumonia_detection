import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
# 필요하다면 import random 추가
import matplotlib.patches as patches
from math import ceil

# 랜덤 시드 설정 (재현성을 위해)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)  # 필요한 경우 주석 해제
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 데이터 경로 설정
train_csv_path = 'C:/Users/smmm/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
train_images_dir = 'C:/Users/smmm/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_images'

# 레이블 데이터 로드
label_data = pd.read_csv(train_csv_path)
columns = ['patientId', 'Target']
label_data = label_data.filter(columns)

# 데이터 분할
train_labels, val_labels = train_test_split(label_data.values, test_size=0.1, random_state=42)

# 이미지 경로 설정
train_paths = [os.path.join(train_images_dir, image[0]) for image in train_labels]
val_paths = [os.path.join(train_images_dir, image[0]) for image in val_labels]

# 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋 클래스 정의
class PneumoniaDataset(Dataset):

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # 이미지 로드
        image = dcmread(f'{self.paths[index]}.dcm')
        image = image.pixel_array.astype(np.float32)

        # 이미지 정규화
        image = (image - image.min()) / (image.max() - image.min())
        image = (255 * image).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')

        label = self.labels[index][1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':  # 추가된 부분
    # 데이터셋 및 데이터로더 생성
    train_dataset = PneumoniaDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PneumoniaDataset(val_paths, val_labels, transform=val_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 모델 정의
    from torchvision.models import resnet18, ResNet18_Weights  # ResNet18_Weights 임포트

    # 모델 로드
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 학습 및 검증 손실과 정확도를 저장할 리스트 초기화
    num_epochs = 20
    best_val_acc = 0.0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 학습 시작
    print("학습 시작")
    for epoch in range(num_epochs):
        print(f"학습 {epoch}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계 기록
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 학습률 스케줄러 스텝
        exp_lr_scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, total=len(val_loader)):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')

        # 가장 좋은 모델 저장
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

    # 최종 평가
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = nn.functional.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    final_acc = 100 * correct / total
    print(f'Final Validation Accuracy: {final_acc:.2f}%')

    # 학습 및 검증 손실 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 학습 및 검증 정확도 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # 혼동 행렬 그리기
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # ROC 곡선 및 AUC 값 계산 및 그리기
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


    # 테스트 이미지 시각화 (발표 자료용)
    def show_test_image():
        # 폐렴으로 라벨링된 이미지 중 하나를 선택
        index = 0
        while True:
            image, label = val_dataset[index]
            if label == 1:
                break
            index += 1

        # 이미지 시각화
        image_np = image.numpy().transpose((1, 2, 0))
        plt.imshow(image_np)
        plt.title('Sample Pneumonia Image')
        plt.axis('off')
        plt.show()

    # 필요할 때 주석 해제하여 테스트 이미지 시각화
    show_test_image()
