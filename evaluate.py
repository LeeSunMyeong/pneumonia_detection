import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.data_loader import PneumoniaDataset
from models.model import get_model

import matplotlib.pyplot as plt
import numpy as np

def evaluate():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 변환 정의
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

    # 검증 데이터셋 생성 (전체 데이터를 사용하거나 별도의 테스트 데이터를 사용)
    val_dataset = PneumoniaDataset(csv_file=csv_file, images_dir=images_dir, transform=val_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 모델 로드
    model = get_model().to(device)
    model.load_state_dict(torch.load('models/pneumonia_model.pth'))
    model.eval()

    # 평가
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy on the validation set: {accuracy:.2f}%')

    # 혼동 행렬 출력 (선택 사항)
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(cr)

if __name__ == '__main__':
    evaluate()
