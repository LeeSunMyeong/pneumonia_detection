# scripts/explore_data.py
from utils.utils import visualize_sample_images


def explore_data():
    pneumonia_dir = '../data/train/pneumonia'
    normal_dir = '../data/train/normal'

    print("폐렴 이미지 샘플:")
    visualize_sample_images(pneumonia_dir)

    print("정상 이미지 샘플:")
    visualize_sample_images(normal_dir)


if __name__ == "__main__":
    explore_data()
