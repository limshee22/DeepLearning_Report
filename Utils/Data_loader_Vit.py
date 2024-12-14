from torchvision import transforms
from d2l import torch as d2l

def load_data(batch_size):
    """바나나 객체 검출 데이터셋을 로드하며 크기를 224x224로 고정합니다."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT 입력 크기 고정
        transforms.ToTensor()
    ])

    train_iter, val_iter = d2l.load_data_bananas(batch_size)

    # 원본 데이터를 변환하여 저장
    train_data = [(transform(X), Y) for X, Y in train_iter]
    val_data = [(transform(X), Y) for X, Y in val_iter]

    return train_data, val_data
