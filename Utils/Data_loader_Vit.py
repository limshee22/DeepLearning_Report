import torch.nn.functional as F
from d2l import torch as d2l

import torch.nn.functional as F
import torch

def preprocess_test_data(test_batch):
    """테스트 데이터 변환."""
    test_img_tensor, test_label = test_batch
    # 각 이미지 텐서를 224x224로 변환
    test_img_tensor = torch.stack([F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0) for img in test_img_tensor])
    return test_img_tensor, test_label

def resize_images_in_loader(data_iter, img_size=(224, 224)):
    """데이터 로더에서 이미지를 지정된 크기로 리사이즈."""
    resized_data = []
    for features, labels in data_iter:
        resized_features = F.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
        resized_data.append((resized_features, labels))
    return resized_data

def load_data(batch_size):
    """바나나 객체 검출 데이터셋을 로드합니다."""
    train_iter, val_iter = d2l.load_data_bananas(batch_size)
    train_iter = resize_images_in_loader(train_iter, img_size=(224, 224))
    val_iter = resize_images_in_loader(val_iter, img_size=(224, 224))
    return train_iter, val_iter


