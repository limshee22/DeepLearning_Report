import torch
from torch import nn
from d2l import torch as d2l
from Model.TinySSD import TinySSD
from Model.TinySSDWithSpatialAttention import TinySSDWithSpatialAttention
from Utils.Data_loader import load_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

def predict_and_visualize(net, X, img, threshold=0.9, save_dir = "TinySSD"):
    """
    TinySSD 모델을 사용해 입력 이미지를 예측하고 결과를 시각화합니다.
    Args:
        net: TinySSD 모델.
        X: 모델에 입력할 이미지 텐서. 크기 (batch_size, channels, height, width).
        img: 원본 이미지(numpy 형식) 또는 텐서. 크기 (height, width, channels).
        threshold: 예측 클래스의 최소 신뢰도 (기본값 0.9).
    """
    # TinySSD 모델 평가 모드로 전환
    net.eval()
    device = next(net.parameters()).device

    # TinySSD 모델을 통해 예측 수행
    with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)

    # Non-Maximum Suppression 후 유효한 예측만 필터링
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1 and float(row[1]) > threshold]

    # 시각화
    plt.figure(figsize=(10, 10))
    
    # 이미지가 텐서인 경우 numpy로 변환
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    
    # 이미지 값 범위 조정 (0-1 범위로)
    if img.max() > 1.0:
        img = img / 255.0
    
    plt.imshow(img)
    
    # 검출된 객체 수 출력
    print(f"검출된 객체 수: {len(idx)}")
    
    for row in output[0][idx]:
        score = float(row[1])
        h, w = img.shape[:2]
        
        # bbox 계산을 텐서 상태로 유지
        bbox = row[2:6] * torch.tensor((w, h, w, h), device=row.device)
        
        # CPU로 이동
        if bbox.device != torch.device('cpu'):
            bbox = bbox.cpu()
        
        # 텐서 상태로 전달
        d2l.show_bboxes(plt.gca(), [bbox], f'{score:.2f}', 'r')  # 색상을 빨간색으로 변경

    plt.axis('off')  # 축 제거
    plt.savefig(f"./Result/{save_dir}/{save_dir}.png", dpi=300)
    plt.close()



# # visualization.py
# import matplotlib.pyplot as plt
# import torchvision

# def predict_and_visualize_vit(net, X, visual_img, threshold=0.9):
#     net.eval()
#     anchors, cls_preds, bbox_preds = net(X)
    
#     cls_probs = torch.softmax(cls_preds, dim=1)
#     detected_boxes = []
    
#     for i, prob in enumerate(cls_probs):
#         if prob[1] > threshold:  # 바나나 클래스의 확률이 threshold보다 높은 경우
#             bbox = bbox_preds[i]
#             detected_boxes.append(bbox.detach().cpu().numpy())
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(visual_img.cpu())
    
#     for bbox in detected_boxes:
#         x, y, w, h = bbox
#         rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='red', linewidth=2)
#         plt.gca().add_patch(rect)
    
#     plt.axis('off')
#     plt.savefig('./Result/ViTBasedTinySSD/ViTBasedTinySSD_detection_result.png')
#     plt.close()

