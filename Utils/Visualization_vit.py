import torch
from torch import nn
from d2l import torch as d2l
from Model.ViTBasedTinySSD import ViTBasedTinySSD
from Utils.Data_loader import load_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

def predict_and_visualize(net, X, img, threshold=0.9):
    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1 and float(row[1]) > threshold]
    plt.figure(figsize=(10, 10))
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.max() > 1.0:
        img = img / 255.0
    plt.imshow(img)
    print(f"검출된 객체 수: {len(idx)}")
    for row in output[0][idx]:
        score = float(row[1])
        h, w = img.shape[:2]
        bbox = row[2:6] * torch.tensor((w, h, w, h), device=row.device)
        if bbox.device != torch.device('cpu'):
            bbox = bbox.cpu()
        d2l.show_bboxes(plt.gca(), [bbox], f'{score:.2f}', 'r')
    plt.axis('off')
    plt.savefig("./Result/ViTBasedTinySSD/ViTBasedTinySSD.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import torchvision

def predict_and_visualize_vit(net, X, visual_img, threshold=0.9):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = torch.softmax(cls_preds, dim=1)
    detected_boxes = []
    for i, prob in enumerate(cls_probs):
        if prob[1] > threshold:
            bbox = bbox_preds[i]
            detected_boxes.append(bbox.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.imshow(visual_img.cpu())
    for bbox in detected_boxes:
        x, y, w, h = bbox
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.savefig('./Result/ViTBasedTinySSD/ViTBasedTinySSD_detection_result.png')
    plt.close()
