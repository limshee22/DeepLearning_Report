import torch
from torch import nn
from d2l import torch as d2l
from Model.ViTBasedTinySSD import ViTBasedTinySSD
from Utils.Data_loader import load_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
from Utils.Visualization_vit import predict_and_visualize

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def train_model(net, train_iter, valid_iter, num_epochs, device):
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    train_losses = []
    valid_losses = []
    train_cls_errors = []
    valid_cls_errors = []
    train_bbox_maes = []
    valid_bbox_maes = []

    best_valid_cls_err = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        train_loss_sum = 0
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            train_loss_sum += l.mean().item()
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())

        train_losses.append(train_loss_sum / len(train_iter))
        train_cls_errors.append(1 - metric[0] / metric[1])
        train_bbox_maes.append(metric[2] / metric[3])
        print(f'Epoch {epoch+1}, Training Class Error: {train_cls_errors[-1]:.2e}, Training BBox MAE: {train_bbox_maes[-1]:.2e}')

        net.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            metric = d2l.Accumulator(4)
            for features, target in valid_iter:
                X, Y = features.to(device), target.to(device)
                anchors, cls_preds, bbox_preds = net(X)
                bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                valid_loss_sum += l.mean().item()
                metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                           bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())

            valid_losses.append(valid_loss_sum / len(valid_iter))
            valid_cls_errors.append(1 - metric[0] / metric[1])
            valid_bbox_maes.append(metric[2] / metric[3])
            print(f'Epoch {epoch+1}, Validation Class Error: {valid_cls_errors[-1]:.2e}, Validation BBox MAE: {valid_bbox_maes[-1]:.2e}')

        checkpoint_path = f'./Result/ViTBasedTinySSD/checkpoint_epoch_{epoch+1}.pth'
        torch.save(net.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

        if valid_cls_errors[-1] < best_valid_cls_err:
            best_valid_cls_err = valid_cls_errors[-1]
            best_model_path = f'./Result/ViTBasedTinySSD/best_model_epoch.pth'
            torch.save(net.state_dict(), best_model_path)
            print(f'Best model updated: {best_model_path}')

    plt.figure(figsize=(15, 7))

    plt.subplot(3, 1, 2)
    plt.plot(range(1, num_epochs + 1), train_cls_errors, label='Training Class Error')
    plt.plot(range(1, num_epochs + 1), valid_cls_errors, label='Validation Class Error')
    plt.xlabel('Epochs')
    plt.ylabel('Class Error')
    plt.legend()
    plt.title('Training and Validation Class Error')

    plt.subplot(3, 1, 3)
    plt.plot(range(1, num_epochs + 1), train_bbox_maes, label='Training BBox MAE')
    plt.plot(range(1, num_epochs + 1), valid_bbox_maes, label='Validation BBox MAE')
    plt.xlabel('Epochs')
    plt.ylabel('BBox MAE')
    plt.legend()
    plt.title('Training and Validation BBox MAE')

    plt.tight_layout()
    plt.savefig("./Result/ViTBasedTinySSD/ViTBasedTinySSD_loss_graph.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinySSD 모델 학습 및 테스트')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기 (기본값: 32)')
    parser.add_argument('--num_epochs', type=int, default=15, help='에포크 수 (기본값: 5)')
    parser.add_argument('--threshold', type=float, default=0.9, help='검출 클래스 최소 신뢰도 (기본값: 0.9)')
    args = parser.parse_args()

    train_iter, valid_iter = load_data(args.batch_size)

    device = d2l.try_gpu()
    net = ViTBasedTinySSD(num_classes=2).to(device)

    train_model(net, train_iter, valid_iter, num_epochs=args.num_epochs, device=device)

    test_batch = next(iter(train_iter))
    test_img_tensor, test_label = test_batch
    test_img_tensor = test_img_tensor[0].unsqueeze(0)
    visual_img = test_img_tensor.squeeze(0).permute(1, 2, 0)

    predict_and_visualize(net, test_img_tensor.to(device), visual_img, threshold=args.threshold)
