"""
NLP 辅助函数（用于 Week 4 实验）
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def plot_training_history(train_losses, train_accs=None, val_accs=None, title="Training History"):
    """
    绘制训练损失和准确率曲线。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss')
    
    if train_accs is not None:
        ax2.plot(train_accs, label='Train Acc')
    if val_accs is not None:
        ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.suptitle(title)
    plt.show()

def compute_accuracy(preds, labels):
    """计算准确率（封装 sklearn）"""
    return accuracy_score(labels, preds)

def print_classification_report(preds, labels, target_names=None):
    """打印分类报告"""
    print(classification_report(labels, preds, target_names=target_names))
