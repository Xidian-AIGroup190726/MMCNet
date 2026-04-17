
import sys
import os
import time

from torch.optim import lr_scheduler
import logging
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import torch
import numpy as np


def setup_logger(log_file):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台处理器（不处理stdout）
    class NonStdoutStreamHandler(logging.StreamHandler):
        def __init__(self):
            super().__init__(sys.stderr)
        def emit(self, record):
            if record.levelno != logging.INFO:
                super().emit(record)

    ch = NonStdoutStreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def save_model(model, optimizer, epoch, best_metric, save_dir, model_name, dataset):
    """
    保存模型和训练状态

    Args:
    - model: 要保存的模型
    - optimizer: 优化器
    - epoch: 当前的epoch
    - best_metric: 最佳的评估指标（例如最高的准确率）
    - save_dir: 保存模型的目录
    - model_name: 模型的名称
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 构造文件名，包含epoch信息
    filename = f"{model_name}_{dataset}_epoch_{epoch}.pth"
    save_path = os.path.join(save_dir, filename)

    # 准备要保存的数据
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }

    # 保存模型
    torch.save(state, save_path)
    print(f"Model saved to {save_path}")

    # 保存最新的模型（覆盖之前的）
    latest_path = os.path.join(save_dir, f"{model_name}_{dataset}_latest.pth")
    torch.save(state, latest_path)
    print(f"Latest model saved to {latest_path}")


def trainval_model(model, train_loader, test_loader, LR, EPOCH, save_dir, netname, dataset, num_classes):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_accuracy = 0
    epoch_pbar = tqdm(range(EPOCH), desc="Training Epochs", file=sys.stdout)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCH}", position=0, leave=True)
        for step, (ms, pan, target, _) in enumerate(batch_pbar):
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            loss = model(pan, ms, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            batch_pbar.set_postfix({"Loss": f"{avg_loss:.4f}"}, refresh=True)

        scheduler.step()

        if epoch == EPOCH - 1:
            test_time, accuracy, precision, recall, f1, oa, aa, kappa, conf_matrix = evaluate_model(model, test_loader)
            epoch_pbar.set_postfix({
                "Accuracy": f"{accuracy:.4f}",
                "OA": f"{oa:.4f}",
                "AA": f"{aa:.4f}",
                "Kappa": f"{kappa:.4f}"
            })

            logging.info(
                f'\nEpoch [{epoch + 1}/{EPOCH}], Accuracy: {accuracy:.4f}, OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}')
            logging.info(f'Precision: {precision}')
            logging.info(f'Recall: {recall}')
            logging.info(f'F1: {f1}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy

                print("\nOA: {:.2f}  \nAA: {:.2f}  \nKappa: {:.2f} ".format(oa * 100, aa * 100, kappa * 100))
                performance_file = f'logs/{netname}_{dataset}_latest.txt'
                with open(performance_file, "a") as f:

                    f.write("epoch{}:\n".format(epoch))
                    f.write('Precision:  ')
                    for i in range(num_classes):
                        f.write('  {:.2f}'.format(precision[i] * 100))
                    f.write("\nrecall:  ")
                    for i in range(num_classes):
                        f.write('  {:.2f}'.format(recall[i] * 100))
                    f.write("\nf1:  ")
                    for i in range(num_classes):
                        f.write('  {:.2f}'.format(f1[i] * 100))
                    f.write("\n")
                    f.write("OA: {:.2f}  \nAA: {:.2f}  \nKappa: {:.2f} ".format(oa * 100, aa * 100, kappa * 100) + "\n")
                    f.write("test_time: {:.2f} S".format(test_time) + "\n")
                    f.write('Confusion Matrix:\n')
                    for i in range(len(conf_matrix)):
                        f.write(str(conf_matrix[i]) + '\n')

                save_model(model, optimizer, epoch, best_accuracy, save_dir, netname, dataset)
                logging.info(f'New best model saved with accuracy: {best_accuracy:.4f}')

def calculate_accuracy(confusion_matrix):
    """计算总体精度（OA）."""
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)


def calculate_accuracy_per_class(confusion_matrix):
    """计算每个类别的精度."""
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)


def calculate_average_accuracy(accuracy_per_class):
    """计算平均精度（AA）."""
    return np.mean(accuracy_per_class)


def calculate_precision_recall_f1(y_true, y_pred, num_classes):
    """计算每个类的查准率、查全率和F1分数."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), average=None)
    return precision, recall, f1


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    begin_test = time.time()
    with torch.no_grad():
        for ms, pan, target, _ in tqdm(test_loader, desc="Evaluating", position=0, leave=True):
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            output = model(pan, ms)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    end_test = time.time()
    test_time = end_test - begin_test

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # Calculate Matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    # Calculate overall accuracy
    accuracy = (all_preds == all_targets).mean()

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)

    # Calculate OA, AA, and Kappa
    oa = calculate_accuracy(conf_matrix)
    aa = np.mean(recall)  # AA is the mean of recall for each class
    kappa = cohen_kappa_score(all_targets, all_preds)

    return test_time, accuracy, precision, recall, f1, oa, aa, kappa, conf_matrix