from dataloader import data_process
import torch.utils.data as Data
import datetime as dt
from Net import mynet
from train_utils2 import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

start_time = dt.datetime.now().strftime('%F %T')
print("程序开始运行时间：" + start_time)
# 网络超参数
Ms4_patch_size = 16
BATCH_SIZE = 256
dim = 64
EPOCH = 20
LR = 0.001

dim_expand_rate = 4
down_rate = [2, 2, 2]
patch_size = [2, 2, 1]

use_reconstruct = True
use_exchange = True
use_sobel = False
use_mha = True
learn = True

num_classes = 12
lam = 0.25
gate_threshold = 0.5
netname = "MMC-Net"
dataset = "big_xian"

dataset_dir = "/data/whh/project/data/" + dataset
cache_path = os.path.join(dataset_dir, f"processed_data_ps{Ms4_patch_size}.pth")

if os.path.exists(cache_path):
    checkpoint = torch.load(cache_path)
    train_data = checkpoint['train_data']
    test_data = checkpoint['test_data']
    all_data = checkpoint['all_data']
    label_test = checkpoint['label_test']
else:
    pan_path = "/data/whh/project/data/" + dataset + "/pan.tif"
    ms_path = "/data/whh/project/data/" + dataset + "/ms4.tif"
    train_label_path = "/data/whh/project/data/" + dataset + "/train.npy"
    test_label_path = "/data/whh/project/data/" + dataset + "/test.npy"
    train_data, test_data, all_data, label_test = data_process(
        pan_path, ms_path, train_label_path, test_label_path, Ms4_patch_size
    )

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = Data.DataLoader(dataset=all_data, batch_size=BATCH_SIZE * 8, shuffle=False, num_workers=0)

model = mynet(dim=dim, num_classes=num_classes,
              use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
              use_exchange=use_exchange, use_sobel=use_sobel,
              use_mha=use_mha, lam=lam, learn=learn,
              Ms4_patch_size=Ms4_patch_size,
              hidden_expand=dim_expand_rate,
              down_rate=down_rate,
              patch_size=patch_size, )
model.cuda()
model.eval()
checkpoint = torch.load(f"{netname}_{dataset}_latest.pth")
model.load_state_dict(checkpoint['model_state_dict'])
test_time, accuracy, precision, recall, f1, oa, aa, kappa, conf_matrix = evaluate_model(model, test_loader)
performance_file = f'{netname}_{dataset}_latest.txt'
with open(performance_file, "a") as f:
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
