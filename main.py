# coding:utf-8
import csv
import os
import random
import time
from thop import profile
import torch
import scipy.io as sio
import numpy as np
import modules
from modules import dataset, loss_fun
from models import newnetwork_upfusion_upmamba
from Toolbox import metric
import warnings

import modules.loss_fun
warnings.filterwarnings("ignore")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


# Press the green button in the gutter to run the script.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
save_bathpath="./result/"


def train(model, loss_op, train_loader, optimizer,a,b):
    model.train()
    total_loss = 0.
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list = [x_i.to(device) for x_i in x]
        labels_, low_feature,_= model(x_list)
        loss = loss_op(low_feature,labels_,a,b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def inference(test_loader, model,is_labeled_pixel):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            _, _,pred=model(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)

    print("Before indexing:", np.unique(y_pred_vector))
    # print("Features shape {}".format(y_pred_vector.shape))
    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]
        acc, kappa, nmi, ari, pur, fscore, ca = metric.cluster_accuracy(y, y_pred)
    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f} F-score={:.4f}'.format(acc, kappa, nmi, ari, pur,fscore))
    # y_pred = y_pred_vector.reshape(1723, 476)
    # y_pred = y_pred_vector.reshape(332, 485)
    # y_pred = y_pred_vector.reshape(325, 220)
    # acc = '{:.4f}'.format(acc)
    # # running_time =513.40 #'{:.2f}'.format(running_time)
    # sio.savemat(f'Berlin_'+acc+'.mat', {'y_pred': y_pred})
    return acc, kappa, nmi, ari, pur,fscore,ca


def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
set_seed=10
def worker_init_fn(worker_id):
        np.random.seed(set_seed + worker_id)
def train_for(DATASET,a,b,pre_train = 20,num_state=16,L=1,g=4,instance_temperature=1,cluster_temperature = 1,is_loadresult=False):
    setup_seed(set_seed)
    # 加载数据集
    if DATASET == 'Trento':
        img_path = ('datasets/Trento/Trento-HSI.mat', 'datasets/Trento/Trento-Lidar.mat')
        gt_path = 'datasets/Trento/Trento-GT.mat'
        patch_image_size = 7  # patch大小
        batch_sizes = 1024
        workers = 2
        train_epoch = 20
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005


    elif DATASET == 'Houston':
        img_path = ('datasets/Houston2013_Data/Houston2013_HSI.mat', 'datasets/Houston2013_Data/Houston2013_MS.mat')
        gt_path = 'datasets/Houston2013_Data/GT.mat'
        patch_image_size = 7  # patch大小
        batch_sizes = 1024
        workers = 2
        train_epoch = 20
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005

    elif DATASET == 'Augsburg':
        img_path = ('datasets/HS-SAR-DSM Augsburg/data_HS_LR.mat', 'datasets/HS-SAR-DSM Augsburg/data_SAR_HR.mat')
        gt_path = 'datasets/HS-SAR-DSM Augsburg/GT.mat'
        patch_image_size = 7  # patch大小 7
        batch_sizes = 1024
        workers = 2
        train_epoch = 20
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005

    elif DATASET == 'Berlin':
        img_path = ('datasets/HS-SAR Berlin/data_HS_LR.mat', 'datasets/HS-SAR Berlin/data_SAR_HR.mat')
        gt_path = 'datasets/HS-SAR Berlin/GT.mat'
        patch_image_size = 13  # patch大小 5
        batch_sizes = 1024
        workers = 2
        train_epoch = 15
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005


    elif DATASET == 'MUUFL':
        img_path = ('datasets/MUUFL/HSI.mat', 'datasets/MUUFL/LiDAR.mat')
        gt_path = 'datasets/MUUFL/gt.mat'
        patch_image_size = 7  # patch大小 7
        batch_sizes = 1024
        workers = 2
        train_epoch = 20
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005


    elif DATASET == 'MDAS':
        img_path = ('datasets/MDAS/MDAS-Sub1-HSI.mat', 'datasets/MDAS/MDAS-Sub1-MSI.mat')
        gt_path = 'datasets/MDAS/MDAS-Sub1-GT.mat'
        patch_image_size = 7  # patch大小
        batch_sizes = 1024
        workers = 2
        train_epoch = 20
        dim_high = 256
        learning_rate = 0.0005
        weight_decay = 0.0005
    print(patch_image_size)
    # dataset_train:
    # data_size(int)=H*W;       n_classes=地物数量      gt_shape(tuple)=(H,W);      n_modality=模态数量
    # in_channels(tuple)=(C1,C2);       y_tensor=标签的tensor形式;      modality(tuple)=两个模态的tensor数据
    print(f"开始加载数据... 当前数据集为{DATASET} batch_size={batch_sizes}")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    dataset_train = dataset.MultiModalDataset(gt_path, *img_path, patch_size=(patch_image_size, patch_image_size))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print("划分训练集数据包...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_sizes,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )
    print("划分测试集数据包...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_sizes * 2,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init_fn
    )
    # 搭建网络
    print("搭建网络...")
    model = newnetwork_upfusion_upmamba.Net(num_views=2, input_sizes=data_loader_train.dataset.in_channels,
                        dim_high_feature=dim_high, num_clusters=class_num,
                        num_state=num_state,token_num=g,group_num=g,L=1)

    model = model.to(device)
    print(device)
    model_loss = loss_fun.DeepMVCLoss(batch_sizes, class_num, instance_temperature, cluster_temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    score_list = []

    model.train()
    t = time.time()
    if (is_loadresult):
        for epoch in range(1, pre_train + 1):
            total_loss = 0.
            print(f"Epoch [{epoch}/{pre_train}]")
            for step, (x, y) in enumerate(data_loader_train):
                optimizer.zero_grad()
                x_list = [x_i.to(device) for x_i in x]
                re_x = model.pre_train(x_list, dim_high)
                model_path = "/data/liuweiqi/idea1_code/model_pth/g8_L1_12_MUUFL_10_7_0.5916.pth"
                model.load_state_dict(torch.load(model_path))
                inference(data_loader_test, model, is_labeled_pixel=False)
                return
    else:
        for epoch in range(1, pre_train + 1):
            total_loss = 0.
            print(f"Epoch [{epoch}/{pre_train}]")
            for step, (x, y) in enumerate(data_loader_train):
                optimizer.zero_grad()
                x_list = [x_i.to(device) for x_i in x]
                re_x = model.pre_train(x_list, dim_high)
                loss = model_loss.re_loss(re_x, x_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(total_loss / len(data_loader_train))

    loss_history = []
    print('Train Start')
    save_acc=0
    for epoch in range(1, train_epoch + 1):
        print(f"Epoch [{epoch}/{train_epoch}]")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        loss_epoch = train(model, model_loss, data_loader_train, optimizer,a,b)
        # t_epoch =time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"Loss: {loss_epoch / len(data_loader_train)}")
        if epoch % 1 == 0:
            acc, kappa, nmi, ari, pur, fsocre,ca = inference(data_loader_test, model, is_labeled_pixel=False)
            if acc>=save_acc:
                save_acc=acc
                torch.save(model.state_dict(),f"{save_bathpath}/{DATASET}_{pre_train}_{patch_image_size}_best.pth")
            score_list.append([acc, kappa, nmi, ari, pur])
        loss_history.append(loss_epoch / len(data_loader_train))
    os.rename(f"{save_bathpath}/{DATASET}_{pre_train}_{patch_image_size}_best.pth", f"{save_bathpath}/{DATASET}_{pre_train}_{patch_image_size}_{save_acc:.4f}.pth")
    print("Total time elapsed: {:.2f}s".format(time.time() - t))
    print("train finished.")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dataname=['MUUFL']
    # num_state=[12]
    # g_num=[8]
    # b=1
    # a=1

    dataname=['Augsburg','MUUFL','Berlin''MDAS','Houston']#
    num_state=[2,4,8,12,16,20]
    g_num=[2,4,8,16]
    b=1
    a=0.8
    pre_train_epoch=[1,5,10,15]
    L_num=[1]
    for j in range(len(dataname)):
                for i in range(0,len(pre_train_epoch)):
                    for k in num_state:
                        for g in g_num: 
                            print(f"当前训练数据为{dataname[j]},L={1},正在计算预训练轮次为{pre_train_epoch[i]},状态数为{k},光谱分组为{g}的最优结果")#,光谱分组为{g}
                            print(f"a={a},b={b}")
                            train_for(dataname[j],a,b,pre_train_epoch[i],k,1,g,0.5,0.7,False)


