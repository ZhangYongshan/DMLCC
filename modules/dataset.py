import torch
from torch.utils.data import Dataset, DataLoader
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np


class MultiModalDataset(Dataset):

    def __init__(self, gt_path, *src_path, patch_size=(7, 7),is_labeled=False, transform=None):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []
        for i in range(n_modality):
            img, gt = p.prepare_data(src_path[i], gt_path)
            x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
            n_samples, n_row, n_col, n_channel = x_patches.shape
            #StandardScaler 实例用于数据标准化，并设置批处理大小为 5000
            scaler = StandardScaler()
            batch_size = 5000
            #对数据进行分批处理，以批大小为 5000 对每个批次的数据进行标准化
            for start_id in range(0, x_patches.shape[0], batch_size):
                n_batch = x_patches[start_id: start_id+batch_size].shape[0]
                scaler.partial_fit(x_patches[start_id: start_id+batch_size].reshape(n_batch, -1))
            #对每个批次的数据进行标准化转换，并重新调整形状
            for start_id in range(0, x_patches.shape[0], batch_size):
                shape = x_patches[start_id: start_id+batch_size].shape
                x_temp = x_patches[start_id: start_id+batch_size].reshape(shape[0], -1)
                x_patches[start_id: start_id+batch_size] = scaler.transform(x_temp).reshape(shape)
            #将 x_patches 的维度顺序调整为 (batch_size, channels, height, width)，
            # 然后将其转换为 PyTorch 的 FloatTensor 并添加到 modality_list 中。
            # 同时，将通道数 n_channel 添加到 in_channels 中
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
            modality_list.append(x_tensor)
            in_channels.append(n_channel)
        #对标签 y_ 进行标准化处理
        y = p.standardize_label(y_)
        # 保存地面真实数据的形状和数据集的大小
        self.gt_shape = gt.shape
        self.data_size = len(y)
        #根据数据是否有标签来设置类别数 n_classes。
        # 如果有标签，则使用唯一标签的数量；如果没有标签，则减去背景类的数量。
        if is_labeled:
            self.n_classes = np.unique(y).shape[0]
        else:
            self.n_classes = np.unique(y).shape[0] - 1  # remove background
        #将标签 y 转换为 PyTorch 的 LongTensor，将模态数据列表 modality_list 和通道数 in_channels 转换为元组并保存。
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)
        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)

    def __getitem__(self, idx):
        x_list = []
        for i in range(self.n_modality):
            x = self.modality_list[i][idx]
            if self.transform is not None:
                x_1, x_2 = self.transform(x)  # # conduct transformation on a single modality
                x_list.append(x_1)
                x_list.append(x_2)
            else:
                x_list.append(x)
        if self.n_modality >= 2 and len(x_list) > 2:  # # when modality >= 2, i.e., 4 augs
            x_list = (x_list[0::2], x_list[1::2])
        if self.n_modality == 1 and len(x_list) == 2:
            x_list = ([x_list[0]], [x_list[1]])
        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size
