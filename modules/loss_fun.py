
import torch
import torch.nn as nn
import contrastive_loss as cl

class DeepMVCLoss(nn.Module):
    def __init__(self, batch_size, num_clusters,instance_temperature,cluster_temperature):
        super(DeepMVCLoss, self).__init__()
        self.criterion_instance=cl.InstanceLoss(batch_size,instance_temperature)
        self.criterion_cluster = cl.ClusterLoss(num_clusters,cluster_temperature)
        self.mse = torch.nn.MSELoss()
    def forward(self,low_feature,label_,a,b):
        loss_instance = self.criterion_instance(low_feature[0], low_feature[1])
        loss_cluster = self.criterion_cluster(label_[0], label_[1])
        return a*loss_cluster+b*loss_instance

    def re_loss(self,re_x,x):
        return self.mse(x[0], re_x[0])+self.mse(x[1], re_x[1])