import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.cluster import SpectralClustering, KMeans
import dataloaders
from . import mae_vit
from .vit import VisionTransformer,VisionTransformer2
import numpy as np
import os
import copy


class All_features(nn.Module):
    def __init__(self,num_classes=10):
        super(All_features, self).__init__()
        self.pretrain_type='1k'
        if self.pretrain_type=='1k':
            self.feat=VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                            num_heads=12, ckpt_layer=0,
                                            drop_path_rate=0
                                            )
            from timm.models import vit_base_patch16_224,vit_base_patch16_224_in21k

            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
            del load_dict['head.weight']
            del load_dict['head.bias']
            self.feat.load_state_dict(load_dict)
        elif self.pretrain_type=='mae':
            #mae_vit
            model = mae_vit.vit_base_patch16(global_pool=True)
            checkpoint = torch.load('./models/mae_finetuned_vit_base.pth', map_location='cpu')
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            mae_vit.interpolate_pos_embed(model, checkpoint_model)
            _ = model.load_state_dict(checkpoint_model, strict=False)
            model.head = nn.Identity()
            self.feat=model
            #maevit




        self.gpu=True
        self.prompt=None
        self.classifer = nn.Sequential(
            nn.Linear(768,1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        self.last=copy.deepcopy(self.classifer)
        self.feature_buffer=[]#特征缓冲列表，保存了过去的所有feature，feature_buffer[i]=[BS,dim],i<=总增量步数
        self.label_buffer=[]

        for p in self.feat.parameters():
            p.requires_grad = False
    def forward(self,x):
        if self.pretrain_type=='1k':
            cls_token,_=self.feat(x)
            y = self.last(cls_token[:, 0, :])
        else:
            cls_token = self.feat.forward_features(x)
            y=self.last(cls_token)
        return y

    def decode(self,cls_token):
        return self.last(cls_token)

    def get_current_features(self,train_dataset,dataset):
        feature_list=[]
        label_list=[]
        train_dataset.transform=dataloaders.utils.get_transform(dataset=dataset, phase='test')
        train_loader=DataLoader(train_dataset, batch_size=10, shuffle=False, drop_last=True, num_workers=0)
        for i, (x, y, task) in enumerate(train_loader):
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            if self.pretrain_type == '1k':
                features,_=self.feat(x)
                feature_list.append(features[:,0,:].detach().clone().cpu())
                label_list.append(y.cpu())
            else:
                cls_token=self.feat.forward_features(x)
                feature_list.append(cls_token.detach().clone().cpu())
                label_list.append(y.cpu())
        feature_list=torch.cat(feature_list,dim=0)#[global_batch,dim]
        label_list=torch.cat(label_list,dim=0)
        self.feature_buffer.append(feature_list)
        self.label_buffer.append(label_list)

    def save_feature_buffer(self,path):
        torch.save((self.feature_buffer,self.label_buffer),path)
        self.feature_buffer=[]
        self.label_buffer=[]

    def load_feature_buffer(self,path):
        self.feature_buffer,self.label_buffer=torch.load(path)

    def before_task(self,root,train_dataset,task_id,dataset,save_distribution=False):
        if save_distribution>0:
            self.last = copy.deepcopy(self.classifer)
            if task_id == 0:
                if os.path.exists(root):
                    os.remove(root)
                self.get_current_distribution(train_dataset, dataset,save_distribution)
                self.save_feature_buffer(root)
            else:
                self.load_feature_buffer(root)
                self.get_current_distribution(train_dataset, dataset,save_distribution)
                self.save_feature_buffer(root)
        else:
            self.last = copy.deepcopy(self.classifer)
            if task_id==0:
                if os.path.exists(root):
                    os.remove(root)
                self.get_current_features(train_dataset,dataset)
                self.save_feature_buffer(root)
            else:
                self.load_feature_buffer(root)
                self.get_current_features(train_dataset,dataset)
                self.save_feature_buffer(root)
    def init_param(self,m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    def get_current_distribution(self,train_dataset,dataset,save_distribution):
        feature_list = []
        label_list = []
        feature_temp=[]
        label_temp=[]
        train_dataset.transform = dataloaders.utils.get_transform(dataset=dataset, phase='train')
        test_mode_dataset=copy.deepcopy(train_dataset)
        test_mode_dataset.transform=dataloaders.utils.get_transform(dataset=dataset, phase='test')
        if save_distribution == 1 or save_distribution==2:
            for i in range(len(train_dataset)):
                epoch=20
                original_img,label,t=test_mode_dataset[i]
                original_mean=self.feat(original_img.unsqueeze(dim=0).cuda().detach())[0][:,0,:]
                imgs=[]
                for j in range(epoch):
                    img,_,_=train_dataset[i]
                    if self.gpu:
                        img=img.cuda()
                    imgs.append(img.unsqueeze(dim=0))
                imgs=torch.cat(imgs,dim=0)

                if self.pretrain_type == '1k':
                    features, _ = self.feat(imgs)
                    features = features[:, 0, :].detach()
                else:
                    cls_token = self.feat.forward_features(imgs)
                    features=cls_token

                mean=torch.mean(features,dim=0).cpu()
                std=torch.std(features,dim=0).cpu()
                feature_list.append({"mean":mean.cpu(),"std":std.cpu(),"original_mean":original_mean.squeeze(dim=0).cpu()})
                label_list.append(label)
                if save_distribution == 2:
                    feature_temp.append(original_mean)
                    label_temp.append(torch.tensor(label).unsqueeze(dim=0).cuda())
        if save_distribution == 2:
            feature_temp = torch.cat(feature_temp, dim=0)  # [BS,dim]
            label_temp=torch.cat(label_temp,dim=0)
            all_labels=torch.unique(label_temp,sorted=False)
            global_index=len(self.feature_buffer)
            for label in all_labels:
                index=torch.nonzero(label_temp==label).squeeze(dim=-1)#得到当前类别的局部位置与全局位置的映射
                local_feature=feature_temp[index]
                local_cos_sim=torch.einsum("bd,kd->bk", F.normalize(local_feature, dim=1),
                                  F.normalize(local_feature, dim=1)) #[bs,bs]
                local_topk = torch.topk(input=local_cos_sim, k=5, dim=1)
                sim_index = index[local_topk.indices] #[bs,5]
                for i in range(len(index)):
                    feature_list[index[i]]["sim_index"]=sim_index[i].cpu()+global_index

        if save_distribution == 3 :#特征合并 #目前最好的效果是n=25，k=2，对应3；n=20，k=3效果次好，对应无后缀；n=25，k=2.但是聚类的时候只聚类中心区域，效果最差，对应2
            # n_clusters=25
            n_clusters=100#defualt 25
            n_outliers_per_cluster=3#defualt 2
            train_loader = DataLoader(test_mode_dataset, batch_size=100, shuffle=False, drop_last=True, num_workers=2)
            feature_temp=[]
            label_temp=[]
            for i, (x, y, task) in enumerate(train_loader):
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                if self.pretrain_type == '1k':
                    features, _ = self.feat(x)
                    feature_temp.append(features[:, 0, :].detach().clone().cpu())
                    label_temp.append(y.cpu())
                else:
                    cls_token = self.feat.forward_features(x)
                    feature_temp.append(cls_token.detach().clone().cpu())
                    label_temp.append(y.cpu())
            feature_temp = torch.cat(feature_temp, dim=0)  # [global_batch,dim]
            label_temp = torch.cat(label_temp, dim=0)
            all_labels=torch.unique(label_temp,sorted=False)
            for label in all_labels:
                index = torch.nonzero(label_temp == label).squeeze(dim=-1)  # 得到当前类别的局部位置与全局位置的映射
                local_feature = feature_temp[index]#得到所有当前label的feature [bs,dim]
                clustering=KMeans(n_clusters=n_clusters,random_state=2024)
                clustering.fit(local_feature.cpu().numpy())
                for i in range(n_clusters):
                    feature=local_feature[clustering.labels_==i,:].cuda()
                    mean1 = torch.mean(feature, dim=0)
                    sim=torch.einsum("bd,d->b",F.normalize(feature, dim=1),F.normalize(mean1, dim=0))
                    if len(feature)>=7:
                        dtopk = torch.topk(input=sim, k=n_outliers_per_cluster, dim=0,largest=False)#取出相似度最小的3个点
                        free_point=feature[dtopk.indices]#目前最好的k是3，n=20
                        topk = torch.topk(input=sim, k=len(feature)-n_outliers_per_cluster, dim=0)#len(feature)-3
                        center_point=feature[topk.indices]
                    else:
                        free_point = feature
                        center_point = None
                    if center_point is not None:
                        mean2=torch.mean(center_point,dim=0).cpu()
                        std = torch.std(center_point, dim=0).cpu()
                        feature_list.append({"mean": mean2.cpu(), "std": std.cpu()})
                        label_list.append(label)
                    for j in range(len(free_point)):
                        feature_list.append({"mean":free_point[j].cpu(),"std":0})
                        label_list.append(label)

        if save_distribution == 4 :#筛选
            train_loader = DataLoader(test_mode_dataset, batch_size=100, shuffle=False, drop_last=True, num_workers=2)
            feature_temp=[]
            label_temp=[]
            for i, (x, y, task) in enumerate(train_loader):
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                if self.pretrain_type == '1k':
                    features, _ = self.feat(x)
                    feature_temp.append(features[:, 0, :].detach().clone().cpu())
                    label_temp.append(y.cpu())
                else:
                    cls_token = self.feat.forward_features(x)
                    feature_temp.append(cls_token.detach().clone().cpu())
                    label_temp.append(y.cpu())
            feature_temp = torch.cat(feature_temp, dim=0)  # [global_batch,dim]
            label_temp = torch.cat(label_temp, dim=0)
            all_labels=torch.unique(label_temp,sorted=False)
            for label in all_labels:
                index = torch.nonzero(label_temp == label).squeeze(dim=-1)  # 得到当前类别的局部位置与全局位置的映射
                local_feature = feature_temp[index]#得到所有当前label的feature [bs,dim]
                indexes = [i for i in range(len(local_feature))]
                index_list = random.sample(indexes, 250)
                for j in range(250):
                    feature_list.append({"mean":local_feature[index_list[j]],"std":0})
                    label_list.append(label)

        if save_distribution == 5 :#特征合并
            n_clusters=125
            train_loader = DataLoader(test_mode_dataset, batch_size=100, shuffle=False, drop_last=True, num_workers=2)
            feature_temp=[]
            label_temp=[]
            for i, (x, y, task) in enumerate(train_loader):
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                if self.pretrain_type == '1k':
                    features, _ = self.feat(x)
                    feature_temp.append(features[:, 0, :].detach().clone().cpu())
                    label_temp.append(y.cpu())
                else:
                    cls_token = self.feat.forward_features(x)
                    feature_temp.append(cls_token.detach().clone().cpu())
                    label_temp.append(y.cpu())
            feature_temp = torch.cat(feature_temp, dim=0)  # [global_batch,dim]
            label_temp = torch.cat(label_temp, dim=0)
            all_labels=torch.unique(label_temp,sorted=False)
            for label in all_labels:
                index = torch.nonzero(label_temp == label).squeeze(dim=-1)  # 得到当前类别的局部位置与全局位置的映射
                local_feature = feature_temp[index]#得到所有当前label的feature [bs,dim]
                clustering=KMeans(n_clusters=n_clusters,random_state=2024)
                clustering.fit(local_feature.cpu().numpy())
                for i in range(n_clusters):
                    feature=local_feature[clustering.labels_==i,:].cuda()
                    if len(feature)>=3:
                        mean1 = torch.mean(feature, dim=0)
                        std1=torch.std(feature,dim=0)
                        feature_list.append({"mean": mean1.cpu(), "std": std1.cpu()})
                        label_list.append(label)
                    else:
                        mean1 = feature[0]
                        feature_list.append({"mean": mean1.cpu(), "std": 0})
                        label_list.append(label)


        self.feature_buffer+=feature_list
        self.label_buffer+=label_list
        print(len(self.feature_buffer))
        print(len(self.label_buffer))

    def origional_params(self):
        task_param = {}
        self.params = {n: p for n, p in self.feat.named_parameters() if p.requires_grad}#正则化特征
        self.regularization_terms = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
        self.regularization_terms[0] = {'importance': 1.0, 'task_param': task_param}
    def l2_loss(self):
        l2_loss=0
        for i, reg_term in self.regularization_terms.items():
            task_reg_loss = 0
            importance = reg_term['importance']
            task_param = reg_term['task_param']
            for n, p in self.params.items():
                task_reg_loss += (importance * (p - task_param[n]) ** 2).sum()
            l2_loss += task_reg_loss
        return l2_loss

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None):
    # return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)
    return All_features(num_classes=out_dim)
