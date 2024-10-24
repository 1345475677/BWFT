from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from types import MethodType

from torch.utils.data import DataLoader

import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from dataloaders.dataloader import feature_dataset

class Only_last(NormalNN):

    def __init__(self, learner_config):
        self.feature_root = learner_config['feature_root']
        super(Only_last, self).__init__(learner_config)

    def update_model(self, features, targets):

        # logits
        logits = self.model.decode(features)
        logits = logits[: ,:self.valid_out_dim]

        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        # ce loss
        total_loss = total_loss

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # print(self.config)
        if self.task_count==0:
            self.data_weighting()#此处无用
            for i, blk in enumerate(self.model.feat.blocks):
                # for p in blk.attn.parameters():
                # if i<6:
                    for p in blk.parameters():
                            p.requires_grad = True
            self.model.origional_params()
            print("start_first_task")
            self.init_optimizer(ca=False)
            for epoch in range(15):
                losslog=[]
                l2log=[]
                if epoch > 0: self.scheduler.step()
                for i, (x, y, task) in enumerate(train_loader):
                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    # model update
                    logits = self.model(x)
                    logits = logits[:, :self.valid_out_dim]

                    dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]
                    total_loss = self.criterion(logits, y.long(), dw_cls)
                    # ce loss
                    l2loss=self.model.l2_loss()
                    losslog.append(total_loss)
                    l2log.append(l2loss.item())
                    total_loss = total_loss+l2loss

                    # step
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                print(sum(losslog) / len(losslog), sum(l2log) / len(l2log))
            for p in self.model.feat.parameters():
                p.detach_()
            self.init_optimizer()
        self.model.before_task(self.feature_root,train_dataset,self.task_count,self.config['dataset'],self.config['save_distribution'])
        feature_data=feature_dataset(self.feature_root,self.config['save_distribution'])
        print(len(feature_data))
        feature_loader = DataLoader(feature_data,
                                  batch_size=self.batch_size, shuffle=True,
                                  drop_last=True,
                                  num_workers=2)
        task=self.task_count
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:

            # data weighting
            self.data_weighting()#此处无用
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch
                # continue
                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y) in enumerate(feature_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y)
                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses, acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        try:
            return batch_time.avg
        except:
            return None

    # sets model optimizers
    def init_optimizer(self,ca=True):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            # params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
            params_to_opt = list(self.model.parameters())
        else:
            # params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
            params_to_opt = list(self.model.parameters())
        print('*****************************************')
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD' ,'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'] ,0.999)

        # create optimizers
        # self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        #
        # # create schedules
        # if self.schedule_type == 'cosine':
        #     self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        # elif self.schedule_type == 'decay':
        #     self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
        lrate=0.01
        weight_decay = 5e-4
        milestones = [13]
        lrate_decay = 0.1
        head_scale=1.
        bcb_lrscale=0.01
        base_params = {'params': list(self.model.feat.parameters()), 'lr': lrate * bcb_lrscale, 'weight_decay': weight_decay}
        base_fc_params = {'params': list(self.model.last.parameters()), 'lr': lrate * head_scale, 'weight_decay': weight_decay}
        network_params = [base_params, base_fc_params]
        self.optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=lrate_decay)
        # self.optimizer = optim.SGD(list(self.model.parameters()), momentum=0.9, lr=0.01,
        #                       weight_decay=0.0005)
        if ca is True:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class All_feature(Only_last):

    def __init__(self, learner_config):
        super(All_feature, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.zoo.All_features(num_classes=self.out_dim)
        return model
