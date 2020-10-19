#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:04:07 2019

@author: viswanatha
"""

import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from eval import evaluate
from retinanet import RetinaNet_efficientnet_b4

def train(args):
    train_csv = args.train_csv
    test_csv  = args.test_csv
    labels_csv = args.labels_csv
    model_type = args.model_type
    epochs     = int(args.epochs)
    batch_size = int(args.batch_size)
    
    dataset_train = CSVDataset(train_file=train_csv, class_list=labels_csv, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CSVDataset(train_file=test_csv, class_list=labels_csv, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    
    retinanet = RetinaNet_efficientnet_b4(num_classes=dataset_train.num_classes(), model_type=model_type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retinanet = retinanet.to(device)
    
    retinanet = torch.nn.DataParallel(retinanet).to(device)
    retinanet.training = True
    
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    loss_hist = collections.deque(maxlen=500)
    
    retinanet.train()
    retinanet.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))
    
    for epoch_num in range(epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                #classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss, regression_loss = retinanet([data['img'].to(device).float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | \
                      Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num,
                                                                               float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        #mAP, MAP  = evaluate(dataset_val, retinanet)
        _, MAP  = evaluate(dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))	
        torch.save(retinanet.module, '{}_retinanet_{}_map{}.pt'.format("EfficientNet" +model_type, 
                                                                       epoch_num, MAP))
        retinanet.eval()
        torch.save(retinanet, 'model_final.pt')
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', help='Path to train csv')
    parser.add_argument('test_csv', help='Path to test csv')
    parser.add_argument('labels_csv', help='Path to class labels')
    parser.add_argument('model_type', help='EfficientNet model type, \
                        must be one of ["b0", "b1", "b2", "b3", "b4", "b5"]', default="b4")
    parser.add_argument('epochs', help='Number of epochs for training', type=int, default=100)
    parser.add_argument('batch_size', help='Batch size for training', type=int, default=1)
    args = parser.parse_args()
    
    train(args)
