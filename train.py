# Base
import time
import os
import copy
import argparse
import pdb
import collections
import sys
import numpy as np
import os.path as osp

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

# Modules
import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

# Evaluation
import coco_eval
import csv_eval

#assert torch.__version__.split('.')[1] == '4'
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--optimizer', help='[SGD | Adam]', type=str, default='SGD')
    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser = parser.parse_args(args)

    # Create the data loaders
    print("\n[Phase 1]: Creating DataLoader for {} dataset".format(parser.dataset))
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2014', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2014', transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=16, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    print('| Num training images: {}'.format(len(dataset_train)))
    print('| Num test images : {}'.format(len(dataset_val)))

    print("\n[Phase 2]: Preparing RetinaNet Detection Model...")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
        retinanet = retinanet.to(device)

    retinanet = torch.nn.DataParallel(retinanet, device_ids=range(torch.cuda.device_count()))
    print("| Using %d GPUs for Train/Validation!" %torch.cuda.device_count())
    retinanet.training = True

    if parser.optimizer == 'Adam':
        optimizer = optim.Adam(retinanet.parameters(), lr=1e-5) # not mentioned
        print("| Adam Optimizer with Learning Rate = {}".format(1e-5))
    elif parser.optimizer == 'SGD':
        optimizer = optim.SGD(retinanet.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        print("| SGD Optimizer with Learning Rate = {}".format(1e-2))
    else:
        raise ValueError('Unsupported Optimizer, must be one of [SGD | Adam]')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn() # Freeze the BN parameters to ImageNet configuration

    # Check if there is a 'checkpoints' path
    if not osp.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')

    print("\n[Phase 3]: Training Model on {} dataset...".format(parser.dataset))
    for epoch_num in range(parser.epochs):
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].to(device), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.001)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num+1, iter_num+1, len(dataloader_train), float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                sys.stdout.flush()

                del classification_loss
                del regression_loss

            except Exception as e:
                print(e)
                continue

        print("\n| Saving current best model at epoch {}...".format(epoch_num+1))
        torch.save(retinanet.state_dict(), './checkpoints/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num+1))

        if parser.dataset == 'coco':
            #print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet, device)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            #print('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet, device)

        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()
    torch.save(retinanet.state_dict(), './checkpoints/model_final.pt')

if __name__ == '__main__':
    main()
