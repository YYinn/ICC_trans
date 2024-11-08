from copy import copy

import numpy as np
import torch
import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from utils.utils import AverageMeter


def train_epoch(model, loader, optimizer, epoch, loss_func1, loss_func2, args, fold=0, multi_use=True):

    total_label = []
    total_predict = []
    total_predict_binary = []

    model.train()
    run_loss = AverageMeter()
    bar = tqdm.tqdm(loader, total=len(loader), postfix={'state' : 'training', 'epoch' : epoch, 'fold' : fold, 'multi_mod' : multi_use})

    for idx, (img, label, path) in enumerate(bar):
        #-----loading image and label(target)
        data, target = img.cuda(), label.cuda()
        # print('data shape', data.shape)
        optimizer.zero_grad()

        if multi_use:
            logits = model(data)
        else:
            logits = model(data[:, args.mods:args.mods+1, ...])

        logits = torch.squeeze(logits, dim=-1)
        print(logits)

        loss = loss_func1(logits, target)# + loss_func2(logits, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)

        # binary 
        pred = copy(logits)
        pred = pred.detach().cpu()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        total_label.append(target.detach().cpu().numpy())
        total_predict.append(logits.detach().cpu())
        total_predict_binary.append(pred.numpy())

    
    total_predict = np.concatenate(total_predict)
    total_predict_binary = np.concatenate(total_predict_binary)
    total_label = np.concatenate(total_label)

    print(f'=== check train === {sum(total_predict_binary)} / {total_predict_binary.shape[0]}')
    # print(total_predict.shape, total_predict_binary.shape, total_label.shape) (292, 1) (292, 1) (292,)

    total_acc = accuracy_score(total_label, total_predict_binary)
    # print('train', total_label, total_predict, total_predict_binary)
    total_f1 = f1_score(total_label, total_predict_binary, zero_division=1)
    total_pre = precision_score(total_label, total_predict_binary, zero_division=1)
    total_recall = recall_score(total_label, total_predict_binary, zero_division=1)
    total_auc = roc_auc_score(total_label, total_predict)


    return run_loss.avg, total_acc, total_f1, total_pre, total_recall, total_auc


def val_epoch(model, loader, epoch, loss_func1, loss_func2, args, fold=0, multi_use=True):

    total_label = []
    total_predict = []
    total_predict_binary = []
    model.eval()
    run_loss = AverageMeter()
    bar = tqdm.tqdm(loader, total=len(loader), postfix={'state' : 'training', 'epoch' : epoch, 'fold' : fold})
    with torch.no_grad():
        for idx, (img, label, path) in enumerate(bar):
            data, target = img.cuda(), label.cuda()

            if multi_use:
                logits = model(data)
            else:
                logits = model(data[:, args.mods:args.mods+1, ...])
            
            logits = torch.squeeze(logits, dim=-1)

            loss = loss_func1(logits, target)# + loss_func2(logits, target)

            run_loss.update(loss.item(), n=args.batch_size)
            # binary 
            pred = copy(logits)
            pred = pred.detach().cpu()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            total_label.append(target.detach().cpu().numpy())
            total_predict.append(logits.detach().cpu())
            total_predict_binary.append(pred.numpy())


        
        total_predict = np.concatenate(total_predict)
        total_predict_binary = np.concatenate(total_predict_binary)
        total_label = np.concatenate(total_label)
        print(f'=== check val === {sum(total_predict_binary)} / {total_predict_binary.shape[0]}')
        # print(total_predict.shape, total_predict_binary.shape, total_label.shape) (292, 1) (292, 1) (292,)

        total_acc = accuracy_score(total_label, total_predict_binary)
        # print('train', total_label, total_predict, total_predict_binary)
        total_f1 = f1_score(total_label, total_predict_binary, zero_division=1)
        total_pre = precision_score(total_label, total_predict_binary, zero_division=1)
        total_recall = recall_score(total_label, total_predict_binary, zero_division=1)
        total_auc = roc_auc_score(total_label, total_predict)

    return run_loss.avg, total_acc, total_f1, total_pre, total_recall, total_auc




def test_epoch(model, loader, args, fold=0, multi_use=True):

    total_label = []
    total_predict = []
    total_predict_binary = []
    model.eval()
    total_path = []
    bar = tqdm.tqdm(loader, total=len(loader), postfix={'state' : 'testing', 'fold' : fold})
    with torch.no_grad():
        for idx, (img, label, path) in enumerate(bar):
            data, target = img.cuda(), label.cuda()

            if multi_use:
                logits = model(data)
            else:
                logits = model(data[:, args.mods:args.mods+1, ...])
            
            logits = torch.squeeze(logits, dim=-1)
            pred = copy(logits)
            pred = pred.detach().cpu()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            total_label.append(target.detach().cpu().numpy())
            total_predict.append(logits.detach().cpu())
            total_predict_binary.append(pred.numpy())
            total_path.append(path)

        total_predict = np.concatenate(total_predict)
        total_predict_binary = np.concatenate(total_predict_binary)
        total_label = np.concatenate(total_label)
        total_path = np.concatenate(total_path)
        # print(total_predict)
        # print(total_predict_binary)
        # print(total_label)
        # os._exit(0)
        # print(total_predict.shape, total_predict_binary.shape, total_label.shape) (292, 1) (292, 1) (292,)

        total_acc = accuracy_score(total_label, total_predict_binary)
        # print('train', total_label, total_predict, total_predict_binary)
        total_f1 = f1_score(total_label, total_predict_binary, zero_division=1)
        total_pre = precision_score(total_label, total_predict_binary, zero_division=1)
        total_recall = recall_score(total_label, total_predict_binary, zero_division=1)
        total_auc = roc_auc_score(total_label, total_predict)

    return total_acc, total_f1, total_pre, total_recall, total_auc, total_predict, total_label, total_path