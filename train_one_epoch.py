from copy import copy
import os
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
        # print(logits)

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
    total_feat = []
    total_img = []
    bar = tqdm.tqdm(loader, total=len(loader), postfix={'state' : 'testing', 'fold' : fold})
    with torch.no_grad():
        for idx, (img, label, path) in enumerate(bar):

            data, target = img.cuda(), label.cuda()

            if multi_use:
                logits, feat = model(data)
            else:
                logits, feat = model(data[:, args.mods:args.mods+1, ...])
            
            logits = torch.squeeze(logits, dim=-1)
            pred = copy(logits)
            pred = pred.detach().cpu()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            total_label.append(target.detach().cpu().numpy())
            total_predict.append(logits.detach().cpu())
            total_predict_binary.append(pred.numpy())
            total_path.append(path)
            
            ## visual onlt
            # if target.sum() == 1:
            # if 'X16/' in path[0] or 'X224/' in path[0] or 'X16/' in path[1] or 'X224/' in path[1]:
            # print(path)
            # breakpoint()
            # if '10/' in path or '136/' in path or '140/' in path:
            #     id = path[0].split('/')[-2] + path[1].split('/')[-2]
            #     print(id)
            #     print(target)
            #     print(logits)
            #     total_img = np.array(img)
            #     enc0, enc1, enc2, enc3, dec4, dec3, dec2, dec1, dec0, out, out2, out0 = feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10], feat[11]
            #     i = 0
            #     for feat_i in (enc0, enc1, enc2, enc3, dec4, dec3, dec2, dec1, dec0, out, out2, out0):
            #         np.save(os.path.join(args.logdir, f'train_feat{i}_{id}.npy'), feat_i.detach().cpu().numpy())
            #         i += 1
            #     np.save(os.path.join(args.logdir, f'train_img_{id}.npy'), total_img)
            #     tmp = os.path.join(args.logdir, f'train_img_{id}.npy')
            #     print(f'saving {tmp}')
                
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




def test_epoch_wolabel(model, loader, args, fold=0, multi_use=True):

    total_label = []
    total_predict = []
    total_predict_binary = []
    model.eval()
    total_path = []
    total_feat = []
    total_img = []
    bar = tqdm.tqdm(loader, total=len(loader), postfix={'state' : 'testing', 'fold' : fold})
    with torch.no_grad():
        for idx, (img, path) in enumerate(bar):

            data = img.cuda()

            if multi_use:
                logits, feat = model(data)
            else:
                logits, feat = model(data[:, args.mods:args.mods+1, ...])
            
            logits = torch.squeeze(logits, dim=-1)
            pred = copy(logits)
            pred = pred.detach().cpu()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            total_predict.append(logits.detach().cpu())
            total_predict_binary.append(pred.numpy())
            total_path.append(path)
            
            # print(path)
            # if  '87/' in path[0] or '87/' in path[1]:
            #     id = path[0].split('/')[-2] + path[1].split('/')[-2]
            #     print(id)
            #     print(logits)
            #     total_img = np.array(img)
            #     enc0, enc1, enc2, enc3, dec4, dec3, dec2, dec1, dec0, out, out2, out0 = feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10], feat[11]
            #     i = 0
            #     for feat_i in (enc0, enc1, enc2, enc3, dec4, dec3, dec2, dec1, dec0, out, out2, out0):
            #         np.save(os.path.join(args.logdir, f'train_feat{i}_{id}.npy'), feat_i.detach().cpu().numpy())
            #         i += 1
            #     np.save(os.path.join(args.logdir, f'train_img_{id}.npy'), total_img)
            #     tmp = os.path.join(args.logdir, f'train_img_{id}.npy')
            #     print(f'saving {tmp}')
        total_predict = np.concatenate(total_predict)
        total_predict_binary = np.concatenate(total_predict_binary)
        total_path = np.concatenate(total_path)       
       
    return total_predict, total_path