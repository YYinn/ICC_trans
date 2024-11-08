import logging
import time
import pandas as pd
import numpy as np
import torch

#---loss
from loss.loss import *
from utils.utils import *
#---model
from models.mymodel import SwinTrans
from models.mymodel1_4m import mymodel
from models.swinunetr import SwinUNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.dataloader import get_loader
from train_one_epoch import train_epoch, val_epoch, test_epoch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_ensem_result(pred, label):

    bi_pred = pred.copy()
    bi_pred[bi_pred > 0.5] = 1
    bi_pred[bi_pred <= 0.5] = 0

    acc = accuracy_score(label, bi_pred)
    f1 = f1_score(label, bi_pred, zero_division=1)
    pre = precision_score(label, bi_pred, zero_division=1)
    recall = recall_score(label, bi_pred, zero_division=1)
    auc = roc_auc_score(label, pred)

    return acc, f1, pre, recall, auc

def initmodel(args, fold):

    #######################################################
    ###################### model ##########################
    #######################################################
    if args.model_name == 'transformer':
        # model = SwinTrans(img_size=(args.resample[0], args.resample[1], args.resample[2]),
        #                 in_channels=args.in_channels,
        #                 out_channels=args.out_channels,
        #                 feature_size=args.feature_size,
        #                 use_checkpoint=args.use_checkpoint,
        #                 multi_mod_use = args.multi_use
        #                 )
        model = SwinUNETR(img_size=(args.resample[0], args.resample[1], args.resample[2]),
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        feature_size=args.feature_size,
                        use_checkpoint=args.use_checkpoint,
                        )
        logging.info(f'Model:SwinTrans input_channel : {args.in_channels}')
    elif args.model_name == 'cnn':
        model = mymodel(args.in_channels)
        logging.info('Model:mymodel single mod')
    else:
        raise('Wrong model name')
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters count, {pytorch_total_params}')

    model.cuda()
    model.eval()

    #########################################################
    ##################### use check point ###################
    #########################################################

    pretrained_dir = f'{args.pretrained_dir}/model/best_model_{fold}_{args.test_indicator}.pt'

    checkpoint = torch.load(pretrained_dir, map_location='cpu')
    print(checkpoint.keys())
    if 'state_dict' in checkpoint:
        model_dict = torch.load(pretrained_dir)["state_dict"]
        model.load_state_dict(model_dict)
    else:
        model_dict = torch.load(pretrained_dir)
        model.load_state_dict(model_dict)

    if 'best_acc' in checkpoint:
        best_acc = checkpoint['best_acc']
        e =  checkpoint['epoch']
        logging.info("=> fold '{}' loaded checkpoint '{}' (bestacc {} in epoch {})".format(fold, pretrained_dir, best_acc, e))
    elif 'best_auc' in checkpoint:
        e =  checkpoint['epoch']
        best_auc = checkpoint['best_auc']
        logging.info("=> fold '{}' loaded checkpoint '{}' (bestauc {} in epoch {})".format(fold, pretrained_dir, best_auc, e))
    else:
        logging.info("=> fold '{}' loaing checkpoint '{}'".format(fold, pretrained_dir))


    return model

def save_result(path, label, predict, fold, phase, args):
    path_simple = path.copy()
    for i in range(path.shape[0]):
        path_simple[i] = path[i].split('/')[-2]
    path_simple = np.expand_dims(path_simple, 1)
    label = np.expand_dims(label, 1)
    predict = np.expand_dims(predict, 1)
    print(label.shape, path_simple.shape, predict.shape )
    
    info = np.hstack((path_simple, label, predict))
    info_df = pd.DataFrame(info, columns=['path', 'label', 'predict'])
    info_df.to_csv(os.path.join(args.logdir, f'{phase}_fold{fold}.csv'), index=False)
    print(info.shape)

def run_test(args):
    logging.info('############################## Start Testing ######################################')
    avg_val_acc = 0.
    avg_val_auc = 0.
    avg_inter_test_acc = 0.
    avg_inter_test_auc = 0.
    avg_ex_test_acc = 0.
    avg_ex_test_auc = 0.

    avg_inter_pred = []
    avg_exter_pred = []
    for fold in range(0, 5):
        #############################################################################
        ######################### Step 1. Initialize model ##########################
        #############################################################################
        model = initmodel(args, fold)

        #############################################################################
        ############################ Step 2. Load Image #############################
        #############################################################################
        
        #---------- B. loader all ####################### 
        train_loader, val_loader, in_test_loader, ex_test_loader = get_loader(args, fold)

        logging.info(f'val dataset {len(val_loader)}')
        logging.info(f'internal test dataset {len(in_test_loader)}')
        logging.info(f'external test dataset {len(ex_test_loader)}')


        ################################################################
        ############################ VAL ###############################
        train_acc, train_f1, train_pre, train_recall, train_auc, train_pred, train_label, train_path = test_epoch(model,
                                                                    train_loader,
                                                                    args=args,
                                                                    fold=fold,
                                                                    multi_use=args.multi_use
                                                                    )################################################################

        val_acc, val_f1, val_pre, val_recall, val_auc, val_pred ,val_label, val_path = test_epoch(model,
                                                                    val_loader,
                                                                    args=args,
                                                                    fold=fold,
                                                                    multi_use=args.multi_use
                                                                    )
        logging.info(f'🟢 val : fold {fold}, | accuracy : {val_acc:5.4f} | auc : {val_auc:5.4f} | f1 : {val_f1:5.4f} | pre : {val_pre:5.4f} | recall : {val_recall:5.4f}')

        intertest_acc, intertest_f1, intertest_pre, intertest_recall, intertest_auc, intertest_pred, intertest_label, intertest_path = test_epoch(model,
                                                                    in_test_loader,
                                                                    args=args,
                                                                    fold=fold,
                                                                    multi_use=args.multi_use
                                                                    )
        logging.info(f'🟢 intertest : fold {fold}, | accuracy : {intertest_acc:5.4f} | auc : {intertest_auc:5.4f} | f1 : {intertest_f1:5.4f} | pre : {intertest_pre:5.4f} | recall : {intertest_recall:5.4f}')

        extertest_acc, extertest_f1, extertest_pre, extertest_recall, extertest_auc, extertest_pred, extertest_label, extertest_path = test_epoch(model,
                                                                    ex_test_loader,
                                                                    args=args,
                                                                    fold=fold,
                                                                    multi_use=args.multi_use
                                                                    )
        logging.info(f'🟢 extertest : fold {fold}, | accuracy : {extertest_acc:5.4f} | auc : {extertest_auc:5.4f} | f1 : {extertest_f1:5.4f} | pre : {extertest_pre:5.4f} | recall : {extertest_recall:5.4f}')

        save_result(train_path, train_label, train_pred, fold, 'train', args)
        save_result(val_path, val_label, val_pred, fold, 'val', args)
        save_result(intertest_path, intertest_label, intertest_pred, fold, 'internal_test', args)
        save_result(extertest_path, extertest_label, extertest_pred, fold, 'external_test', args)

        avg_val_acc += val_acc
        avg_val_auc += val_auc
        avg_inter_test_acc += intertest_acc
        avg_inter_test_auc += intertest_auc
        avg_ex_test_acc += extertest_acc
        avg_ex_test_auc += extertest_auc

        avg_inter_pred.append(intertest_pred)
        avg_exter_pred.append(extertest_pred)

    avg_val_acc /= 5
    avg_val_auc /= 5
    avg_inter_test_acc /= 5
    avg_inter_test_auc /= 5
    avg_ex_test_acc /= 5
    avg_ex_test_auc /= 5

    logging.info('======================== average of 5 folds =======================')
    logging.info(f'🟢 avg_val : accuracy : {avg_val_acc} | auc : {avg_val_auc}')
    logging.info(f'🟢 avg_intertest : accuracy : {avg_inter_test_acc} | auc : {avg_inter_test_auc}')
    logging.info(f'🟢 avg_extertest : accuracy : {avg_ex_test_acc} | auc : {avg_ex_test_auc}')

    logging.info('======================== ensemble of 5 folds =======================')
    avg_inter_pred = np.array(avg_inter_pred)
    avg_inter_pred = avg_inter_pred.mean(0)
    ensem_intertest_acc, ensem_intertest_f1, ensem_intertest_pre, ensem_intertest_recall, ensem_intertest_auc = get_ensem_result(avg_inter_pred, intertest_label)
    save_result(intertest_path, intertest_label, avg_inter_pred, 'ensem', 'ensemble_internal_test', args)
    logging.info(f'🟢 ensemble_intertest : accuracy : {ensem_intertest_acc:5.4f} | auc : {ensem_intertest_auc:5.4f} | f1 : {ensem_intertest_f1:5.4f} | pre : {ensem_intertest_pre:5.4f} | recall : {ensem_intertest_recall:5.4f}')


    avg_exter_pred = np.array(avg_exter_pred)
    avg_exter_pred = avg_exter_pred.mean(0)
    ensem_extertest_acc, ensem_extertest_f1, ensem_extertest_pre, ensem_extertest_recall, ensem_extertest_auc = get_ensem_result(avg_exter_pred, extertest_label)
    save_result(extertest_path, extertest_label, avg_exter_pred, 'ensem', 'ensemble_external_test', args)
    logging.info(f'🟢 ensemble_extertest : accuracy : {ensem_extertest_acc:5.4f} | auc : {ensem_extertest_auc:5.4f} | f1 : {ensem_extertest_f1:5.4f} | pre : {ensem_extertest_pre:5.4f} | recall : {ensem_extertest_recall:5.4f}')




    