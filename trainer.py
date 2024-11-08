import logging
import random
import time

import numpy as np
import torch

#---loss
from loss.loss import *
#---model
from models.mymodel import SwinTrans
from models.mymodel1_4m import mymodel
from models.swinunetr import SwinUNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from train_one_epoch import train_epoch, val_epoch
from utils.dataloader import get_loader
from utils.utils import *


def initmodel(args):

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
    
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters count, {pytorch_total_params}')

    model.cuda()

    #########################################################
    ###################### find nan #########################
    #########################################################
    # def nan_hook(self, inp, output):
    #     if not isinstance(output, tuple):
    #         outputs = [output]
    #     else:
    #         outputs = output

    #     for i, out in enumerate(outputs):
    #         if isinstance(out, list):
    #             for g in out:
    #                 nan_mask = torch.isnan(g)
    #                 if nan_mask.any():
    #                     print("In", self.__class__.__name__)
    #                     raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    #         else:
    #             nan_mask = torch.isnan(out)
    #             if nan_mask.any():
    #                 print("In", self.__class__.__name__)
    #                 raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    # for submodule in model.modules():
    #     submodule.register_forward_hook(nan_hook)

    #########################################################
    ##################### use check point ###################
    #########################################################
    start_epoch = 0
    # if args.resume_ckpt:
    #     pretrained_dir = args.pretrained_dir

    #     checkpoint = torch.load(pretrained_dir, map_location='cpu')
    #     if 'state_dict' in checkpoint:
    #         model_dict = torch.load(pretrained_dir)["state_dict"]
    #         model.load_state_dict(model_dict)
    #     else:
    #         model_dict = torch.load(pretrained_dir)
    #         model.load_state_dict(model_dict)

    #     if 'epoch' in checkpoint:
    #         start_epoch = checkpoint['epoch']
    #     if 'best_acc' in checkpoint:
    #         best_acc = checkpoint['best_acc']
    #         print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
    #     if 'best_auc' in checkpoint:
    #         best_auc = checkpoint['best_auc']
    #         print("=> loaded checkpoint '{}' (epoch {}) (bestauc {})".format(args.checkpoint, start_epoch, best_auc))

    #     logging.info(f'Using pretrained weights {pretrained_dir}')

    #########################################################
    ######################## loss ###########################
    #########################################################
    criterion1 = BCEFocalLoss()
    criterion2 = FocalLoss()
    
    #########################################################
    ####################### optimizer #######################
    #########################################################
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     eps=1e-4,
                                     weight_decay=args.reg_weight)
        logging.info('optimizer : adam')
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      eps=1e-4,
                                      weight_decay=args.reg_weight)
        logging.info('optimizer : adamw')
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
        logging.info('optimizer : SGD')
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    #########################################################
    ###################### lrschedule #######################
    #########################################################
    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
        logging.info('scheduler : warmup_cosine')
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        logging.info('scheduler : cosine_anneal')
        # if args.checkpoint is not None:
        #     scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    
    return model, optimizer, criterion1, criterion2, scheduler, start_epoch

def seed_torch(seed=45):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def run_training(args, tensorboard_writer = None):
    # seed_torch()
    logging.info('############################## Start Training ######################################')
    total_best_auc = []
    for fold in range(5):
        best_acc = 0
        best_auc = 0
        # intest_atbestauc = 0
        # extest_atbestauc = 0
        min_loss = None 
        es = 0
        #############################################################################
        ######################### Step 1. Initialize model ##########################
        #############################################################################
        model, optimizer, loss_func1, loss_func2, scheduler, start_epoch = initmodel(args)

        #############################################################################
        ############################ Step 2. Load Image #############################
        #############################################################################
        #---------- A. loader per mod
        # mod1_loader, mod2_loader, mod3_loader, mod4_loader, mod5_loader, mod6_loader, mod7_loader= get_loader(args, fold)
        # for epoch in range(start_epoch, args.max_epochs):
            
        #     for train_loader in (mod1_loader, mod2_loader, mod3_loader, mod4_loader, mod5_loader, mod6_loader, mod7_loader):
        #         if args.distributed:
        #             train_loader.sampler.set_epoch(epoch)
        #             torch.distributed.barrier()
        #     print(args.rank, time.ctime(), 'Epoch:', epoch)
        #     epoch_time = time.time()

        #     train_loader = [mod1_loader, mod2_loader, mod3_loader, mod4_loader, mod5_loader, mod6_loader, mod7_loader]

        #---------- B. loader all ####################### 
        train_loader, val_loader, in_test_loader, ex_test_loader = get_loader(args, fold)

        # if fold <3:
        #     continue

        for epoch in range(start_epoch, args.max_epochs):

            ################################################################
            ########################### TRAIN ##############################
            ################################################################
            train_loss, train_acc, train_f1, train_pre, train_recall, train_auc = train_epoch(model,
                                                                                            train_loader,
                                                                                            optimizer,
                                                                                            epoch=epoch,
                                                                                            loss_func1=loss_func1,
                                                                                            loss_func2=loss_func2,
                                                                                            args=args,
                                                                                            fold=fold,
                                                                                            multi_use=args.multi_use)
            logging.info(f'🟢 training : fold {fold}, epoch {epoch}, loss : {train_loss:5.4f} | accuracy : {train_acc:5.4f} | \
            f1 : {train_f1:5.4f} | pre : {train_pre:5.4f} | recall : {train_recall:5.4f} | auc : {train_auc:5.4f}')
        
            tensorboard_writer.add_scalar(f'train/loss_fold{fold}', train_loss, epoch)
            tensorboard_writer.add_scalar(f'train/acc_fold{fold}', train_acc, epoch)
            tensorboard_writer.add_scalar(f'train/f1_fold{fold}', train_f1, epoch)
            tensorboard_writer.add_scalar(f'train/pre_fold{fold}', train_pre, epoch)
            tensorboard_writer.add_scalar(f'train/rec_fold{fold}', train_recall, epoch)
            tensorboard_writer.add_scalar(f'train/auc_fold{fold}', train_auc, epoch)
            # for name, param in model.named_parameters():  # 返回网络的
            #     if 'NoneType' not in str(type(param.grad)):
            #         tensorboard_writer.add_histogram(name + '_grad', param.grad, epoch)
            #     # tensorboard_writer.add_histogram(name + '_data', param, epoch)

            ################################################################
            ############################ VAL ###############################
            ################################################################
            if (epoch+1) % args.val_every == 0:
                val_loss, val_acc, val_f1, val_pre, val_recall, val_auc  = val_epoch(model,
                                                                                    val_loader,
                                                                                    epoch=epoch,
                                                                                    loss_func1=loss_func1,
                                                                                    loss_func2=loss_func2,
                                                                                    args=args,
                                                                                    fold=fold,
                                                                                    multi_use=args.multi_use
                                                                                    )
                logging.info(f'🟢 val : fold {fold}, epoch {epoch}, loss : {val_loss:5.4f} | accuracy : {val_acc:5.4f} | \
                f1 : {val_f1:5.4f} | pre : {val_pre:5.4f} | recall : {val_recall:5.4f} | auc : {val_auc:5.4f}')
            
                tensorboard_writer.add_scalar(f'val/loss_fold{fold}', val_loss, epoch)
                tensorboard_writer.add_scalar(f'val/acc_fold{fold}', val_acc, epoch)
                tensorboard_writer.add_scalar(f'val/f1_fold{fold}', val_f1, epoch)
                tensorboard_writer.add_scalar(f'val/pre_fold{fold}', val_pre, epoch)
                tensorboard_writer.add_scalar(f'val/rec_fold{fold}', val_recall, epoch)
                tensorboard_writer.add_scalar(f'val/auc_fold{fold}', val_auc, epoch)

                if(val_acc > best_acc):
                    best_acc = val_acc
                    # torch.save(model.state_dict(), f'{args.logdir}best_model_{fold}_acc.pth')
                    logging.info(f'✅ get better acc {best_acc}')
                    save_acc_checkpoint(model, epoch, args, best_acc=best_acc, filename=f'best_model_{fold}_acc.pt')

                if(val_auc > best_auc):
                    best_auc = val_auc
                    # torch.save(model.state_dict(), f'{args.logdir}best_model_{fold}_auc.pth')
                    logging.info(f'✅ get better auc {best_auc}')
                    save_auc_checkpoint(model, epoch, args, best_auc=best_auc, filename=f'best_model_{fold}_auc.pt')

                    # in_test_loss, in_test_acc, in_test_f1, in_test_pre, in_test_recall, in_test_auc  = val_epoch(model,
                    #                                                                 in_test_loader,
                    #                                                                 epoch=epoch,
                    #                                                                 loss_func1=loss_func1,
                    #                                                                 loss_func2=loss_func2,
                    #                                                                 args=args,
                    #                                                                 fold=fold,
                    #                                                                 multi_use=args.multi_use
                    #                                                                 )
                    # logging.info(f'🟢 in-test :                auc : {in_test_auc:5.4f} | accuracy : {in_test_acc:5.4f} | \
                    # f1 : {in_test_f1:5.4f} | pre : {in_test_pre:5.4f} | recall : {in_test_recall:5.4f} ')
                    # intest_atbestauc = in_test_auc

                    # ex_test_loss, ex_test_acc, ex_test_f1, ex_test_pre, ex_test_recall, ex_test_auc  = val_epoch(model,
                    #                                                                 ex_test_loader,
                    #                                                                 epoch=epoch,
                    #                                                                 loss_func1=loss_func1,
                    #                                                                 loss_func2=loss_func2,
                    #                                                                 args=args,
                    #                                                                 fold=fold,
                    #                                                                 multi_use=args.multi_use
                    #                                                                 )
                    # logging.info(f'🟢 ex-test :                auc : {ex_test_auc:5.4f} | accuracy : {ex_test_acc:5.4f} | \
                    # f1 : {ex_test_f1:5.4f} | pre : {ex_test_pre:5.4f} | recall : {ex_test_recall:5.4f} ')
                    # extest_atbestauc = ex_test_auc

                if min_loss is None or val_loss < min_loss:
                    min_loss = val_loss
                    es = 0
                else: 
                    es += 1
                logging.info(f'early stop {es} / {args.es}')
                if es >= args.es :
                    logging.info('Early stop')
                    total_best_auc.append(best_auc)
                    break
            
            torch.save(model.state_dict(), f'{args.logdir}/latest_model.pth')

            if scheduler is not None:
                scheduler.step()
        total_best_auc.append(best_auc)
    logging.info(f'Training Finished !, Best AUC : {total_best_auc}') 

