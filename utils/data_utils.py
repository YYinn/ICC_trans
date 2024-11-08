# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import random
from torch.utils.data.sampler import SubsetRandomSampler
import logging

def datafold_read(datalist,
                  basedir,
                  fold=0,
                  key='training'):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr=[]
    val=[]
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

#------------set random seed-----------------------
def seed_torch(seed=45):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(147)+worker_id)
    seed_torch(seed=147)
    pass


    ###############################################################################
    ############################# dataloader per mods #############################
    ###############################################################################
    # total_loader = []
    # for mod in args.mods:
    #     datalist_json = f'{args.json_list}json_mod_{mod}.json'
    #     train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)

    #     print(f'Mod : {mod}, training data : {len(train_files)}, validate data : {len(validation_files)}')
        
    #     if args.test_mode:

    #         val_ds = data.Dataset(data=validation_files, transform=test_transform)
    #         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    #         test_loader = data.DataLoader(val_ds,
    #                                     batch_size=1,
    #                                     shuffle=False,
    #                                     num_workers=args.workers,
    #                                     sampler=val_sampler,
    #                                     pin_memory=True,
    #                                     )

    #         loader = test_loader
    #     else:
    #         train_ds = data.Dataset(data=train_files, transform=train_transform)
            
    #         # train_sampler = Sampler(train_ds) if args.distributed else None
    #         train_sampler = SubsetRandomSampler(train_ds)
    #         train_loader = data.DataLoader(train_ds,
    #                                     batch_size=args.batch_size,
    #                                     shuffle=(train_sampler is None),
    #                                     num_workers=args.workers,
    #                                     sampler=train_sampler,
    #                                     pin_memory=True,
    #                                     worker_init_fn=_init_fn
    #                                     )
    #         val_ds = data.Dataset(data=validation_files, transform=val_transform)
    #         # val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    #         val_sampler = SubsetRandomSampler(val_ds)
    #         val_loader = data.DataLoader(val_ds,
    #                                     batch_size=1,
    #                                     shuffle=False,
    #                                     num_workers=args.workers,
    #                                     sampler=val_sampler,
    #                                     pin_memory=True,
    #                                     )
    #         loader = [train_loader, val_loader]
    #     total_loader.append(loader)
    # # total_loader = np.array(total_loader)
    # return total_loader[0], total_loader[1], total_loader[2], total_loader[3], total_loader[4], total_loader[5], total_loader[6]

    ###############################################################################
    ############################# dataloader all ##################################
    ###############################################################################

    datalist_json = args.json_list 
    logging.info(datalist_json)
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)

    if args.test_mode:

        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(val_ds,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    sampler=val_sampler,
                                    pin_memory=True,
                                    )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(train_ds,
                                    batch_size=args.batch_size,
                                    shuffle=(train_sampler is None),
                                    num_workers=args.workers,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    )
                                    
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    sampler=val_sampler,
                                    pin_memory=True,
                                    )
        loader = [train_loader, val_loader]


    print('data_utils', len(train_loader), len(val_loader))

    return loader


# from torch.utils.data import Dataset

# class TrainDataLoader(Dataset):
#     def __init__(self, args, fold):
#         self.total_loader = get_loader(args, fold)

#     def __getitem__(self, item):
#         return [self.total_loader[i][0][item] for i in range(7)]

#     def __len__(self):
#         return len(self.total_loader[0][0])