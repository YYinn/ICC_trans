import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.transforms import AddChannel, Compose, RandAffine, RandRotate90, RandFlip, apply_transform, ToTensor, RandAdjustContrast, RandShiftIntensity

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


def get_loader(args, fold=0):

    train_img_list, train_label_list, val_img_list, val_label_list = json_reader(json_path=args.train_json_list, fold=fold)

    train_ds = Datareader(train_img_list, train_label_list, transform='train', resample=args.resample)
    valid_ds = Datareader(val_img_list, val_label_list, transform='valid', resample=args.resample)

    train_data = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_torch(args.seed))
    valid_data = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_torch(args.seed))   

    in_test_img_list, in_test_label_list = json_reader_test(json_path=args.intest_json_list)
    in_test_ds = Datareader(in_test_img_list, in_test_label_list, transform='valid', resample=args.resample)
    in_test_data = DataLoader(in_test_ds, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_torch(args.seed))

    ex_test_img_list, ex_test_label_list = json_reader_test(json_path=args.extest_json_list)
    ex_test_ds = Datareader(ex_test_img_list, ex_test_label_list, transform='valid', resample=args.resample)
    ex_test_data = DataLoader(ex_test_ds, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_torch(args.seed))

    return train_data, valid_data, in_test_data, ex_test_data

def zscore_normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std

def normalization(data):

    return (data - data.min()) / (data.max() - data.min())

def resample_to_size(npy_image, target_size, order=1):
    source_size = npy_image.shape
    scale = np.array(target_size) / source_size
    # zoom_factor = source_size / np.array(target_size)
    target_npy_image = zoom(npy_image, scale, order=order)
    return target_npy_image

def preprocess_img(img, transform=None, resample=None):

    img = normalization(img)
    # img = zscore_normalize(img)

    if resample is not None:
        img = resample_to_size(img, resample)

    if transform:
        trans_img = apply_transform(transform, img.astype(np.float32))
        trans_img = torch.tensor(trans_img)
    else:
        trans_img = torch.tensor(img)
        trans_img = trans_img.unsqueeze(dim=0)
    return trans_img


train_transforms = Compose(
    [
    AddChannel(), 
    # RandAffine(prob=0.1, translate_range=(4, 10, 10), padding_mode="border", as_tensor_output=False),
    RandFlip(prob=0.2), 
    RandRotate90(prob=0.2, spatial_axes=(1,2))], 
    RandAdjustContrast(prob=0.2),
    RandShiftIntensity(offsets=0.2, prob=0.1)
    )


val_transforms = Compose(
    AddChannel()
    )



class Datareader(Dataset):
    def __init__(self, img_list, label_list, transform=None, resample=None):
        self.img_list = img_list 
        self.label_list = label_list
        if transform == 'train':
            self.transform = train_transforms
        else:
            self.transform = val_transforms
        self.resample = resample

    def __getitem__(self, item):
        multi_img = []
        for mod_idx in range(7):
            img = np.load(self.img_list[item][mod_idx])

            if len(img.shape) == 4:
                img = img[..., 0]
            # print(img.shape)
        
            img = preprocess_img(img, self.transform, self.resample) 
            multi_img.append(img)
        multi_img = torch.tensor(np.concatenate(multi_img))
        # print(multi_img.shape) #[7, 32, 128, 128])

        label = self.label_list[item]
        label = torch.tensor(label)

        return multi_img, label.float(), self.img_list[item][0]
    def __len__(self):
        return len(self.label_list)



def json_reader(json_path, fold=0, key='training'):
    with open(json_path) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    train_img_list = []
    train_label_list = []

    val_img_list = []
    val_label_list = []

    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val_img_list.append(d['img_path'])
            val_label_list.append(d['label'])
        else:
            train_img_list.append(d['img_path'])
            train_label_list.append(d['label'])

    print('train len', len(train_label_list), 'val len', len(val_label_list))
    val_img_list = np.array(val_img_list)
    val_label_list = np.array(val_label_list)
    
    train_img_list = np.array(train_img_list)
    train_label_list = np.array(train_label_list)
    
    return train_img_list, train_label_list, val_img_list, val_label_list


def json_reader_test(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    json_data = json_data['test']
    test_img_list = []
    test_label_list = []

    for d in json_data:
        test_img_list.append(d['img_path'])
        test_label_list.append(d['label'])

    print('test len', len(test_img_list))
    test_img_list = np.array(test_img_list)
    test_label_list = np.array(test_label_list)
    
    return test_img_list, test_label_list