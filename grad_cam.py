import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_cam import GradCAM, show_cam_on_image, center_crop_img

from models.swinunetr import SwinUNETR
def main():
    model = SwinUNETR(img_size=(32, 64, 64),
                        in_channels=7,
                        out_channels=1,
                        feature_size=48,
                        use_checkpoint='/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/model/best_model_0_acc.pt',
                        )
    
    # load image
    # img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2024-12-17T21:30:40/train_img_X1659.npy')[0, ...] ## 直接读取(2, 7, 32, 64, 64)
    # img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2024-12-17T21:30:40/train_img_X224G967.npy')[0, ...]
    # img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/10.npy')
    # img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/136.npy')
    img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/87.npy')
    # img = np.load('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/140.npy')
    img = torch.tensor(img)
    # print(img.shape)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img, dim=0)

    cam = GradCAM(model=model, target_layers=[model.decoder1.conv_block.norm3], use_cuda=False)


    grayscale_cam = cam(input_tensor=input_tensor, target_category=0)


    grayscale_cam = grayscale_cam[0, :]
    # np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2024-12-17T21:30:40/X16_cam.npy', grayscale_cam)
    # np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2024-12-17T21:30:40/X224_cam.npy', grayscale_cam)
    # np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/10_cam.npy', grayscale_cam)
    # np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/136_cam.npy', grayscale_cam)
    np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/87_cam.npy', grayscale_cam)
    # np.save('/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34/test_bestauc_2025-08-14T20:13:19/140_cam.npy', grayscale_cam)
    
    # visualization = show_cam_on_image(img / 255.,
    #                                   grayscale_cam,
    #                                   use_rgb=True)
    # plt.imshow(visualization)
    # # plt.show()
    # plt.savefig('test.png')
    # plt.close()


if __name__ == '__main__':
    main()