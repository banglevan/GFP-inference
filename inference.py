import argparse
import cv2
import glob

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from config import config

class ImageSRbyGFP:
    def __init__(self):
        self.device = 'cuda'
        self.configurations()
        self.load_model()

        config['aligned'] = False
        config['only_center_face'] = True
        config['suffix'] = None

    def prediction(self, image: np.ndarray):
        #BGR image read by cv2
        restored_img = self.restorer.enhance_img(
            image,
            has_aligned=config['aligned'],
            only_center_face=config['only_center_face'],
            paste_back=True,
            weight=config['weight'])

        plt.imshow(restored_img)
        plt.show()

        return restored_img

    def load_model(self):
        if config['bg_upsampler'] == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                self.bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=config['bg_tile'],
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            self.bg_upsampler = None

        self.restorer = GFPGANer(
            model_path=self.model_path,
            upscale=config['upscale'],
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=self.bg_upsampler)

    def configurations(self):
        if config['version'] == '1':
            self.arch = 'original'
            self.channel_multiplier = 1
            self.model_name = 'GFPGANv1'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'

        elif config['version']== '1.2':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANCleanv1-NoCE-C2'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'

        elif config['version'] == '1.3':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.3'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

        elif config['version'] == '1.4':
            self.arch = 'clean'
            self.channel_multiplier = 2
            self.model_name = 'GFPGANv1.4'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

        elif config['version'] == 'RestoreFormer':
            self.arch = 'RestoreFormer'
            self.channel_multiplier = 2
            self.model_name = 'RestoreFormer'
            self.url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'

        else:
            raise ValueError(f"Wrong model version {config['version']}.")

        self.model_path = os.path.join('gfpgan/weights', self.model_name + '.pth')
        if not os.path.isfile(self.model_path):
            self.model_path = os.path.join('gfpgan/weights', self.model_name + '.pth')

        if not os.path.isfile(self.model_path):
            # download pre-trained models from url
            self.model_path = self.url


if __name__ == '__main__':
    predictor = ImageSRbyGFP()
    image = cv2.imread(r'C:\BANGLV\GFPGAN_infer\data\ex2.jpg')
    predictor.prediction(image)