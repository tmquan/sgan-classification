
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import random
from pprint import pprint
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import tensorpack.dataflow as df
from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData
from tensorpack.utils import get_rng
from tensorpack.utils.argtools import shape2d

import albumentations as AB
import sklearn.metrics
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Domain libraries
import torchxrayvision as xrv
import pretrainedmodels as cmp 

class MultiLabelDataset(df.RNGDataFlow):
    def __init__(self, folder, types=14, is_train='train', channel=1,
                 resize=None, debug=False, shuffle=False, pathology=None, 
                 fname='train.csv', balancing=None):

        self.version = "1.0.0"
        self.description = "Vinmec is a large dataset of chest X-rays\n",
        self.citation = "\n"
        self.folder = folder
        self.types = types
        self.is_train = is_train
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if self.channel == 1:
            self.imread_mode = cv2.IMREAD_GRAYSCALE
        else:
            self.imread_mode = cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.debug = debug
        self.shuffle = shuffle
        self.csvfile = os.path.join(self.folder, fname)
        print(self.folder)
        # Read the csv
        self.df = pd.read_csv(self.csvfile)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df = self.df.infer_objects()
        
        self.pathology = pathology
        self.balancing = balancing
        if self.balancing == 'up':
            self.df_majority = self.df[self.df[self.pathology]==0]
            self.df_minority = self.df[self.df[self.pathology]==1]
            print(self.df_majority[self.pathology].value_counts())
            self.df_minority_upsampled = resample(self.df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=self.df_majority[self.pathology].value_counts()[0],    # to match majority class
                                     random_state=123) # reproducible results

            self.df_upsampled = pd.concat([self.df_majority, self.df_minority_upsampled])
            self.df = self.df_upsampled
    def reset_state(self):
        self.rng = get_rng(self)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        indices = list(range(self.__len__()))
        if self.is_train == 'train':
            self.rng.shuffle(indices)

        for idx in indices:
            fpath = os.path.join(self.folder, 'data')
            fname = os.path.join(fpath, self.df.iloc[idx]['Images'])
            image = cv2.imread(fname, self.imread_mode)
            assert image is not None, fname
            # print('File {}, shape {}'.format(fname, image.shape))
            if self.channel == 3:
                image = image[:, :, ::-1]
            if self.resize is not None:
                image = cv2.resize(image, tuple(self.resize[::-1]))
            if self.channel == 1:
                image = image[:, :, np.newaxis]

            # Process the label
            if self.is_train == 'train' or self.is_train == 'valid':
                label = []
                if self.types == 6:
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Cardiomegaly'])
                    label.append(self.df.iloc[idx]['Fracture'])
                    label.append(self.df.iloc[idx]['Lung_Lesion'])
                    label.append(self.df.iloc[idx]['Pleural_Effusion'])
                    label.append(self.df.iloc[idx]['Pneumothorax'])
                if self.types == 4:
                    label.append(self.df.iloc[idx]['Covid'])
                    label.append(self.df.iloc[idx]['Airspace_Opacity'])
                    label.append(self.df.iloc[idx]['Consolidation'])
                    label.append(self.df.iloc[idx]['Pneumonia'])
                elif self.types == 2:
                    assert self.pathology is not None
                    label.append(self.df.iloc[idx]['No_Finding'])
                    label.append(self.df.iloc[idx][self.pathology])
                else:
                    pass
                # Try catch exception
                label = np.nan_to_num(label, copy=True, nan=0)
                label = np.array(label>0.16, dtype=np.float32)
                types = label.copy()
                yield [image, types]
            elif self.is_train == 'test':
                yield [image]  # , np.array([-1, -1, -1, -1, -1])
            else:
                pass

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def DiceScore(output, target, smooth=1.0, epsilon=1e-7, axis=(2, 3)):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        output (Numpy tensor): tensor of ground truth values for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        target (Numpy tensor): tensor of predictions for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    y_true = target
    y_pred = output
    dice_numerator = 2*np.sum(y_true*y_pred, axis=axis) + epsilon
    dice_denominator = (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + epsilon)
    dice_coefficient = np.mean(dice_numerator / dice_denominator)

    return dice_coefficient

class SoftDiceLoss(nn.Module):
    def init(self):
        super(SoftDiceLoss, self).init()

    def forward(self, output, target, smooth=1.0, epsilon=1e-7, axis=(1)):
        """
        Compute mean soft dice loss over all abnormality classes.

        Args:
            y_true (Torch tensor): tensor of ground truth values for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            y_pred (Torch tensor): tensor of soft predictions for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                          denominator in formula for dice loss.
                          Hint: pass this as the 'axis' argument to the K.sum
                                and K.mean functions.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_loss (float): computed value of dice loss.  
        """
        y_true = target
        y_pred = output
        dice_numerator = 2*torch.sum(y_true*y_pred, dim=axis) + epsilon
        dice_denominator = (torch.sum(y_true*y_true, dim=axis) + torch.sum(y_pred*y_pred, dim=axis) + epsilon)
        dice_coefficient = torch.mean(dice_numerator / dice_denominator)
        
        dice_loss = 1 - dice_coefficient
        return dice_loss

class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_size = self.hparams.shape // 16  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.hparams.latent_dim+self.hparams.types, 1024 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024), 
            nn.Conv2d(1024, 4*512, 3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512), 
            # nn.Upsample(scale_factor=2), 
            # nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(512, 4*256, 3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256), 
            # nn.Upsample(scale_factor=2), 
            # nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(256, 4*128, 3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2), 
            # nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(128, 4*64, 3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if self.hparams.arch.lower() == 'densenet121':
            self.discrim = getattr(models, self.hparams.arch)(
                pretrained=self.hparams.pretrained)
            self.discrim.features.conv0 = nn.Conv2d(1, 64, 
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.discrim.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Identity(),
                # nn.Linear(1024, self.hparams.types),  # 5 diseases
                # nn.Sigmoid(),
            )
            print(self.discrim)

            self.adv_layer = nn.Sequential(nn.Linear(1024, self.hparams.types), nn.Sigmoid())
            self.aux_layer = nn.Sequential(nn.Linear(1024, self.hparams.types), nn.Sigmoid())

        else:
            ValueError
    def forward(self, img):
        # out = self.conv_blocks(img)
        # out = out.view(out.shape[0], -1)
        out = self.discrim(img)
        pred = self.adv_layer(out)
        prob = self.aux_layer(out)

        return pred, prob

class SGAN(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.average_type = 'binary'
        # networks
        image_shape = (1, self.hparams.shape, self.hparams.shape)
        self.generator = Generator(self.hparams)
        self.discriminator = Discriminator(self.hparams)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # cache for generated images
        self.generated_image = None
        self.last_image = None
        self.adversarial_loss = SoftDiceLoss() #torch.nn.BCELoss()
        self.probability_loss = SoftDiceLoss() #torch.nn.BCELoss()

    def forward(self, x):
        true_or_fake, prob = self.discriminator(x)
        return prob
    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, label = batch
        image = image / 128.0 - 1.0
        self.last_image = image

        # p = torch.empty(label.shape).random_(2).type_as(image)
        p = torch.empty(label.shape).uniform_(0, 1).type_as(image)
        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(image.shape[0], self.hparams.latent_dim)
            z = z.type_as(image)
            z = torch.cat([z, p*2-1], dim=1)

            # generate images
            self.generated_image = self.generator(z)

            # log sampled images
            fake_image = self.generated_image[:16] / 2.0 + 0.5
            fake_grid = torchvision.utils.make_grid(fake_image)
            self.logger.experiment.add_image('fake_images', fake_grid, self.current_epoch)

            true_image = image[:16]
            true_grid = torchvision.utils.make_grid(true_image) / 2.0 + 0.5
            self.logger.experiment.add_image('true_images', true_grid, self.current_epoch)


            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            fake = torch.ones_like(p)
            fake = fake.type_as(image)

            # adversarial loss is binary cross-entropy
            pred, prob = self.discriminator(self.generated_image)
            g_loss = 10*(self.adversarial_loss(pred, fake) +  self.probability_loss(prob, p)) 
            # g_loss = 10*(-torch.mean(pred) +  self.probability_loss(prob, p)) 
            # g_loss = self.adversarial_loss(validity, valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            true = torch.ones_like(p)
            real_pred, real_prob = self.discriminator(image)
            real_loss = (self.adversarial_loss(real_pred, true) +  self.probability_loss(real_prob, label)) 
            # real_loss = (-torch.mean(real_pred) +  self.probability_loss(real_prob, label)) 

            # how well can it label as fake?
            fake = torch.zeros_like(p)
            fake_pred, fake_prob = self.discriminator(self.generated_image.detach())
            fake_loss = (self.adversarial_loss(fake_pred, fake) +  self.probability_loss(fake_prob, p)) 
            # fake_loss = (torch.mean(fake_pred) +  self.probability_loss(fake_prob, p)) 

            # for params in self.discriminator.parameters():
            #     params.data.clamp_(-0.01, 0.01)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def custom_step(self, batch, batch_idx, prefix='val'):
        """Summary

        Args:
            batch (TYPE): Description
            batch_idx (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        image, target = batch
        image = image / 128.0 - 1.0
        # output = self.forward(images)
        real_or_fake, output = self.discriminator(image)
        loss = self.probability_loss(output, target)

        result = OrderedDict({
            f'{prefix}_loss': loss,
            f'{prefix}_output': output,
            f'{prefix}_target': target,
        })
        # self.logger.experiment.add_image(f'{prefix}_images',
        #                                  torchvision.utils.make_grid(images / 255.0),
        #                                  dataformats='CHW')
        return result

    def validation_step(self, batch, batch_idx, prefix='val'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def test_step(self, batch, batch_idx, prefix='test'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def custom_epoch_end(self, outputs, prefix='val'):
        loss_mean = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        np_output = torch.cat([x[f'{prefix}_output'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()

        # Casting to binary
        np_output = 1.0 * (np_output >= self.hparams.threshold).astype(np.float32)
        np_target = 1.0 * (np_target >= self.hparams.threshold).astype(np.float32)

        result = {}
        result[f'{prefix}_loss'] = loss_mean

        tqdm_dict = {}
        tqdm_dict[f'{prefix}_loss'] = loss_mean

        tb_log = {}
        tb_log[f'{prefix}_loss'] = loss_mean

        f1_scores = []
        if np_output.shape[0] > 0 and np_target.shape[0] > 0:
            for p in range(self.hparams.types):
                f1_score = sklearn.metrics.fbeta_score(
                    np_target[:, p], np_output[:, p], beta=1, average=self.average_type)
                f2_score = sklearn.metrics.fbeta_score(
                    np_target[:, p], np_output[:, p], beta=2, average=self.average_type)
                precision_score = sklearn.metrics.precision_score(
                    np_target[:, p], np_output[:, p], average=self.average_type)
                recall_score = sklearn.metrics.recall_score(
                    np_target[:, p], np_output[:, p], average=self.average_type)

                f1_scores.append(f1_score)
                tqdm_dict[f'{prefix}_f1_score_{p}'] = f'{f1_score:0.4f}'
                tqdm_dict[f'{prefix}_f2_score_{p}'] = f'{f2_score:0.4f}',
                tqdm_dict[f'{prefix}_precision_score_{p}'] = f'{precision_score:0.4f}'
                tqdm_dict[f'{prefix}_recall_score_{p}'] = f'{recall_score:0.4f}'

                tb_log[f'{prefix}_f1_score_{p}'] = f1_score
                tb_log[f'{prefix}_f2_score_{p}'] = f2_score
                tb_log[f'{prefix}_precision_score_{p}'] = precision_score
                tb_log[f'{prefix}_recall_score_{p}'] = recall_score

            tqdm_dict[f'{prefix}_f1_score_mean'] = f'{np.array(f1_scores).mean():0.4f}'
            tb_log[f'{prefix}_f1_score_mean'] = np.array(f1_scores).mean()
        pprint(tqdm_dict)
        result['log'] = tb_log

        return result

    def validation_epoch_end(self, outputs, prefix='val'):
        return self.custom_epoch_end(outputs, prefix=prefix)

    def test_epoch_end(self, outputs, prefix='test_'):
        return self.custom_epoch_end(outputs, prefix=prefix)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=20)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)
        return [opt_g, opt_d], [sch_g, sch_d]

    def train_dataloader(self):
        ds_train = MultiLabelDataset(folder=self.hparams.data,
                                     is_train='train',
                                     fname='train_v7.1.csv',
                                     types=self.hparams.types,
                                     pathology=self.hparams.pathology,
                                     resize=int(self.hparams.shape),
                                     balancing='None')

        ds_train.reset_state()
        ag_train = [
            # imgaug.Albumentations(
            #     AB.SmallestMaxSize(self.hparams.shape, p=1.0)),
            imgaug.ColorSpace(mode=cv2.COLOR_GRAY2RGB),
            # imgaug.Affine(shear=10),
            imgaug.RandomChooseAug([
                imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),
                imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),
            ]),
            imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), p=0.5)),
            imgaug.RandomOrderAug([
                imgaug.Affine(shear=10, border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(translate_frac=(0.01, 0.02), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
                imgaug.Affine(scale=(0.5, 1.0), border=cv2.BORDER_CONSTANT, 
                    interp=cv2.INTER_AREA),
            ]),
            imgaug.RotationAndCropValid(max_deg=10, interp=cv2.INTER_AREA),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0),
                                                aspect_ratio_range=(0.8, 1.2),
                                                interp=cv2.INTER_AREA, target_shape=self.hparams.shape),
            imgaug.ColorSpace(mode=cv2.COLOR_RGB2GRAY),
            imgaug.ToFloat32(),
        ]
        ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        # Label smoothing
        ag_label = [
            imgaug.BrightnessScale((0.8, 1.2), clip=False),
        ]
        # ds_train = AugmentImageComponent(ds_train, ag_label, 1)
        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                       torch.tensor(dp[1]).float()])
        return ds_train

    def on_epoch_end(self):
        z = torch.randn(16, self.hparams.latent_dim)
        z = z.type_as(self.last_image)
        p = torch.empty(16, self.hparams.types).random_(2).type_as(z)
        z = torch.cat([z, p], dim=1)

        # log sampled images
        sample_image = self.generator(z)
        grid = torchvision.utils.make_grid(sample_image) / 2.0 + 0.5
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

    def val_dataloader(self):
        """Summary

        Returns:
            TYPE: Description
        """
        ds_valid = MultiLabelDataset(folder=self.hparams.data,
                                     is_train='valid',
                                     fname='valid_v7.1.csv',
                                     types=self.hparams.types,
                                     pathology=self.hparams.pathology,
                                     resize=int(self.hparams.shape),)

        ds_valid.reset_state()
        ag_valid = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_AREA),
            imgaug.ToFloat32(),
        ]
        ds_valid = AugmentImageComponent(ds_valid, ag_valid, 0)
        ds_valid = BatchData(ds_valid, self.hparams.batch, remainder=True)
        ds_valid = MultiProcessRunner(ds_valid, num_proc=4, num_prefetch=16)
        ds_valid = PrintData(ds_valid)
        ds_valid = MapData(ds_valid,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                       torch.tensor(dp[1]).float()])
        return ds_valid

    @pl.data_loader
    def test_dataloader(self):
        ds_test = MultiLabelDataset(folder=self.hparams.data,
                                     is_train='valid',
                                     fname='test_v7.1.csv',
                                     types=self.hparams.types,
                                     pathology=self.hparams.pathology,
                                     resize=int(self.hparams.shape),
                                     fold_idx=None,
                                     n_folds=1)

        ds_test.reset_state()
        ag_test = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_AREA),
            imgaug.ToFloat32(),
        ]
        ds_test = AugmentImageComponent(ds_test, ag_test, 0)
        ds_test = BatchData(ds_test, self.hparams.batch, remainder=True)
        ds_test = MultiProcessRunner(ds_test, num_proc=4, num_prefetch=16)
        ds_test = PrintData(ds_test)
        ds_test = MapData(ds_test,
                          lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                      torch.tensor(dp[1]).float()])
        return ds_test

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='se_resnext101_32x4d')
        parser.add_argument('--epochs', default=250, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2222,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--debug', action='store_true',
                            help='use fast mode')
        return parser


def get_args():
    """Summary

    Returns:
        TYPE: Description
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data', metavar='DIR', default=".", type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save', metavar='DIR', default="train_log", type=str,
                               help='path to save output')
    parent_parser.add_argument('--info', metavar='DIR', default="train_log",
                               help='path to logging output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('--percent_check', default=1.0, type=float,
                               help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--val_check_interval', default=0.2, type=float,
                               help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--fast_dev_run', default=False, action='store_true',
                               help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')

    parent_parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parent_parser.add_argument('--types', type=int, default=1)
    parent_parser.add_argument('--threshold', type=float, default=0.5)
    parent_parser.add_argument('--pathology', default='All')
    parent_parser.add_argument('--shape', type=int, default=320)
    parent_parser.add_argument('--folds', type=int, default=5)

    # Inference purpose
    # parent_parser.add_argument('--load', help='load model')
    parent_parser.add_argument('--load', action='store_true',
                               help='path to logging output')
    parent_parser.add_argument('--pred', action='store_true',
                               help='run predict')
    parent_parser.add_argument('--eval', action='store_true',
                               help='run offline evaluation instead of training')

    parser = SGAN.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = SGAN(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(str(hparams.save),
                              str(hparams.pathology),
                              str(hparams.shape),
                              str(hparams.types),
                              # str(hparams.folds),
                              # str(hparams.valid_fold_index),
                              'ckpt'),
        save_top_k=10,
        verbose=True,
        monitor='val_f1_score_mean',  # TODO
        mode='max'
    )

    trainer = pl.Trainer(
        train_percent_check=hparams.percent_check,
        val_percent_check=hparams.percent_check,
        test_percent_check=hparams.percent_check,
        num_sanity_val_steps=0,
        default_save_path=os.path.join(str(hparams.save),
                                       str(hparams.pathology),
                                       str(hparams.shape),
                                       str(hparams.types),
                                       # str(hparams.folds),
                                       # str(hparams.valid_fold_index)
                                       ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        val_check_interval=hparams.val_check_interval,
    )
    if hparams.eval:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())


