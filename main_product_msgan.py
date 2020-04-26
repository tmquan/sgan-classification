
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
from torch.nn import DataParallel
from torch.nn.functional import avg_pool2d
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

from CustomLayer import *

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




class MSGGAN(LightningModule):
    def __init__(self, depth=8, latent_size=64,
                 use_eql=True, use_ema=False, ema_decay=0.999, normalize_latents=False, 
                 hparams=None):
        """ constructor for the class """
        super().__init__()
        # from torch.nn import DataParallel
        self.hparams = hparams
        self.average_type = 'binary'
        self.gen = Generator(depth, latent_size, use_eql=use_eql)


        device = torch.device('cuda')
        # Parallelize them if required:
        if device == torch.device('cuda'):
            self.gen = DataParallel(self.gen)
            self.dis = Discriminator(depth, latent_size,
                                     use_eql=use_eql, gpu_parallelize=True)
        else:
            self.dis = Discriminator(depth, latent_size, use_eql=True)
        self.classifier = Classifier(self.hparams)
        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.normalize_latents = normalize_latents
        if self.use_ema:
            from MSG_GAN.CustomLayers import update_average
            # updater function:
            self.ema_updater = update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        from torch.nn.functional import avg_pool2d
    
    def forward(self, x):
        pass
    
    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = torch.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))
        return generated_images

    def optimize_discriminator(self, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        return loss

    def optimize_generator(self, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # print(noise.shape)
        # print(len(fake_samples), len(real_batch))

        fake_image = fake_samples[-1][:16]        
        fake_grid = torchvision.utils.make_grid(fake_image) / 2.0 + 0.5
        self.logger.experiment.add_image('fake_images', fake_grid, self.current_epoch)

        true_image = real_batch[-1][:16]
        true_grid = torchvision.utils.make_grid(true_image) / 2.0 + 0.5
        self.logger.experiment.add_image('true_images', true_grid, self.current_epoch)

        return loss

    def create_grid(self, samples, img_files):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate
        from numpy import sqrt, power

        # dynamically adjust the colour of the images
        samples = [Generator.adjust_dynamic_range(sample) for sample in samples]

        # resize the samples to have same resolution:
        for i in range(len(samples)):
            samples[i] = interpolate(samples[i],
                                     scale_factor=power(2,
                                                        self.depth - 1 - i))
        # save the images:
        for sample, img_file in zip(samples, img_files):
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
                       normalize=True, scale_each=True, padding=0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, labels = batch
        images = images / 128.0 - 1.0

        # create a list of downsampled images from the real images:
        images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                 for i in range(1, self.depth)]
        images = list(reversed(images))

        # sample some random latent points
        gan_input = th.randn(labels.shape[0], self.latent_size)
        # p = torch.empty(labels.shape).uniform_(0, 1).type_as(images[0])
        p = torch.empty(labels.shape).random_(2).type_as(images[0])
        gan_input[:,-self.hparams.types:] = p*2-1

        # normalize them if asked
        if self.normalize_latents:
            gan_input = (gan_input
                         / gan_input.norm(dim=-1, keepdim=True)
                         * (self.latent_size ** 0.5))
                
        # # Log the architecture
        # if self.current_epoch == 0 and batch_idx == 0:
        #     self.logger.experiment.add_graph(self.dis, images)
        #     self.logger.experiment.add_graph(self.gen, torch.randn(gan_input.shape))
        #     self.logger.experiment.add_graph(self.classifier, torch.randn(images[-1].shape))

        if optimizer_idx == 0:
            g_loss = self.optimize_generator(gan_input, images, loss_fn=LSGAN(self.dis))
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.optimize_discriminator(gan_input,images, loss_fn=LSGAN(self.dis))
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train classifier
        if optimizer_idx == 2:
            fake_samples = self.gen(gan_input)[-1] # get the maximum resolution
            fake_targets = p
            fake_outputs = self.classifier(fake_samples)

            true_samples = images[-1]
            true_targets = labels
            true_outputs = self.classifier(true_samples)

            # print(fake_targets.shape, fake_samples.shape, true_targets.shape, true_samples.shape)
            c_loss = SoftDiceLoss()(torch.cat([fake_outputs, true_outputs], dim=0), 
                                  torch.cat([fake_targets, true_targets], dim=0))
            tqdm_dict = {'c_loss': c_loss}
            output = OrderedDict({
                'loss': c_loss,
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
        output = self.classifier(image)
        loss = SoftDiceLoss()(output, target)

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
        result = {}
        tqdm_dict = {}
        tb_log = {}
        
        loss_mean = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        np_output = torch.cat([x[f'{prefix}_output'] for x in outputs], axis=0).squeeze_(0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'] for x in outputs], axis=0).squeeze_(0).to('cpu').numpy()

        # Casting to binary
        np_output = 1.0 * (np_output >= self.hparams.threshold).astype(np.float32)
        np_target = 1.0 * (np_target >= self.hparams.threshold).astype(np.float32)

        
        result[f'{prefix}_loss'] = loss_mean
        tqdm_dict[f'{prefix}_loss'] = loss_mean
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

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10)
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(b1, b2))
        sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=10)
        
        return [opt_g, opt_d, opt_c], [sch_g, sch_d, sch_c]

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
            ds_train = FixedSizeData(ds_train, 1)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train,
                           lambda dp: [torch.tensor(np.transpose(dp[0], (0, 3, 1, 2))),
                                       torch.tensor(dp[1]).float()])
        return ds_train

    def on_epoch_end(self):
        pass
        # z = torch.randn(16, self.hparams.latent_dim)
        # z = z.type_as(self.last_image)
        # p = torch.empty(16, self.hparams.types).random_(2).type_as(z)
        # z = torch.cat([z, p], dim=1)

        # # log sampled images
        # sample_image = self.generator(z)
        # grid = torchvision.utils.make_grid(sample_image) / 2.0 + 0.5
        # self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

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
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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

    parent_parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
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

    parser = MSGGAN.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = MSGGAN(hparams=hparams)
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
