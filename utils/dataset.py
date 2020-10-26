from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time

from torch.utils.data.dataloader import default_collate

import os
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class pad_collate():
    def __init__(self,n_frames_per_step,args):
        self.args = args
        self.n_frames_per_step = n_frames_per_step
    def __call__(self,batch):
        # Right zero-pad mel-spec
        max_input_len = max([x[2] for x in batch])
        max_target_len = max([x[4] for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        for i, elem in enumerate(batch):
            img,vis_info,img_length, mel, mel_length,key = elem
            
            output_length = mel.shape[1]
            input_dim = mel.shape[0]
            mel_padded = np.zeros((input_dim,max_target_len), dtype=np.float)
            gate_padded = np.zeros((max_target_len), dtype=np.float)
            mel_padded[:mel.shape[0], :mel.shape[1]] = mel       
            gate_padded[output_length-1:] = 1
            batch[i] = (img,vis_info,img_length,mel_padded, gate_padded, output_length, key)
        
        batch.sort(key=lambda x: x[-2], reverse=True)
        return default_collate(batch)

class pad_collate_BU():
    def __init__(self,args):
        self.args = args
    def __call__(self,batch):
        # Right zero-pad mel-spec
        max_input_len = max([x[1] for x in batch])
        for i, elem in enumerate(batch):
            img,length,key = elem
            input_length = img.shape[0]
            img_padded = np.zeros((max_input_len,img.shape[1]), dtype=np.float)
            img_padded[:img.shape[0],:img.shape[1]] = img
            img = img_padded          
            
            batch[i] = (img, key)

        return default_collate(batch)


def get_imgs(img_path, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if transform is not None:
        img = transform(img)

    return normalize(img)

# dataloader for the main programer
# used for RNN
class I2SData(data.Dataset):
    def __init__(self, args, data_dir, split='train',
                    img_size=224,
                    transform=None, target_transform=None):
        self.args = args
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.target_transform = target_transform
        self.embeddings_num = 5    #each image has 5 captions
        self.imsize = img_size
        self.data_dir = data_dir

        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(data_dir, split)
        if split != 'train' and not args.only_val:
            self.filenames = self.filenames[:6]
       
        self.number_example = len(self.filenames)


    def load_filenames(self, data_dir, split):      
        
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f) 
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):

        key = self.filenames[index]
        cls_id = index      
        data_dir = self.data_dir
        if self.args.img_format == 'img':
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize, self.transform, normalize=self.norm)   
            input_length = 49
        elif self.args.img_format == 'vector':
            img_name = '%s/image_feature_vector/%s.npy' % (data_dir, key)
            imgs = np.load(img_name)
            input_length = 1
        elif self.args.img_format == 'tensor':
            img_name = '%s/image_feature_tensor/%s.npy' % (data_dir, key)
            imgs = np.load(img_name)
            input_length = 49
        elif self.args.img_format == 'BU':
            img_name = '%s/bottom_up_features_36_info/%s.npy' % (data_dir, key)
            data = np.load(img_name,allow_pickle=True).item()
            img = data['features']
            boxs = data['boxes']
            confid = data['scores']
            clss = data['class']
            input_length = 36
            imgs = torch.from_numpy(img).float()
            cls_label = torch.from_numpy(clss).long()
            bbox = torch.from_numpy(boxs)
            confid = torch.from_numpy(confid).float()
            w_est = torch.max(bbox[:, [0, 2]])*1.+1e-5
            h_est = torch.max(bbox[:, [1, 3]])*1.+1e-5
            bbox[:, [0, 2]] /= w_est
            bbox[:, [1, 3]] /= h_est
            one_hot_label = torch.zeros(36, 1601).scatter_(1,cls_label.unsqueeze(-1) , 1).float()
            rel_area = (bbox[:, 3]-bbox[:, 1])*(bbox[:, 2]-bbox[:, 0])
            rel_area.clamp_(0)
            vis_info = torch.cat((bbox, rel_area.view(-1, 1), confid.view(-1, 1)), -1) # confident score
            vis_info = torch.cat((F.layer_norm(vis_info, [6]),one_hot_label), dim=-1)
        else:
            print('wrong image format')

        if self.split=='train':
            audio_ix = random.randint(0, self.embeddings_num)

            audio_file = '%s/mel_80/%s' % (data_dir, key) + '_' + str(audio_ix) +'.npy'
            audios = np.load(audio_file,allow_pickle=True)
            mel = audios.astype('float32')
            mel_length = mel.shape[-1]
            return imgs,vis_info,input_length, mel, mel_length, key
        else:
            return imgs,vis_info, key


    def __len__(self):
        return self.number_example

