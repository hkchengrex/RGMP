import os
from os import path
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob

class YOUTUBE_VOS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, all_frames_root, start_idx, end_idx):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations')
        self.image_dir = path.join(root, 'JPEGImages')
        self.all_frames_image_dir = path.join(all_frames_root, 'JPEGImages')
        self.videos = []
        self.num_skip_frames = {}
        self.num_frames = {}
        self.shape = {}
        self.frames_name = {}
        self.skip_frames_name = {}

        self_vid_list = sorted(os.listdir(self.image_dir))

        print('This process handles video %d to %d out of %d' % (start_idx, end_idx, len(self_vid_list)))

        self_vid_list = self_vid_list[start_idx:end_idx+1]

        for vid in self_vid_list:
            self.videos.append(vid)
            self.num_skip_frames[vid] = len(os.listdir(os.path.join(self.image_dir, vid)))
            self.skip_frames_name[vid] = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.num_frames[vid] = len(os.listdir(os.path.join(self.all_frames_image_dir, vid)))
            self.frames_name[vid] = sorted(os.listdir(os.path.join(self.all_frames_image_dir, vid)))
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['num_skip_frames'] = self.num_skip_frames[video]
        info['num_objects'] = 0
        info['frames_name'] = self.frames_name[video]
        info['skip_frames_name'] = self.skip_frames_name[video]

        N_all_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)

        N_frames = np.empty((self.num_skip_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_skip_frames[video],)+self.shape[video], dtype=np.uint8)
        for i, f in enumerate(self.skip_frames_name[video]):
            img_file = path.join(self.image_dir, video, f)
            N_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.
 
            mask_file = path.join(self.mask_dir, video, f.replace('.jpg', '.png'))
            N_masks[i] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            info['num_objects'] = max(info['num_objects'], N_masks[i].max())

        for i, f in enumerate(info['frames_name']):
            img_file = path.join(self.all_frames_image_dir, video, f)
            N_all_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.

        Ms = np.zeros((self.num_skip_frames[video],)+self.shape[video]+(info['num_objects'],), dtype=np.uint8)
        for o in range(info['num_objects']):
            Ms[:,:,:,o] = (N_masks == (o+1)).astype(np.uint8)

        # padding size to be divide by 32
        nf, h, w, _ = Ms.shape
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        # print(new_h, new_w)
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        Ms = np.pad(Ms, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        N_frames = np.pad(N_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        N_all_frames = np.pad(N_all_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        info['pad'] = ((lh,uh),(lw,uw))

        Ms = torch.from_numpy(np.transpose(Ms.copy(), (3, 0, 1, 2)).copy()).float()
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        AFs = torch.from_numpy(np.transpose(N_all_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        return Fs, Ms, AFs, info



if __name__ == '__main__':
    pass
