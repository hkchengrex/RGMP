from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models


# general libs
import cv2
from PIL import Image
import numpy as np
import argparse
import os
import time
from os import path
#
from model import RGMP
from utils import ToCudaVariable, ToLabel, upsample, downsample
from youtube_dataset import YOUTUBE_VOS_MO_Test

from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description="RGMP")
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1, help='inclusive', type=int)
    parser.add_argument('--code', type=str, help='Code name', default='def')
    parser.add_argument("--vos_root", type=str, help="path to data", default='../YouTube/vos/train_480p')
    parser.add_argument("--af_root", type=str, help="path to data", default='../YouTube/vos/all_frames/train_480p')
    return parser.parse_args()
args = get_arguments()
start_idx = args.start
end_idx = args.end
code = args.code
VOS_ROOT = args.vos_root
AF_ROOT = args.af_root


def Encode_MS(val_F1, val_P1, scales):
    ref = {}
    for sc in scales:
        if sc != 1.0:
            msv_F1, msv_P1 = downsample([val_F1, val_P1], sc)
            # msv_F1, msv_P1 = ToCudaVariable([msv_F1, msv_P1], volatile=True)
            ref[sc] = model.module.Encoder(msv_F1, msv_P1)[0]
        else:
            # msv_F1, msv_P1 = ToCudaVariable([val_F1, val_P1], volatile=True)
            msv_F1, msv_P1 = val_F1, val_P1
            ref[sc] = model.module.Encoder(msv_F1, msv_P1)[0]

    return ref

def Propagate_MS(ref, val_F2, val_P2, scales):
    h, w = val_F2.size()[2], val_F2.size()[3]
    msv_E2 = {}
    for sc in scales:
        if sc != 1.0:
            # print(sc, val_P2.shape)
            msv_F2, msv_P2 = downsample([val_F2, val_P2], sc)
            # msv_F2, msv_P2 = ToCudaVariable([msv_F2, msv_P2], volatile=True)
            r5, r4, r3, r2  = model.module.Encoder(msv_F2, msv_P2)
            e2 = model.module.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = upsample(F.softmax(e2[0], dim=1)[:,1].data.cpu(), (h,w))
        else:
            msv_F2, msv_P2 = val_F2, val_P2
            r5, r4, r3, r2  = model.module.Encoder(msv_F2, msv_P2)
            e2 = model.module.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = F.softmax(e2[0], dim=1)[:,1].data.cpu()

    val_E2 = torch.zeros(val_P2.size())
    for sc in scales:
        val_E2 += msv_E2[sc]
    val_E2 /= len(scales)
    return val_E2


def Infer_MO(Fs, Ms, AFs, scales=[0.5, 0.75, 1.0]):

    b, _, t, h, w = Fs.shape
    _, _, at, h, w = AFs.shape
    _, k, _, _, _ = Ms.shape


    Es = torch.zeros((b, k+1, at, h, w), dtype=torch.float32)
    Es[:,1:,0] = Ms[:,:,0]

    last_idx = -1
    last_mask = Ms[:,:,0]
    for t_step in range(1, at):
        curr_idx = max(0, min(t-1, t_step//5))
        next_idx = min(t-1, curr_idx+1)
        # Avoid recomputing reference
        if curr_idx != last_idx:
            if curr_idx != next_idx:
                currM = Ms[:,:,curr_idx].sum((0,2,3))
                nextM = Ms[:,:,next_idx].sum((0,2,3))
                curr_obj = torch.nonzero(currM)
                next_obj = torch.nonzero(nextM)
                objects_exist = list(torch.unique(torch.cat([curr_obj, next_obj])))
                objects_exist = [int(i.item()) for i in objects_exist]
                ref_idx = []
                # We pick the frame with most matching pixels
                for obj in objects_exist:
                    if currM[obj] > nextM[obj]:
                        ref_idx.append(curr_idx)
                    else:
                        ref_idx.append(next_idx)
            else:
                currM = Ms[:,:,curr_idx].sum((0,2,3))
                objects_exist = torch.nonzero(currM)
                objects_exist = list(objects_exist)
                objects_exist = [int(i.item()) for i in objects_exist]
                ref_idx = []
                # We pick the frame with most matching pixels
                for obj in objects_exist:
                    ref_idx.append(curr_idx)

            print(objects_exist)

            if len(objects_exist) > 1:
                # Now compute the reference features
                ref_feat = [None] * (k+1)
                last_mask.fill_(0)
                # Background feature, uses sum to represent all foreground which is later subtracted from 1
                ref_feat[0] = Encode_MS(Fs[:,:,curr_idx], torch.sum(Ms[:,1:,curr_idx], dim=1), scales)
                for i, obj_idx in enumerate(objects_exist):
                    ref_feat[obj_idx+1] = Encode_MS(Fs[:,:,ref_idx[i]], Ms[:,obj_idx,ref_idx[i]], scales)
                    last_mask[:,obj_idx] = Ms[:,obj_idx,ref_idx[i]]
            else:
                # For num_objects = 1, we skip to use the faster SO evaluation
                obj_idx = objects_exist[0]
                ref_feat = Encode_MS(Fs[:,:,curr_idx], Ms[:,obj_idx,curr_idx], scales)
                last_mask = Ms[:,:,curr_idx]

        last_idx = curr_idx

        if len(objects_exist) == 1:
            Es[:,obj_idx+1,t_step] = Propagate_MS(ref_feat, AFs[:,:,t_step], last_mask[:,obj_idx], scales)
            Es[:,0,t_step] = 1 - Es[:,obj_idx+1,t_step]
            last_mask.fill_(0)
            last_mask[:,obj_idx] = Es[:,obj_idx+1,t_step]
        else:
            # Background prop
            Es[:,0,t_step] = 1-Propagate_MS(ref_feat[0], AFs[:,:,t_step], torch.sum(last_mask[:,:], dim=1), scales)
            for obj_idx in range(k):
                if obj_idx in objects_exist:
                    # Objects prop
                    Es[:,obj_idx+1,t_step] = Propagate_MS(ref_feat[obj_idx+1], AFs[:,:,t_step], last_mask[:,obj_idx], scales)
                    last_mask[:,obj_idx] = Es[:,obj_idx+1,t_step]
                else:
                    # Reset mask
                    last_mask[:,obj_idx].fill_(0)

        Es[:,:,t_step] = torch.clamp(Es[:,:,t_step], 1e-7, 1-1e-7)
        Es[:,:,t_step] = torch.log((Es[:,:,t_step] /(1-Es[:,:,t_step])))
        Es[:,:,t_step] = F.softmax(Es[:,:,t_step], dim=1)

    return Es


Testset = YOUTUBE_VOS_MO_Test(VOS_ROOT, AF_ROOT, start_idx=start_idx, end_idx=end_idx)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(RGMP()).cuda()

model.load_state_dict(torch.load('weights.pth'))

palette = Image.open(path.join(VOS_ROOT, 'Annotations/0a2f2bd294/00000.png')).getpalette()

model.eval() # turn-off BN
torch.set_grad_enabled(False)
for seq, (Fs, Ms, AFs, info) in enumerate(Testloader):
    # Fs, Ms, AFs = Fs[0], Ms[0], AFs[0]
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0]
    num_objects = info['num_objects'][0]
    frames_name = info['frames_name']

    Fs = Fs.cuda(non_blocking=True)
    AFs = AFs.cuda(non_blocking=True)
    Ms = Ms.cuda(non_blocking=True)

    # tt = time.time()
    all_E = Infer_MO(Fs, Ms, AFs, scales=[0.5, 0.75, 1.0])
    # print('{} | num_objects: {}, FPS: {}'.format(seq_name, num_objects, num_frames /(time.time()-tt)))

    # Save results for quantitative eval ######################
    folder = 'results/%s_%d_%d' % (code, start_idx, end_idx)
    test_path = os.path.join(folder, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for i, f in enumerate(tqdm(frames_name)):
        E = all_E[0,:,i].cpu().numpy()
        # make hard label
        E = ToLabel(E)
    
        (lh,uh), (lw,uw) = info['pad'] 
        E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        img_E = Image.fromarray(E)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, f[0].replace('.jpg','.png')))

