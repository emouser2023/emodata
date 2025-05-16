import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
import time
import decord
from randaugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    # @property
    # def num_frames(self):
    #     return int(self._data[1])

    @property
    def label(self):
        return int(self._data[1])


class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1, root_path=''):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.root_path = root_path

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    
    def _load_video_old(self, video_path, indices):        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frames_2 =[]
        #######################################################
        # tic = time.time()
        # for index in indices:
        #     # cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        #     cap.set(1,index)
        #     ret, frame = cap.read()
        #     if ret:
        #         frame = cv2.resize(frame, (224,224))
        #         frames_2.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # # print('Time taken by selected frames: ',time.time()-tic)
        # cap.release()
        #########################################################
        cap = cv2.VideoCapture(video_path)
        # tic = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224,224))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))      
        # print('Time taken by all frames: ',time.time()-tic)
        ###########################################################    
        cap.release()
        return frames
    

    # def _load_video(self, video_path, indices):
    #     frames = []
    #     video = decord.VideoReader(video_path)
    #     for frame in video:
    #         frames.append(frame.asnumpy())
    #     return frames
    
    # def _load_video(self, video_path, indices):
    #     video = decord.VideoReader(video_path)
    #     frames = []
    #     for index in indices:
    #         frames.append(video[index].asnumpy())
    #     return frames
    
    def _load_video(self, record):
        video_path = record.path
        # video = decord.VideoReader(video_path)
        video = decord.VideoReader(os.path.join(self.root_path, video_path))
        number_of_frame = len(video)
        indices = self._sample_indices(number_of_frame) if self.random_shift else self._get_val_indices(number_of_frame)
        frames = []
        for index in indices:
            try:
                frames.append(video[index-1].asnumpy())
            except:
                print("An exception occurred")
                print(indices)
                print(len(video))

        return frames
        
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(num_frames // 2),
                    num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(num_frames),
                randint(num_frames,
                        size=self.total_length - num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, num_frames):
        if self.num_segments == 1:
            return np.array([num_frames //2], dtype=np.int) + self.index_bias
        
        if num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), num_frames) + self.index_bias
            return np.array([i * num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + self.index_bias
        offset = (num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        # segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record):
        frames = self._load_video(record)
        images = list()
        for i in range(len(frames)):
            try:
                seg_imgs = [Image.fromarray(frames[i]).convert('RGB')]

                ################ To make each frame constant in video which remove the motion ##################
                # seg_imgs = [Image.fromarray(frames[len(frames)//2]).convert('RGB')]
                ################ To make each frame constant in video which remove the motion ##################

            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                # print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)