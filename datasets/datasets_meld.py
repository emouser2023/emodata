import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import json
import clip
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import torchaudio
import numbers
import math
import torch
import time
import decord
from randaugment import RandAugment
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

#########################################

import torchvision.transforms.functional as TF

def save_face_masks(tensor, save_dir='/media/sdb_access/Emotion_multi_model_CLIP/test_img', prefix="mask"):
    """
    Save each mask in the tensor as a binary image.

    Args:
        tensor (torch.Tensor): Tensor of shape [T, H, W]
        save_dir (str): Directory to save the images
        prefix (str): Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor = tensor.detach().cpu()  # Ensure it's on CPU

    for i, mask in enumerate(tensor):
        # Convert to uint8 image (0 or 255)
        mask_img = (mask * 255).byte()
        img = TF.to_pil_image(mask_img)
        img.save(os.path.join(save_dir, f"{prefix}_{i:03}.png"))

    print(f"Saved {len(tensor)} masks to '{save_dir}'")




def create_masks_tensor(frames, face_boxes, size=224):
    """
    Create a [T, H, W] tensor of resized binary face masks.

    Args:
        frames (list of np.ndarray): List of RGB frames
        face_boxes (list of list): List of bounding boxes per frame
        size (tuple): Desired output size (H, W)

    Returns:
        torch.Tensor: Binary face masks of shape [T, H, W]
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size,size)),
        transforms.ToTensor(),  # Converts to [1, H, W] float in [0,1]
    ])

    masks = []

    for frame, boxes in zip(frames, face_boxes):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1

        mask_tensor = transform(mask)  # [1, 224, 224]
        masks.append(mask_tensor.squeeze(0))  # [224, 224]

    return torch.stack(masks)  # [T, 224, 224]


########################################




def create_face_masks(frames, face_boxes):
    masks = []
    for frame, boxes in zip(frames, face_boxes):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
        masks.append(mask)
    return masks


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


class MELD_DATASETS(data.Dataset):
    def __init__(self, list_file,base_json_path,
                 num_segments=1, new_length=1, transform=None,
                 random_shift=True, test_mode=False, index_bias=1, root_path='', config=None, sample_rate=16000):

        self.list_file = list_file
        self.base_json_path = base_json_path
        self.num_segments = num_segments
        self.seg_length = new_length
        # self.image_tmpl = image_tmpl
        self.transform = transform
        self.sample_rate = sample_rate
        # self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=80)
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        # self.labels_file = labels_file
        self.root_path = root_path
        self.config = config

        # if self.index_bias is None:
        #     if self.image_tmpl == "frame{:d}.jpg":
        #         self.index_bias = 0
        #     else:
        #         self.index_bias = 1
        self._parse_list()
        self.initialized = False

    
    

    
    def _load_video(self, video_path):
        # video_path = record.path
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

        return frames, indices, number_of_frame
        
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    # @property
    # def classes(self):
    #     classes_all = pd.read_csv(self.labels_file)
    #     return classes_all.values.tolist()
    
    def _transform_audio(self, audio, sample_rate=16000):
        return torchaudio.transforms.Resample(orig_freq=audio[1], new_freq=sample_rate)(audio)
    
    def _parse_list(self):
        # self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        self.json_list = [x.strip() for x in open(self.list_file)]

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
        json_file = self.json_list[index]
        json_path = os.path.join(self.base_json_path,json_file)

        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading JSON {json_path}: {e}")

        # segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(json_data)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record):

        video_path = os.path.join(self.root_path,record['video_path'].lstrip("/"))
        try:
            frames, indices, num_of_frames = self._load_video(video_path)
        except Exception as e:
            # print(f"⚠️ Error loading audio {audio_path}: {e}")
            return None
        audio_path = video_path.replace('mp4','mp3') 

        try:
            text = clip.tokenize(record['text'])[0] 
        except Exception as e:
            # print(f"⚠️ Error loading audio {audio_path}: {e}")
            return None

        # if num_of_frames!=len(record['face_boxes']) or num_of_frames!=len(record['human_boxes']):
        #     print('Number of frame not equal to bboxes')

        # text_logits = torch.tensor(record['text_emotion_logits'])
        # audio_logits = torch.tensor(record['audio_emotion_logits'])
        # print(indices)
        face_boxes = [record['face_boxes'][ind-1] for ind in indices]  
        human_boxes = [record['human_boxes'][ind-1] for ind in indices] 

        face_mask_binary = create_masks_tensor(frames, face_boxes, size=self.config.data.input_size)
        human_mask_binary = create_masks_tensor(frames, human_boxes, size=self.config.data.input_size)

        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.dim() == 2 and waveform.shape[0] >= 1:  # Multi-channel
                waveform = waveform.mean(dim=0)  # Average channels to mono
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            waveform = waveform / torch.max(torch.abs(waveform))
        except Exception as e:
            # print(f"⚠️ Error loading audio {audio_path}: {e}")
            return None


        label = torch.tensor(record['class id'])
        # audio = self.audio_transform(waveform)
        
        
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
        process_video = self.transform(images)
        # print('audio shape in dataloader', waveform.shape)
        return {'vidio':process_video, 'face_mask':face_mask_binary , 'human_mask':human_mask_binary,
                 'text':text, 'waveform':waveform, 'label':label}

    def __len__(self):
        return len(self.json_list)