# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import clip
import torch.nn as nn
from datasets import Emotion_DATASETS, MELD_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import *
from modules.Audio_encoder import AudioEncoder
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
import numpy as np


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    video = torch.stack([item['vidio'] for item in batch])  # [B, C, H, W]
    texts = torch.stack([item['text'] for item in batch])    # [B, 77]
    audio_lengths = [item['waveform'].shape[0] for item in batch]
    face_mask = torch.stack([item['face_mask'] for item in batch])
    human_mask = torch.stack([item['human_mask'] for item in batch])
    text_logits = torch.stack([item['text_logits'] for item in batch])
    audio_logits = torch.stack([item['audio_logits'] for item in batch])
    label = torch.stack([item['label_id'] for item in batch])
    max_length = max(audio_lengths)
    # print("max length----",max_length)
    audios = torch.zeros(len(batch), max_length)  # [B, max_length]
    for i, audio in enumerate([item['waveform'] for item in batch]):
        # print('audio shape',audio.shape)
        audios[i, :audio.shape[0]] = audio  # 1D audio now
    # print('audios.shape---',audios.shape)
    return video, texts, audios,face_mask,human_mask,text_logits, audio_logits,label

class LinearProbe(nn.Module):
    def __init__(self, input_dim_per_modality=512, num_modalities=3, hidden_dim=512, num_classes=7, learning_rate=0.01, epochs=100):
        super(LinearProbe, self).__init__()
        total_input_dim = input_dim_per_modality * num_modalities  # 1536

        self.fusion_layer = nn.Linear(total_input_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(dtype=torch.float16, device=self.device)

    def forward(self, visual_features, text_features, audio_features):
        visual_features = visual_features.to(self.device, dtype=torch.float16)
        text_features = text_features.to(self.device, dtype=torch.float16)
        audio_features = audio_features.to(self.device, dtype=torch.float16)

        combined_features = torch.cat((visual_features, text_features, audio_features), dim=1)
        hidden = torch.relu(self.fusion_layer(combined_features))
        outputs = self.clf(hidden)
        return outputs

    def fit(self, visual_features, text_features, audio_features, targets):
        # Ensure targets are long (integer) for CrossEntropyLoss
        targets = targets.to(self.device, dtype=torch.long)

        self.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self(visual_features, text_features, audio_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def predict(self, visual_features, text_features, audio_features):
        self.eval()
        with torch.no_grad():
            outputs = self(visual_features, text_features, audio_features)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.cpu()

    def predict_proba(self, visual_features, text_features, audio_features):
        self.eval()
        with torch.no_grad():
            outputs = self(visual_features, text_features, audio_features)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().float().numpy()

    def evaluate(self, visual_features, text_features, audio_features, targets):
        # Convert targets to integer type (long) before numpy conversion
        targets = targets.to(dtype=torch.long).cpu().numpy()  # Ensure integer type
        pred_labels = self.predict(visual_features, text_features, audio_features).numpy()
        pred_probs = self.predict_proba(visual_features, text_features, audio_features)

        # Metrics
        accuracy = accuracy_score(targets, pred_labels)
        f1 = f1_score(targets, pred_labels, average='weighted')
        map_score = average_precision_score(np.eye(7)[targets], pred_probs, average='macro')
        auc = roc_auc_score(np.eye(7)[targets], pred_probs, average='macro', multi_class='ovr')

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mAP': map_score,
            'AUC': auc
        }

def eval_collate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))

    # {'vidio':process_video, 'face_mask':face_mask_binary , 'human_mask':human_mask_binary,
    #              'text':text, 'text_logits':text_logits, 'audio_logits':audio_logits, 'waveform':waveform}
    
    video = torch.stack([item['vidio'] for item in batch])  # [B, C, H, W]
    texts = torch.stack([item['text'] for item in batch])    # [B, 77]
    audio_lengths = [item['waveform'].shape[0] for item in batch]
    face_mask = torch.stack([item['face_mask'] for item in batch])
    human_mask = torch.stack([item['human_mask'] for item in batch])
    label = torch.stack([item['label'] for item in batch])

    #face_mask_binary , human_mask_binary, text, text_logits, audio_logits,


    max_length = max(audio_lengths)
    # print("max length----",max_length)
    audios = torch.zeros(len(batch), max_length)  # [B, max_length]
    for i, audio in enumerate([item['waveform'] for item in batch]):
        # print('audio shape',audio.shape)
        audios[i, :audio.shape[0]] = audio  # 1D audio now
    # print('audios.shape---',audios.shape)
    return video, texts, audios,face_mask,human_mask,label







class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)



def validate_new( train_loader,test_loader, val_loader, device, model,model_audio, config):

    # Initialize the probe
    probe = LinearProbe(
        input_dim_per_modality=512,
        num_modalities=3,
        hidden_dim=512,
        num_classes=7,
        learning_rate=0.01,
        epochs=2000
    )
    model.eval()
    model_audio.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    all_visiual_features_train = []
    all_text_features_train = []
    all_audio_features_train = []
    all_targets_train = []

    with torch.no_grad():
        for iii, (image, texts, audios,face_mask,human_mask,label) in enumerate(tqdm(train_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            texts = texts.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            face_mask = face_mask.to(device).view(-1,1,h,w )
            human_mask = human_mask.to(device).view(-1,1,h,w )
            image_features = model.encode_image(image_input,face_mask,human_mask).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            audio_features = model_audio(audios)
            audio_features = audio_features.type(image_features.dtype)

            
            label = label.type(image_features.dtype).to(device)

            all_visiual_features_train.append(image_features)
            all_text_features_train.append(text_features)
            all_audio_features_train.append(audio_features)
            all_targets_train.append(label)

    all_targets_train = torch.cat(all_targets_train)
    all_visiual_features_train = torch.cat(all_visiual_features_train)
    all_text_features_train = torch.cat(all_text_features_train)
    all_audio_features_train = torch.cat(all_audio_features_train)



    all_visiual_features_test = []
    all_text_features_test = []
    all_audio_features_test = []
    all_targets_test = []

    with torch.no_grad():
        for iii, (image, texts, audios,face_mask,human_mask,label) in enumerate(tqdm(test_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            texts = texts.to(device)


            image_input = image.to(device).view(-1, c, h, w)
            face_mask = face_mask.to(device).view(-1,1,h,w )
            human_mask = human_mask.to(device).view(-1,1,h,w )
            image_features = model.encode_image(image_input,face_mask,human_mask).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            audio_features = model_audio(audios)
            audio_features = audio_features.type(image_features.dtype)
            label = label.type(image_features.dtype).to(device)

            all_visiual_features_test.append(image_features)
            all_text_features_test.append(text_features)
            all_audio_features_test.append(audio_features)
            all_targets_test.append(label)

    all_targets_test = torch.cat(all_targets_test)
    all_visiual_features_test = torch.cat(all_visiual_features_test)
    all_text_features_test = torch.cat(all_text_features_test)
    all_audio_features_test = torch.cat(all_audio_features_test)

    # Train on all three modalities
    probe.fit(all_visiual_features_train, all_text_features_train, all_audio_features_train, all_targets_train)

    # Evaluate on test data
    metrics = probe.evaluate(all_visiual_features_test, all_text_features_test, all_audio_features_test, all_targets_test)
    print("Multimodal Metrics:", metrics)
    return metrics


def validate_2( test_loader, device, model,model_audio,fusion_model, config):

    # Initialize the probe
    model.eval()
    model_audio.eval()
    fusion_model.eval()
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for iii, (image, texts, audios,face_mask,human_mask,text_logits, audio_logits, label) in enumerate(tqdm(test_loader)):
        # for iii, (image, texts, audios,face_mask,human_mask,label) in enumerate(tqdm(test_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            texts = texts.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            face_mask = face_mask.to(device).view(-1,1,h,w )
            human_mask = human_mask.to(device).view(-1,1,h,w )
            image_features = model.encode_image(image_input,face_mask,human_mask).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(texts)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            audio_features = model_audio(audios)
            audio_features = audio_features.type(image_features.dtype)
            # audio_features /= audio_features.norm(dim=-1, keepdim=True)
            label = label.type(image_features.dtype).to(device)

            class_logits = fusion_model(image_features, text_features, audio_features)

            # predicted class index
            preds = torch.argmax(class_logits, dim=1)  # [b]
            
            # count correct predictions
            correct_predictions += (preds == label).sum().item()
            total_samples += label.size(0)
    
    # compute accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    
    return accuracy




def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='configs/emotion_clip.yaml')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                         config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)


    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_audio = AudioEncoder()
    fusion_model = MultimodalFusionClassifier(embed_dim=512, num_classes=config.data.number_of_class)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    model_audio = torch.nn.DataParallel(model_audio).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(model_audio)
    wandb.watch(fusion_model)


    test_data = Emotion_DATASETS(config.data.test_list, root_path= config.data.test_base_video_path, json_path=config.data.test_base_json_path, num_segments=config.data.num_segments,
                       transform=transform_val, config = config)
    test_loader = DataLoader(test_data,batch_size=config.data.batch_size,collate_fn=collate_fn,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)


    # MELD_train_data = MELD_DATASETS(config.eval_data.train_list, config.eval_data.base_json_path_train, num_segments=config.data.num_segments,random_shift=False,
    #                    transform=transform_val, root_path=config.eval_data.base_video_path,config = config)
    # meld_train_loader = DataLoader(MELD_train_data,batch_size=config.eval_data.batch_size,collate_fn=eval_collate_fn,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)

    # meld_test_data = MELD_DATASETS(config.eval_data.test_list,config.eval_data.base_json_path_test, random_shift=False,num_segments=config.data.num_segments,
    #                    transform=transform_val, root_path=config.eval_data.base_video_path,config = config)
    # meld_test_loader = DataLoader(meld_test_data,batch_size=config.eval_data.batch_size,collate_fn=eval_collate_fn, num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    
    # meld_dev_data = MELD_DATASETS(config.eval_data.dev_list,config.eval_data.base_json_path_dev, random_shift=False,num_segments=config.data.num_segments,
    #                    transform=transform_val, root_path=config.eval_data.base_video_path,config = config)
    # meld_dev_loader = DataLoader(meld_dev_data,batch_size=config.eval_data.batch_size, collate_fn=eval_collate_fn, num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)



    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_audio.load_state_dict(checkpoint['audio_model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))


    best_prec1 = 0.0
    # prec1 = validate_new( meld_train_loader,meld_test_loader, meld_dev_loader, device, model, model_audio, config)
    metric = validate_2( test_loader, device, model, model_audio, fusion_model,config)

if __name__ == '__main__':
    main()
