# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets import Emotion_DATASETS, MELD_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import sys
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import matplotlib.pyplot as plt
import pprint
from modules.Visual_Prompt import *
from modules.Audio_encoder import AudioEncoder
# from utils.KLLoss import KLLoss, SentimentDivergenceAwareTriModalLoss
from utils.KLLoss import *
from test import  *
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image,face_mask,human_mask):
        return self.model.encode_image(image,face_mask,human_mask)
    

    
def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

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

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='configs/emotion_clip.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--traning_name', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    args.traning_name = config['training_name']    
    # working_dir = os.path.join('/media/sda1_acces/Code/Arxiv_code_change_dataloading/ActionCLIP/exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    working_dir = os.path.join(config['weight_save_dir'] , config['network']['type'], config['network']['arch'], config['data']['dataset'], args.traning_name)
    wandb.init(project=config['network']['type'],name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))
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
   

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32
    # model, clip_state_dict = clip.load(config.network.arch,device=device,jit=False) #Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_audio = AudioEncoder()
    fusion_model = MultimodalFusionClassifier(embed_dim=512, num_classes=config.data.number_of_class)

    for name, p in model.named_parameters():
        if 'Adapter' not in name:
            p.requires_grad = False


    ###########################################################
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('CLIP_model Trainable Parameters: %.3fM' % parameters)

    parameters = filter(lambda p: p.requires_grad, model_audio.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Audio model Trainable Parameters: %.3fM' % parameters)
    
    ############################################################
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    model_audio = torch.nn.DataParallel(model_audio).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(model_audio)
    wandb.watch(fusion_model)

    train_data = Emotion_DATASETS(config.data.train_list, root_path= config.data.base_video_path,json_path=config.data.base_json_path, num_segments=config.data.num_segments, random_shift=config.data.random_shift,
                       transform=transform_train, config = config)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,collate_fn=collate_fn,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)


    test_data = Emotion_DATASETS(config.data.test_list, root_path= config.data.test_base_video_path, json_path=config.data.test_base_json_path, num_segments=config.data.num_segments,
                       transform=transform_val, config = config)
    test_loader = DataLoader(test_data,batch_size=config.data.batch_size,collate_fn=collate_fn,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)


    if device == "cpu":
        model_text.float()
        model_image.float()
        # model_audio.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)
        # clip.model.convert_weights(model_audio)

    # loss_img = KLLoss()
    # loss_txt = KLLoss()
    # emotion_loss = SentimentDivergenceAwareTriModalLoss()
    emotion_loss = ExtendedReweightedClipLoss(world_size=1)
    classifcation_loss  = nn.CrossEntropyLoss()
    # emotion_loss = SMTCLoss()

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
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_audio.load_state_dict(checkpoint['audio_model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    optimizer = _optimizer(config, model,model_audio,fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)



    # metric = validate_2( test_loader, device, model, model_audio, fusion_model,config)

    loss=[]
    top_1_acc=[]
    best_prec1 = 0.0
    # if config.solver.evaluate:
    #     prec1 = validate(start_epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug)
    #     return

    # for k,v in model.named_parameters():
    #     if v.requires_grad:
    #         print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        # print('------------------------------------------------------------------------')
        # print('Epoch %d start ..'%epoch)
        model_image.train()
        model_text.train()
        model_audio.train()
        fusion_model.train()
        tic = time.time()
        epoch_loss=[]
        # for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
        for kkk,(images, texts, audios,face_mask,human_mask,text_logits, audio_logits, label) in enumerate(train_loader):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            label = label.to(device)
            b,t,c,h,w = images.size()
            images = images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            face_mask = face_mask.to(device).view(-1,1,h,w )
            human_mask = human_mask.to(device).view(-1,1,h,w )
            texts = texts.to(device)
            audios = audios.to(device)
            text_logits = text_logits.to(device)
            audio_logits = audio_logits.to(device)

            image_embedding = model_image(images,face_mask,human_mask)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = image_embedding.mean(dim=1, keepdim=False)

            text_embedding = model_text(texts)
            audio_embedding = model_audio(audios)
            audio_embedding = audio_embedding.type(image_embedding.dtype)

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()

            # image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            # text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            # audio_embedding /= audio_embedding.norm(dim=-1, keepdim=True)
            class_logits = fusion_model(image_embedding, text_embedding, audio_embedding)

            total_loss = emotion_loss(image_embedding, text_embedding, audio_embedding, logit_scale, text_logits, audio_logits) + classifcation_loss(class_logits,label)
            # total_loss = classifcation_loss(class_logits,label)
            wandb.log({"train_total_loss": total_loss})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            epoch_loss.append(total_loss.item())
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if kkk%10==0:
                print('Epoch:%d  iteration:%d/%d, total loss:%f, lr:%f '%(epoch,kkk,len(train_loader),total_loss.item(), optimizer.param_groups[0]['lr']))



        # epoch_saving(epoch, model, model_audio,fusion_model, optimizer, filename)
        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            print('Test accuracy')
            prec1 = validate_2( test_loader, device, model, model_audio, fusion_model,config)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))



        txt_path = '{}/log.txt'.format(working_dir)
        if os.path.exists(txt_path):
            with open(txt_path, 'a+') as f:
                f.write('\n')
                f.write('Epoch:%d  iteration:%d/%d, total loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(), optimizer.param_groups[0]['lr']))
                f.write('Testing: {}/{}\n'.format(prec1,best_prec1))
                f.close()
        else:
            with open(txt_path, mode="wt") as f:
                f.write('Epoch:%d  iteration:%d/%d, total loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(), optimizer.param_groups[0]['lr']))
                f.write('Testing: {}/{}\n'.format(prec1,best_prec1))
                f.close()
        
        
        print('Saving:')
        filename1 = "{}/last_model.pt".format(working_dir)
        # filename = "{}/epoch_{}_model.pt".format(working_dir,epoch)
        top_1_acc.append(prec1/100)
        loss.append(np.mean(epoch_loss))
        # epoch_saving(epoch, model,  optimizer, filename)
        epoch_saving(epoch, model,model_audio,fusion_model,  optimizer, filename1)
        if is_best:
            print('Saving best weight based on K-400 accuracy at epoch %d'%epoch)
            best_saving(working_dir, epoch, model,model_audio,fusion_model, optimizer)

        print('Epoch %d end ..'%epoch)
        ##############graph_plot################
        X = list(range(len(loss)))
        plt.plot(X, loss, color='r', label='Training loss')
        plt.plot(X, top_1_acc, color='g', label='Test Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Training loss and Accuracy")
        plt.title("Traing graph")
        plt.legend()
        plt.savefig('{}/Graph_plot.png'.format(working_dir))
        plt.close()
        print('Time taken by epoch %d:'%epoch, print_time(time.time()-tic))
        print('------------------------------------------------------------------------')

        
        

if __name__ == '__main__':
    main()
