import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_dataloader import *
from model_layer import *
from model_utils import *



class TRAINER(object):
    def __init__(self, opt):
        self.opt   = opt
        self.epoch = opt.epoch
        self.lr    = opt.lr
        self.batch = opt.batch
        self.step  = opt.step
    
        train_filename   = readlines("./splits/{}.txt".format("train"))
        valid_filename   = readlines("./splits/{}.txt".format("valid"))
        train_dataset    = SpeechCommandDataset("./dataset", train_filename, True)
        valid_dataset    = SpeechCommandDataset("./dataset", valid_filename, False)
        self.templet     = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"

        self.train_dataloader = DataLoader(
            train_dataset, batch_size = self.batch, shuffle = True, drop_last = True)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size = self.batch, shuffle = True, drop_last = True)
        self.train_length     = len(self.train_dataloader)
        self.valid_length     = len(self.valid_dataloader)
        print(">>>   Train length: {}   Valid length: {}".format(self.train_length, self.valid_length))

        if self.opt.model == "stft":
            self.model = STFT_TCResnet(
                filter_length = 256, hop_length = 129, bins = 129, 
                channels = self.opt.cha, channel_scale = self.opt.scale, num_classes = 12).to("cuda:0")
        elif self.opt.model == "mfcc":
            self.model = MFCC_TCResnet(
                bins = 40, channel_scale = self.opt.scale, num_classes = 12).to("cuda:0")

        print(">>>   Num of model parameters")
        parameter_number(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size = self.step, gamma = 0.1, last_epoch = -1)
        self.loss_name = {
            "train_loss": 0, "train_accuracy": 0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0, "valid_accuracy": 0, "valid_total": 0, "valid_correct": 0}


    def model_train(self):
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(self.train_dataloader)):
                waveform, labels = waveform.to("cuda:0"), labels.to("cuda:0")
                logits   = self.model(waveform)

                self.optimizer.zero_grad()
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                self.loss_name["train_loss"]     += loss.item() / self.train_length
                _, predict  = torch.max(logits.data, 1)
                self.loss_name["train_total"]    += labels.size(0)
                self.loss_name["train_correct"]  += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]
                
            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(self.valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to("cuda:0"), labels.to("cuda:0")
                    logits = self.model(waveform)
                    loss   = self.criterion(logits, labels)
                    
                    self.loss_name["valid_loss"]     += loss.item() / self.valid_length
                    _, predict  = torch.max(logits.data, 1)
                    self.loss_name["valid_total"]    += labels.size(0)
                    self.loss_name["valid_correct"]  += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
            
            self.scheduler.step()
            self.model_save()
            print(self.templet.format(self.epo+1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"], 
                self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))


    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if self.loss_name["valid_accuracy"] >= 92.0:
            torch.save(self.mode.state_dict(),
                os.path.join(save_directory, "best_" + str(self.loss_name["valid_accuracy"]) + ".pt"))

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(), 
                os.path.join(save_directory, "model" + str(self.epoch+1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))
        


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description = "Input optional guidance for training")
        parser.add_argument("--epoch", 
            default = 200, type = int, help = "모델의 에포크")
        parser.add_argument("--lr", 
            default = 0.05, type = float, help = "러닝 레이트")
        parser.add_argument("--batch", 
            default = 128, type = int, help = "배치 사이즈")
        parser.add_argument("--step", 
            default = 30, type = int, help = "스텝 수")

        parser.add_argument("--model", 
            default = "stft", type = str, help = ["stft", "mfcc"])
        parser.add_argument("--cha", 
            default = config["tc-resnet8"], type = list, help = "모델 레이어의 채널 리스트")
        parser.add_argument("--scale", 
            default = 3, type = int, help = "채널의 스케일링")
        parser.add_argument("--freq", 
            default = 30, type = int, help = "저장하는 빈도수")
        parser.add_argument("--save",   
            default = "stft", type = str, help = "저장하는 모델 파일 이름")
        args = parser.parse_args()
        return args

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    TRAINER(options(config)).model_train()
