import os
import random
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



__classes__ = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]


class SpeechCommandDataset(Dataset):
    def __init__(self, datapath, filename, is_training):
        super(SpeechCommandDataset, self).__init__()
        """
        Args:
            datapath: "./datapath"
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.sampling_rate  = 16000 # 샘플링 레이트
        self.sample_length  = 16000 # 음성의 길이
        self.datapath       = datapath
        self.filename       = filename
        self.is_training    = is_training

        # 카테고리를 키로, 인덱스를 숫자로 매칭
        self.class_encoding = {category: index for index, category in enumerate(__classes__)}
        
        # 음성 augmentations을 위해 데이터 노이즈 백그라운드를 불러옴
        self.noise_path     = os.path.join(self.datapath, "_background_noise_")
        self.noise_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks = True)):
            for fn in sorted(filenames):
                name, ext = fn.split(".")
                if ext == "wav":
                    self.noise_dataset.append(os.path.join(root, fn)) 
                    # 확장자가 wav인 파일의 경로만 노이즈 데이터셋에 추가
        
        # 음성 데이터의 파일 경로를 라벨과 함께 묶음
        self.speech_dataset = self.combined_path()

                    
    def combined_path(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/") # 음성의 종류와 음성의 이름

            if category in __classes__[:-2]: # "yes부터 go"
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, category]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가

            elif category == "_silence_": # 음성 경로의 종류가 _silence_ 이면
                dataset_list.append(["silence", "silence"]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가

            else: # 만약 음성이 명시적이지도 않고 silence도 아닌 unknown이면 해당 경로는 라벨 unknown과 묶어서 리턴
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, "unknown"]) # 음성 경로와 카테고리를 묶어서 페어로 리스트 추가
        return dataset_list

    
    def load_audio(self, speech_path):
        waveform, sr = torchaudio.load(speech_path) # 오디오 로드

        if waveform.shape[1] < self.sample_length: # 오디오의 길이가 sample_length보다 짧으면
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]]) # 오른쪽으로 16000으로 패딩
        else:
            pass # 아니면 그대로 패스
        
        if self.is_training == True: # 훈련 모드이면 길이 augmentation
            pad_length = int(waveform.shape[1] * 0.1) # 양쪽으로 패딩할 작은 패딩 길이 정의
            waveform   = F.pad(waveform, [pad_length, pad_length]) # 양쪽으로 패딩
            
            # length augmentations을 하고 늘린 길이에서 자를 길이만큼 빼고 + 1 값중 랜덤 offset 설정
            offset   = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size = (1, )).item()
            waveform = waveform.narrow(1, offset, self.sample_length) # 이 중 랜덤으로 sample_length만큼 자름

            if self.noise_augmen == True: # 노이즈 augmentation 옵션
                noise_index = torch.randint(0, len(self.noise_dataset), size = (1,)).item()
                noise, noise_sampling_rate = torchaudio.load(self.noise_dataset[noise_index])

                offset = torch.randint(0, noise.shape[1] - self.sample_length + 1, size = (1, )).item()
                noise  = noise.narrow(1, offset, self.sample_length)
                
                # 노이즈를 sample_length 길이만큼 잘라서 waveform에 붙임
                background_volume = torch.rand(size = (1, )).item() * 0.1
                waveform.add_(noise.mul_(background_volume)).clamp(-1, 1) # -1, 1 이상은 clamp
            else:
                pass # 아니면 패스
        return waveform
    

    def one_hot(self, speech_category): # 카테고리를 숫자 값으로 인코딩하는 함수
        encoding = self.class_encoding[speech_category]
        return encoding
    

    def __len__(self):
        return len(self.speech_dataset)
    

    def __getitem__(self, index):
        self.noise_augmen = self.is_training and random.random() > 0.5 # 노이즈 증강 여부

        speech_path       = self.speech_dataset[index][0] # 패스
        speech_category   = self.speech_dataset[index][1] # 카테고리
        label             = self.one_hot(speech_category)

        if speech_path == "silence": # 경로가 silence이면 torch.zeros로 waveform 생성
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform  = self.load_audio(speech_path) # 해당 소리 파일의 경로로 소리를 불러옴
    
        return waveform, label # (소리, 라벨) 리턴