import os
import sys
import time


def readlines(datapath):
    # Read all the lines in a text file and return as a list
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def parameter_number(model): # 모델의 파라미터를 계산하는 함수
    num_params = 0
    for tensor in list(model.parameters()):
        tensor      = tensor.view(-1)
        num_params += len(tensor)
    print(">>>  ", num_params)


def sample_dataset(dataloader): # 모델 데이터로터에서 배치 샘플 하나를 추출
    sample = 0
    start = time.time()
    for index, data in enumerate(dataloader):
        sample = data
        if index == 0:
            break  
    print("batch sampling time:  ", time.time() - start)
    return sample