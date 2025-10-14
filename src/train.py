"""
모델 학습 스크립트
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pickle

from models import CNNModel, LSTMModel
from dataset import get_loader


def train_model(config):
    """
    이미지 캡셔닝 모델 학습
    
    Args:
        config (dict): 학습 설정 딕셔너리
    """
    # GPU/CPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 저장 폴더 생성
    if not os.path.exists(config['model_save_path']):
        os.makedirs(config['model_save_path'])
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    
    # 단어 사전 로드
    with open(config['vocab_path'], 'rb') as f:
        vocabulary = pickle.load(f)
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # 데이터 로더 생성
    data_loader = get_loader(
        config['image_path'],
        config['caption_path'],
        vocabulary,
        transform,
        config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # 모델 초기화
    encoder = CNNModel(config['embedding_size']).to(device)
    decoder = LSTMModel(
        config['embedding_size'],
        config['hidden_size'],
        len(vocabulary),
        config['num_layers']
    ).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + \
             list(encoder.linear_layer.parameters()) + \
             list(encoder.batch_norm.parameters())
    optimizer = optim.Adam(params, lr=config['learning_rate'])
    
    # 학습 루프
    total_steps = len(data_loader)
    
    for epoch in range(config['num_epochs']):
        for i, (imgs, caps, lens) in enumerate(data_loader):
            # 데이터를 device로 이동
            imgs = imgs.to(device)
            caps = caps.to(device)
            targets = pack_padded_sequence(
                caps, lens, batch_first=True, enforce_sorted=False
            )[0]
            
            # 순전파
            features = encoder(imgs)
            outputs = decoder(features, caps, lens)
            loss = criterion(outputs, targets)
            
            # 역전파 및 최적화
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 로그 출력
            if i % config['log_step'] == 0:
                perplexity = np.exp(loss.item())
                print(f'Epoch [{epoch}/{config["num_epochs"]}], '
                      f'Step [{i}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Perplexity: {perplexity:.4f}')
            
            # 모델 저장
            if (i+1) % config['save_step'] == 0:
                torch.save(
                    decoder.state_dict(),
                    os.path.join(config['model_save_path'],
                                 f'decoder-{epoch+1}-{i+1}.ckpt')
                )
                torch.save(
                    encoder.state_dict(),
                    os.path.join(config['model_save_path'],
                                 f'encoder-{epoch+1}-{i+1}.ckpt')
                )


if __name__ == '__main__':
    # 학습 설정
    config = {
        'image_path': './data_dir/train2014/train2014',
        'caption_path': './data_dir/annotations_trainval2014/annotations/captions_train2014.json',
        'vocab_path': './data_dir/vocabulary.pkl',
        'model_save_path': './models_dir/',
        'embedding_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'num_epochs': 5,
        'batch_size': 128,
        'num_workers': 2,
        'learning_rate': 0.001,
        'log_step': 10,
        'save_step': 1000
    }
    
    train_model(config)