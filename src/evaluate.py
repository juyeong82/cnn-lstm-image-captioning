"""
모델 평가 및 추론 스크립트
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import re

from models import CNNModel, LSTMModel
from utils import load_image


def generate_caption(image_path, encoder, decoder, vocabulary, device, transform):
    """
    단일 이미지에 대한 캡션 생성
    
    Args:
        image_path (str): 이미지 파일 경로
        encoder (CNNModel): 학습된 인코더
        decoder (LSTMModel): 학습된 디코더
        vocabulary (Vocab): 단어 사전
        device (torch.device): 연산 장치
        transform (callable): 이미지 변환 함수
    
    Returns:
        str: 생성된 캡션 문장
    """
    # 이미지 로드 및 전처리
    img = load_image(image_path, transform)
    img_tensor = img.to(device)
    
    # 인코더를 통한 특징 추출
    with torch.no_grad():
        features = encoder(img_tensor)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()
    
    # 인덱스를 단어로 변환
    predicted_caption = []
    for token_index in sampled_ids:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
    
    # <start>와 <end> 제거
    sentence = ' '.join(predicted_caption)
    matched = re.search(r"<start>\s*(.*?)\s*<end>", sentence)
    caption = matched.group(1) if matched else ' '.join(predicted_caption[1:-1])
    
    return caption


def evaluate_model(config):
    """
    모델 평가 및 시각화
    
    Args:
        config (dict): 평가 설정 딕셔너리
    """
    # GPU/CPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 이미지 전처리 (학습 시와 동일한 정규화 값 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    
    # 단어 사전 로드
    with open(config['vocab_path'], 'rb') as f:
        vocabulary = pickle.load(f)
    
    # 모델 로드
    encoder = CNNModel(config['embedding_size']).eval()
    decoder = LSTMModel(
        config['embedding_size'],
        config['hidden_size'],
        len(vocabulary),
        config['num_layers']
    )
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # 체크포인트 로드
    encoder.load_state_dict(torch.load(config['encoder_path'], map_location=device))
    decoder.load_state_dict(torch.load(config['decoder_path'], map_location=device))
    
    print(f"Models loaded from {config['encoder_path']} and {config['decoder_path']}")
    
    # 테스트 이미지 리스트
    test_images = config['test_images']
    
    # 각 이미지에 대해 캡션 생성 및 시각화
    for img_path in test_images:
        caption = generate_caption(
            img_path, encoder, decoder, vocabulary, device, transform
        )
        
        print(f"\nImage: {img_path}")
        print(f"Caption: {caption}")
        
        # 이미지와 캡션 시각화
        img = Image.open(img_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(caption, fontsize=14, wrap=True)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 평가 설정
    config = {
        'vocab_path': './data_dir/vocabulary.pkl',
        'encoder_path': './models_dir/encoder-2-3000.ckpt',
        'decoder_path': './models_dir/decoder-2-3000.ckpt',
        'embedding_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'test_images': [
            './test_images/sample1.png',
            './test_images/sample2.png',
            './test_images/sample3.png'
        ]
    }
    
    evaluate_model(config)