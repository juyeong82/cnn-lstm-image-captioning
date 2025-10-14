"""
CNN-LSTM 이미지 캡셔닝 모델 정의
- CNNModel: ResNet-152 기반 인코더
- LSTMModel: LSTM 기반 디코더
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNModel(nn.Module):
    """
    이미지 인코더: ResNet-152 기반
    이미지를 고정 크기의 특징 벡터로 변환
    """
    
    def __init__(self, embedding_size):
        """
        Args:
            embedding_size (int): 출력 특징 벡터의 차원 (예: 256)
        """
        super(CNNModel, self).__init__()
        
        # ImageNet으로 사전 학습된 ResNet-152 로드
        weights = models.ResNet152_Weights.IMAGENET1K_V1
        resnet = models.resnet152(weights=weights)
        
        # 마지막 FC layer 제거 (분류기 부분 제외)
        module_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*module_list)
        
        # 사전 학습된 가중치 동결 (Fine-tuning 안 함)
        for param in self.resnet_module.parameters():
            param.requires_grad = False
        
        # ResNet 출력(2048차원)을 임베딩 크기로 변환
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
    
    def forward(self, input_images):
        """
        Args:
            input_images (Tensor): (batch_size, 3, 224, 224)
        
        Returns:
            Tensor: (batch_size, embedding_size) - 이미지 특징 벡터
        """
        # ResNet 특징 추출: (batch, 2048, 1, 1)
        resnet_features = self.resnet_module(input_images)
        
        # Flatten: (batch, 2048)
        resnet_features = self.flatten(resnet_features)
        
        # Linear 변환: (batch, embedding_size)
        resnet_features = self.linear_layer(resnet_features)
        
        # Batch Normalization: (batch, embedding_size)
        final_features = self.batch_norm(resnet_features)
        
        return final_features


class LSTMModel(nn.Module):
    """
    캡션 디코더: LSTM 기반
    이미지 특징 벡터로부터 순차적으로 단어를 생성하여 캡션 생성
    """
    
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, 
                 num_layers, max_seq_len=20):
        """
        Args:
            embedding_size (int): 단어 임베딩 차원 (인코더 출력과 동일해야 함)
            hidden_layer_size (int): LSTM hidden state 차원 (예: 512)
            vocabulary_size (int): 전체 단어 사전 크기
            num_layers (int): LSTM 레이어 개수
            max_seq_len (int): 생성할 캡션의 최대 길이
        """
        super(LSTMModel, self).__init__()
        
        # 단어 인덱스 → 임베딩 벡터 변환
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        
        # LSTM 레이어
        self.lstm_layer = nn.LSTM(
            embedding_size, 
            hidden_layer_size, 
            num_layers, 
            batch_first=True
        )
        
        # LSTM 출력 → 단어 점수(logits) 변환
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        
        self.max_seq_len = max_seq_len
        
    def forward(self, input_features, capts, lens):
        """
        학습 시 사용: Teacher Forcing 방식
        
        Args:
            input_features (Tensor): (batch, embedding_size) - 이미지 특징
            capts (Tensor): (batch, seq_len) - 정답 캡션 인덱스
            lens (list): 각 캡션의 실제 길이
        
        Returns:
            Tensor: (total_words, vocab_size) - 예측된 단어 점수
        """
        # 캡션 단어들을 임베딩 벡터로 변환
        embeddings = self.embedding_layer(capts)
        
        # 이미지 특징을 캡션 시퀀스 맨 앞에 추가
        # (batch, embedding_size) → (batch, 1, embedding_size)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        
        # 패딩된 시퀀스를 효율적으로 처리하기 위해 packing
        from torch.nn.utils.rnn import pack_padded_sequence
        lstm_input = pack_padded_sequence(
            embeddings, lens, batch_first=True, enforce_sorted=False
        )
        
        # LSTM 순전파
        hidden_variables, _ = self.lstm_layer(lstm_input)
        
        # Linear layer를 통해 단어 점수 계산
        model_outputs = self.linear_layer(hidden_variables[0])
        
        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        """
        추론 시 사용: Greedy Search로 캡션 생성
        
        Args:
            input_features (Tensor): (1, embedding_size) - 이미지 특징
            lstm_states (tuple, optional): 이전 LSTM state
        
        Returns:
            Tensor: (1, max_seq_len) - 생성된 단어 인덱스 시퀀스
        """
        sampled_indices = []
        
        # 첫 입력은 이미지 특징 벡터
        lstm_inputs = input_features.unsqueeze(1)  # (1, 1, embedding_size)
        
        for i in range(self.max_seq_len):
            # LSTM 한 스텝 실행
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)
            
            # 단어 점수 계산
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))
            
            # 가장 높은 점수의 단어 선택 (Greedy)
            _, predicted_outputs = model_outputs.max(1)
            sampled_indices.append(predicted_outputs)
            
            # 예측된 단어를 다음 입력으로 사용
            lstm_inputs = self.embedding_layer(predicted_outputs)
            lstm_inputs = lstm_inputs.unsqueeze(1)  # (1, 1, embedding_size)
        
        # 리스트를 텐서로 변환
        sampled_indices = torch.stack(sampled_indices, 1)  # (1, max_seq_len)
        
        return sampled_indices