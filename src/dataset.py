"""
COCO 데이터셋 로더 및 전처리
CustomCocoDataset과 DataLoader 정의
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nltk
from pycocotools.coco import COCO


class CustomCocoDataset(Dataset):
    """
    COCO 데이터셋을 PyTorch Dataset 형식으로 래핑
    """
    
    def __init__(self, imgs_path, coco_json_path, vocabulary, transform=None):
        """
        Args:
            imgs_path (str): 이미지 폴더 경로
            coco_json_path (str): COCO annotation JSON 파일 경로
            vocabulary (Vocab): 단어 사전 객체
            transform (callable, optional): 이미지 변환 함수
        """
        self.root = imgs_path
        self.coco_data = COCO(coco_json_path)
        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        특정 인덱스의 (이미지, 캡션) 쌍 반환
        
        Returns:
            image (Tensor): 전처리된 이미지
            caption (Tensor): 단어 인덱스로 변환된 캡션
        """
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        
        # 캡션 ID로 캡션과 이미지 ID 가져오기
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]['caption']
        image_id = coco_data.anns[annotation_id]['image_id']
        
        # 이미지 파일 경로
        image_path = coco_data.loadImgs(image_id)[0]['file_name']
        
        # 이미지 로드 및 RGB 변환
        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        
        # Transform 적용 (리사이즈, 정규화 등)
        if self.transform is not None:
            image = self.transform(image)

        # 캡션을 단어 인덱스 리스트로 변환
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_indices = []
        
        # <start> 토큰 추가
        caption_indices.append(vocabulary('<start>'))
        
        # 각 단어를 인덱스로 변환
        caption_indices.extend([vocabulary(token) for token in word_tokens])
        
        # <end> 토큰 추가
        caption_indices.append(vocabulary('<end>'))
        
        # 텐서로 변환
        ground_truth = torch.Tensor(caption_indices)
        
        return image, ground_truth


def collate_function(data_batch):
    """
    가변 길이 캡션들을 하나의 배치로 묶기 위한 collate 함수
    
    Args:
        data_batch (list): [(image, caption), ...] 리스트
    
    Returns:
        images (Tensor): (batch_size, 3, 224, 224)
        targets (Tensor): (batch_size, padded_length)
        lengths (list): 각 캡션의 실제 길이
    """
    # 캡션 길이 기준 내림차순 정렬 (pack_padded_sequence 요구사항)
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    
    # 이미지와 캡션 분리
    imgs, caps = zip(*data_batch)
    
    # 이미지를 하나의 텐서로 스택
    imgs = torch.stack(imgs, 0)
    
    # 캡션 길이 저장
    cap_lens = [len(cap) for cap in caps]
    
    # 가장 긴 캡션 길이에 맞춰 제로 패딩
    tgts = torch.zeros(len(caps), max(cap_lens)).long()
    
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i, :end] = cap[:end]
    
    return imgs, tgts, cap_lens


def get_loader(imgs_path, coco_json_path, vocabulary, transform, 
               batch_size, shuffle, num_workers):
    """
    DataLoader 생성 헬퍼 함수
    
    Returns:
        DataLoader: 배치 단위 데이터 로더
    """
    coco_dataset = CustomCocoDataset(
        imgs_path=imgs_path,
        coco_json_path=coco_json_path,
        vocabulary=vocabulary,
        transform=transform
    )
    
    custom_data_loader = DataLoader(
        dataset=coco_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_function
    )
    
    return custom_data_loader