"""
유틸리티 함수 모음
- 단어 사전 구축
- 이미지 전처리
- 결과 시각화 등
"""

import os
import pickle
import nltk
from collections import Counter
from pycocotools.coco import COCO
from PIL import Image


class Vocab(object):
    """단어와 인덱스를 매핑하는 단어 사전 클래스"""
    
    def __init__(self):
        self.w2i = {}  # word to index
        self.i2w = {}  # index to word
        self.index = 0

    def __call__(self, token):
        """단어를 인덱스로 변환"""
        if token not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]

    def __len__(self):
        """단어 사전 크기"""
        return len(self.w2i)
    
    def add_token(self, token):
        """단어 사전에 새 토큰 추가"""
        if token not in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


def build_vocabulary(json, threshold):
    """
    COCO JSON 파일로부터 단어 사전 구축
    
    Args:
        json (str): COCO annotation JSON 파일 경로
        threshold (int): 단어 사전에 포함될 최소 등장 빈도
    
    Returns:
        Vocab: 구축된 단어 사전 객체
    """
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    
    # 모든 캡션 토큰화 및 빈도 계산
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # threshold 이상 등장한 단어만 선택
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    # Vocab 객체 생성 및 특수 토큰 추가
    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    # 필터링된 단어들 추가
    for token in tokens:
        vocab.add_token(token)
    
    return vocab


def load_image(image_file_path, transform=None):
    """
    추론용 이미지 로드 및 전처리
    
    Args:
        image_file_path (str): 이미지 파일 경로
        transform (callable, optional): 이미지 변환 함수
    
    Returns:
        Tensor: 전처리된 이미지 텐서
    """
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize([224, 224], Image.Resampling.BILINEAR)

    if transform is not None:
        img = transform(img).unsqueeze(0)  # 배치 차원 추가

    return img


def save_vocabulary(vocab, save_path):
    """단어 사전 저장"""
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {save_path}")


def load_vocabulary(vocab_path):
    """저장된 단어 사전 로드"""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab