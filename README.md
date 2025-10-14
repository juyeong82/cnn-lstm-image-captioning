# 🖼️ CNN-LSTM Image Captioning with PyTorch

[![Python](https://img.shields.io/badge/Python-3.12.7-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **사진 한 장만 보여주면, AI가 문장으로 사진을 묘사합니다.**
>
> 이 프로젝트는 ResNet-152 CNN Encoder와 LSTM Decoder를 결합하여 이미지의 핵심 내용을 파악하고 자연어 캡션을 생성하는 딥러닝 모델을 구현했습니다.

---
## 1. 샘플 예측 결과
---
### Image 1
![Generated Caption Example 1](results/sample_predictions/sample1.png)

---
### Image 2
![Generated Caption Example 2](results/sample_predictions/sample2.png)

---
### Image 3
![Generated Caption Example 3](results/sample_predictions/sample3.png)

---

## 2. 프로젝트 개요

이 프로젝트는 **Encoder-Decoder 아키텍처**를 사용하여 이미지의 내용을 자동으로 설명하는 자연어 문장을 생성합니다.

### 주요 특징
- **CNN Encoder**: 사전 학습된 **ResNet-152** 모델을 사용하여 이미지에서 풍부한 시각적 특징(feature)을 추출합니다.
- **lSTM Decoder**: 추출된 특징 벡터를 초기 입력으로 받아, **LSTM 네트워크**가 문맥에 맞는 단어를 순차적으로 생성합니다.
- **대규모 데이터셋**: **MS COCO 2014** 데이터셋의 약 41만 개 이미지-캡션 쌍을 통해 학습하여 일반화 성능을 높였습니다.

---

### 데이터 흐름 (batch_size=128 기준)
1. **이미지 입력**: `(128, 3, 224, 224)`
2. **CNN 특징 추출**: `(128, 2048)` → Linear → `(128, 256)`
3. **LSTM 입력**: 이미지 벡터 + 단어 임베딩 → `(128, seq_len, 256)`
4. **단어 예측**: `(total_words, vocab_size)` → Softmax → 다음 단어 선택

---

## 3. 학습 결과

### 성능 지표
- **최종 Loss**: 2.1087
- **Perplexity**: 8.2372
- **학습 Epoch**: 2 epochs
- **배치 크기**: 128
- **Vocabulary Size**: 9,948개 단어

---

## 4. 설치 및 실행 방법

### 1. 환경 설정

```

# 레포지토리 클론

git clone https://github.com/juyeong82/cnn-lstm-image-captioning.git

cd cnn-lstm-image-captioning

# 가상 환경 생성 (권장)

python -m venv venv

source venv/bin/activate  # Windows: venvScriptsactivate

# 필수 패키지 설치

pip install -r requirements.txt

# NLTK 데이터 다운로드

python -c "import nltk; [nltk.download](http://nltk.download)('punkt')"

```

### 2. 데이터 준비

```

# COCO 데이터셋 다운로드 (약 13GB)

mkdir -p data_dir

cd data_dir

# 학습 이미지

wget http://images.cocodataset.org/zips/train2014.zip

unzip [train2014.zip](http://train2014.zip)

# Annotation 파일

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_[trainval2014.zip](http://trainval2014.zip)

cd ..

```

### 3. 단어 사전 구축

```

python src/[build_vocab.py]

```


### 4. 모델 학습

```

python src/[train.py]

```

학습 중 로그:
```

Epoch [0/5], Step [0/3236], Loss: 9.2052, Perplexity: 9948.82

Epoch [0/5], Step [10/3236], Loss: 5.7601, Perplexity: 317.36

...

Epoch [1/5], Step [3230/3236], Loss: 2.1087, Perplexity: 8.24

```

**체크포인트 저장**: 1000 스텝마다 `models_dir/encoder-{epoch}-{step}.ckpt`, `decoder-{epoch}-{step}.ckpt` 저장

### 5. 캡션 생성 (추론)

```

python src/[evaluate.py]

```

---

## 🛠️ 기술 스택

### 딥러닝 프레임워크
- **PyTorch 2.0+**: 모델 구현 및 학습
- **torchvision**: 사전 학습된 ResNet-152, 이미지 전처리

### 데이터 처리
- **pycocotools**: COCO 데이터셋 처리
- **NLTK**: 자연어 토큰화 (punkt tokenizer)
- **Pillow**: 이미지 로딩 및 리사이즈

---

## 핵심 개념

### 1. Encoder-Decoder 아키텍처
- **인코더 (CNN)**: 이미지 → 고정 길이 특징 벡터 (256차원)로 압축
- **디코더 (LSTM)**: 특징 벡터 → 단어 시퀀스로 디코딩


### 2. Teacher Forcing
학습 시 이전 예측 단어가 아닌 **정답 단어**를 다음 입력으로 사용하여 학습 속도 향상:
```

# t=1: 이미지 → "A"

# t=2: "A" (정답) → "dog"

# t=3: "dog" (정답) → "playing"

```

### 3. Greedy Search (추론 시)
각 시점에서 **가장 확률이 높은 단어**를 선택하여 문장 생성:
```

for t in range(max_length):

logits = model(input_t)

word = argmax(logits)  # 최대 확률 단어 선택

input_t+1 = embedding(word)

```

### 4. Pack Padded Sequence
가변 길이 캡션을 효율적으로 처리하기 위해 패딩된 시퀀스를 압축:
```

packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

```

---


## 📝 학습 하이퍼파라미터 튜닝
```

# src/[train.py]의 config 수정

config = {

'embedding_size': 256,      # 임베딩 차원 (256, 512 권장)

'hidden_size': 512,         # LSTM hidden size (512, 1024 권장)

'num_layers': 1,            # LSTM 레이어 수

'learning_rate': 0.001,     # 학습률 (0.001~0.0001)

'batch_size': 128,          # 배치 크기 (GPU 메모리에 따라 조정)

}

```

### GPU 메모리 부족 시
- `batch_size` 줄이기 (128 → 64 → 32)
- `num_workers` 줄이기 (2 → 0)
- 이미지 크기 줄이기 (224 → 196)

---

## 👤 개발자

**Juyeong Park**  
- Email: [ju0korea@korea.ac.kr]
- GitHub: [@juyeong82]

