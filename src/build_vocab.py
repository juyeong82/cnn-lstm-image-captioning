# build_vocab.py (새 파일 생성)
from utils import build_vocabulary
import pickle

vocab = build_vocabulary(
    'data_dir/annotations_trainval2014/annotations/captions_train2014.json',
    threshold=4
)

with open('data_dir/vocabulary.pkl', 'wb') as f:
    pickle.dump(vocab, f)
    
print(f"✓ Vocabulary saved: {len(vocab)} words")