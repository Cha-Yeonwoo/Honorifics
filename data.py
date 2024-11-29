import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from Tokenizer import return_tokenize_with_transformers
from sklearn.model_selection import train_test_split

# 패딩을 적용하는 collate_fn 정의
def collate_fn(batch):
    texts, scores = zip(*batch)
    
    # 모든 텍스트 텐서를 동일한 차원으로 패딩 적용
    # 텍스트 텐서의 차원을 확인하고 필요시 임베딩 차원 추가
    if len(texts[0].shape) == 1:  # 단일 차원 [seq_len]일 경우
        padded_texts = pad_sequence([torch.tensor(t) for t in texts], batch_first=True, padding_value=0)
    else:  # 다차원 [seq_len, embed_dim]일 경우
        max_len = max([t.size(1) for t in texts])
        padded_texts = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0) for t in texts
        ])

        # for t in texts:
        #     a=torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0) 
        #     print(a.shape)

    # scores는 별도의 패딩 필요 없음
    scores = torch.stack(scores)
    
    return padded_texts, scores

class KoreanTextDataset(Dataset):
    def __init__(self, data, tokenizer='BERT'):
        self.korean_text = data['Korean'].values  #
        self.scores = data['score'].values         
        self.tokenizer = tokenizer 

        # self.max_len = 0
        # for text in self.korean_text:
        #     tokenized_text = return_tokenize_with_transformers(text, self.tokenizer)
        #     self.max_len = max(self.max_len, tokenized_text.size(0))

        print(f"Ill data: {data['score'].isnull().sum()}")


    def __len__(self):
        return len(self.korean_text)

    def __getitem__(self, idx):
        korean_text = self.korean_text[idx]
        score = torch.tensor(self.scores[idx]) # dtype=torch.int
        tokenized_text = return_tokenize_with_transformers(korean_text, self.tokenizer)
        # return korean_text, score
        return tokenized_text, score

def prepare_dataloader(dataset):

    # 데이터셋 인덱스를 train, val, test로 분할 (70, 15, 15)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    # Subset을 사용해 train, val, test 데이터셋 생성
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # 각각의 데이터셋에 대해 DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,  collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,  collate_fn=collate_fn)

    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    data = pd.read_csv('./refined_data/output_pair_A.csv')  # 'your_data.csv'를 실제 파일명으로 바꿔주세요
    dataset = KoreanTextDataset(data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 배치 크기와 셔플 여부 설정

    print(len(data_loader))
    a = iter(data_loader)
  