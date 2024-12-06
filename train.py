import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import wandb
from tqdm import tqdm

from data import KoreanTextDataset, prepare_dataloader
from model import TransformerEncoder


# 클래스 비율에 따라 가중치 설정
def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    weights = [total_count / (num_classes * label_counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Encoder with Imbalanced Data")
    parser.add_argument('--input_dim', type=int, default=200000, help='Vocabulary size for token embedding')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=128, help='Feedforward dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer encoder layers')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension size for each token')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--wandb_project', type=str, default="Honorifics", help='WandB project name')
    parser.add_argument('--exp_name', type=str, default="Honorific_exp", help='WandB project name')
    parser.add_argument('--checkpoint_dir', type=str, default="./log_EXAONE", help='Directory to save model checkpoints')
    return parser.parse_args()


def print_model_info(model):
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Summary:")
    print(model)
    print(f"\nTotal Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    # wandb에 모델 정보 로깅
    wandb.log({"Total Parameters": total_params, "Trainable Parameters": trainable_params})


def train():
    args = parse_args()

    input_dim = 100 
    num_classes = 3
    batch_size = 32
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    
    wandb.init(project=args.wandb_project, name=args.exp_name)
    wandb.config.update(args)

    data = pd.read_csv('./refined_data/completed_output_pair_A.csv') 
    dataset = KoreanTextDataset(data)
    train_loader, val_loader, _ = prepare_dataloader(dataset)

    print("Done loading datasets.")

    model = TransformerEncoder(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim
    ).cuda()

    print_model_info(model)  # 모델 정보 출력

    # labels = [label for _, label in dataset]  
    # labels = data['score'].tolist()  # 'Score'를 실제 레이블 열 이름으로 바꾸세요

    # class_weights = compute_class_weights(labels, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    step = 0
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            
            outputs = model(inputs)
            step+=1
            
            # CrossEntropyLoss에 맞게 출력 차원 조정
            outputs = outputs.reshape(-1, num_classes)  # (batch_size, num_classes)
            targets = targets.reshape(-1)  # (batch_size , )
            # print(outputs.shape)
            # print(targets.shape)
            
            # 손실 계산 및 역전파
            train_loss = criterion(outputs, targets)
            train_loss.backward()
            optimizer.step()

            wandb.log({"train/loss": train_loss.item()})

            running_loss += train_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        # 에폭 당 손실과 정확도 출력
        epoch_loss = running_loss / total
        train_accuracy = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                # outputs = outputs.view(-1, args.output_dim)
                # targets = targets.view(-1)
                outputs = outputs.reshape(-1, num_classes)  # (batch_size, num_classes)
                targets = targets.reshape(-1) 
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

        wandb.log({
            "step": step + 1,
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        }) 

        print(f"Epoch [{epoch+1}/{args.num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
              
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")





if __name__ == "__main__":
    train()