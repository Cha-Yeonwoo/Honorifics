import torch
from torch.optim import AdamW
# from transformers import AutoModelForSequenceClassification, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import wandb
import os
import pandas as pd
import argparse

from data import EnKoDataset, prepare_dataloader
from model import TransformerEncoder

parser = argparse.ArgumentParser(description="Fine-tune EXAONE")



parser.add_argument('--exp_name', type=str, default="trans_exp", help='WandB project name')
parser.add_argument('--checkpoint_dir', type=str, default="./log_trans", help='Directory to save model checkpoints')

args = parser.parse_args()

# WandB 초기화
wandb.init(project="honorifics", name=args.exp_name, config={
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 3
})



# GPU 또는 CPU 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("./ckpts")
model = AutoModelForCausalLM.from_pretrained("./ckpts")
model.to(device)

Honorific_model = TransformerEncoder(input_dim=200000)
Honorific_model.load_state_dict(torch.load('/home/elicer/Honorifics/log_EXAONE/model_epoch_10.pth')['model_state_dict'])

data = pd.read_csv('./refined_data/completed_output_pair_A.csv') 
dataset = EnKoDataset(data, tokenizer)
train_loader, val_loader, _ = prepare_dataloader(dataset)

# 옵티마이저 및 학습 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
num_training_steps = wandb.config.num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 손실 함수 정의
criterion = torch.nn.MSELoss()

# 체크포인트 디렉토리 설정
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# Honorific_model = TransformerEncoder(input_dim=100)
# Honorific_model.load_state_dict(torch.load("/home/elicer/Honorifics/log/model_epoch_4.pth")
Honorific_model = Honorific_model.cuda()

# # Freeze Hnonorific model
# for param in Honorific_model.parameters():
#     param.requires_grad = False


if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

def preprocess_text(input_text):
    prompt = f"""다음 문장을 한국어로 번역해줘.
    {input_text}

    '다음은 번역 결과입니다.'같은 말 넣지 말고 그냥 번역한 문장만 출력해줘."""

    messages = [
        {"role": "system", "content": "You are a translator that translates from English to Korean."},
        {"role": "user", "content": prompt}
    ]

    # 인풋으로 받은 영어 문장+프롬프트 인코딩
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    return input_ids

def postprocess_text(output):
    output_text = tokenizer.decode(output)

    start_idx = output_text.find("[|assistant|]") + len("[|endofturn|]")
    output_text = output_text[start_idx:].strip()  # 앞 제거

    endofturn_idx = output_text.rfind("[|endofturn|]")
    translated_text = output_text[:endofturn_idx].strip() # 뒤 제거

    pure_tokens = tokenizer.encode(translated_text)
    # print(pure_tokens)


    return torch.tensor(pure_tokens).unsqueeze(0)


# Fine-Tuning 루프
for epoch in range(wandb.config.num_epochs):
    print(f"Epoch {epoch + 1}/{wandb.config.num_epochs}")

    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc="Training"):
        # inputs = {key: val.to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
        # labels = batch["labels"].to(device)
        
        en, ko = batch

        bs, _, _ = en.size()

        en = en.squeeze(1)
       
        outputs = model.generate(
            en.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128
        )

        output_list = [postprocess_text(outputs[i]) for i in range(bs)]

        max_len = max([t.size(1) for t in output_list])
        print(max_len)
        # for t in output_list:
        #     print(torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0))
        output_tokens= torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0) for t in output_list
        ])

        # print(output_tokens.shape)

        # print(ko.shape)
        GT_honorific = Honorific_model(ko.cuda())
        predicted_honorific = Honorific_model(output_tokens.cuda())

        # print(GT_honorific.shape)
        # print(predicted_honorific.shape)

        loss = criterion(GT_honorific, predicted_honorific)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # WandB에 현재 배치의 loss 기록
        wandb.log({"batch_train_loss": loss.item()})
        _, predicted = torch.max(predicted_honorific, 1)
        correct += (predicted == GT_honorific.argmax(-1)).sum().item()
        total += GT_honorific.size(0)

    avg_train_loss = train_loss / len(train_loader)
    acc = correct/total
    print(f"Training loss: {avg_train_loss:.4f}")
    wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1, "train_acc": acc})

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):

            en, ko = batch

            bs, _, _ = en.size()

            en = en.squeeze(1)
        
            outputs = model.generate(
                en.to("cuda"),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=128
            )

            output_list = [postprocess_text(outputs[i]) for i in range(bs)]

            max_len = max([t.size(1) for t in output_list])
            print(max_len)
            # for t in output_list:
            #     print(torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0))
            output_tokens= torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.size(1)), "constant", 0) for t in output_list
            ])

            # print(output_tokens.shape)

            # print(ko.shape)
            GT_honorific = Honorific_model(ko.cuda())
            predicted_honorific = Honorific_model(output_tokens.cuda())

            loss = criterion(GT_honorific, predicted_honorific)
            val_loss += loss.item()

            _, predicted = torch.max(predicted_honorific, 1)
            correct += (predicted == GT_honorific.argmax(-1)).sum().item()
            total += GT_honorific.size(0)


    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    wandb.log({"epoch_val_loss": avg_val_loss, "val_accuracy": accuracy, "epoch": epoch + 1})

    # 모델 체크포인트 저장
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    wandb.log({"checkpoint_path": checkpoint_path})

# Test 성능 평가
# model.eval()
# test_loss = 0
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Testing"):
#         inputs = {key: val.to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
#         labels = batch["labels"].to(device)

#         outputs = model(**inputs)
#         loss = criterion(outputs.logit, labels)
#         test_loss += loss.item()

#         predictions = torch.argmax(outputs.logits, dim=-1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)

# avg_test_loss = test_loss / len(test_loader)
# test_accuracy = correct / total
# print(f"Test loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
# wandb.log({"test_loss": avg_test_loss, "test_accuracy": test_accuracy})

# Fine-tuned 모델 저장
final_model_path = "./fine_tuned_model"
model.save_pretrained(final_model_path)
print(f"Fine-tuned model saved at {final_model_path}")
wandb.log({"final_model_path": final_model_path})

# WandB 종료
wandb.finish()
