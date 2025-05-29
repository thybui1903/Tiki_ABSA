import os
import argparse
import torch
import pandas as pd
import json
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from evaluation.evaluator import DistilBERTModelEvaluator
from transformers import DistilBertConfig, DistilBertForSequenceClassification
import torch
from config import MODEL_NAME

# ==================== Dataset ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== Load functions ====================
def load_data(data_path):
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def load_model_and_tokenizer(model_path, num_labels=18, pretrained_tokenizer_name=MODEL_NAME):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo model rỗng từ config
    config = DistilBertConfig(num_labels=num_labels)
    model = DistilBertForSequenceClassification(config)

    # Load state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Load tokenizer riêng biệt
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_tokenizer_name)

    return model, tokenizer

# ==================== Evaluate ====================
def evaluate(model, tokenizer, texts, labels, batch_size, max_length, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TextDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluator = DistilBERTModelEvaluator(model, tokenizer, device)
    results = evaluator.evaluate_single_epoch(dataloader)

    # Classification report
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    detailed_report = evaluator.detailed_classification_report(all_labels, all_preds, class_names)
    return {**results, **detailed_report}

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['Tiêu cực', 'Tích cực', 'Bình thường'])
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print("📦 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.num_labels)

        print("📄 Loading data...")
        texts, labels = load_data(args.data_path)

        print("🔍 Evaluating...")
        results = evaluate(model, tokenizer, texts, labels, args.batch_size, args.max_length, args.class_names)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"evaluation_{timestamp}.json")

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Evaluation completed! Results saved to {out_path}")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    main()
