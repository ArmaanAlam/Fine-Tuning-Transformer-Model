import torch
from torch import nn
from transformers import BertTokenizer, BertModel

bert_model_name = "bert-base-uncased"
num_classes = 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class BERTClassifier(nn.Module):

    def __init__(self, model_name, num_classes):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            self.bert.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        output = self.dropout(
            pooled_output
        )

        return self.fc(output)


# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    bert_model_name
)

# Create model
model = BERTClassifier(
    bert_model_name,
    num_classes
).to(device)

# Load checkpoint
checkpoint = torch.load(
    "bert_classifier.pth",
    map_location=device
)

# Load weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Evaluation mode
model.eval()


def predict_sentiment(
        text,
        max_length=128
):

    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(
        device
    )

    attention_mask = encoding[
        'attention_mask'
    ].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(
            outputs,
            dim=1
        )

    return (
        "positive"
        if preds.item() == 1
        else "negative"
    )