# lab_activity_1_mini_transformer_imdb.py
# Mini Transformer Encoder → IMDB subset classification + attention visualization

import os, random, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ------------------------------
# Reproducibility & Device
# ------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------
# Hyperparameters (keep small for speed)
# ------------------------------
MODEL_CKPT = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4

# Mini model dims
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
INTERMEDIATE_SIZE = 256
DROPOUT = 0.1
NUM_LABELS = 2

# ------------------------------
# Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

# ------------------------------
# Dataset (IMDB tiny subset)
# ------------------------------
print("Loading IMDB (small subset for speed)...")
imdb = load_dataset("imdb")

# Select small subsets for fast training
train_small = imdb["train"].shuffle(seed=SEED).select(range(1000))  # 1k
test_small  = imdb["test"].shuffle(seed=SEED).select(range(200))    # 200

train_small = train_small.map(tokenize_batch, batched=True, remove_columns=["text"])
test_small  = test_small.map(tokenize_batch, batched=True, remove_columns=["text"])

# Set PyTorch format
cols = ["input_ids", "attention_mask", "label"]
train_small.set_format(type="torch", columns=cols)
test_small.set_format(type="torch", columns=cols)

train_loader = DataLoader(train_small, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_small,  batch_size=BATCH_SIZE)

# ------------------------------
# Mini Transformer Components
# ------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None, return_weights=False):
        # Q, K, V: [B, T, d]
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)  # [B, T, T]
        if mask is not None:
            # mask: [B, 1, T] or [B, T] -> broadcast to [B, T, T] for encoder
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        output = torch.bmm(weights, V)       # [B, T, d]
        if return_weights:
            return output, weights
        return output

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, mask=None, return_weights=False):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        return self.attn(Q, K, V, mask=mask, return_weights=return_weights)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, return_all_weights=False):
        # Concatenate heads
        if return_all_weights:
            head_outputs = []
            head_weights = []
            for h in self.heads:
                out, w = h(x, mask=mask, return_weights=True)
                head_outputs.append(out)
                head_weights.append(w)
            x = torch.cat(head_outputs, dim=-1)
            x = self.out_proj(x)
            # stack weights: [num_heads, B, T, T]
            head_weights = torch.stack(head_weights, dim=0)
            return x, head_weights
        else:
            x = torch.cat([h(x, mask=mask, return_weights=False) for h in self.heads], dim=-1)
            x = self.out_proj(x)
            return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, intermediate_dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, intermediate_dim)
        self.lin2 = nn.Linear(intermediate_dim, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.lin2(self.act(self.lin1(x))))

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_positions=512, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_positions, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(pos_ids)
        x = self.layer_norm(x)
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mha   = MultiHeadAttention(embed_dim, num_heads)
        self.ffn   = FeedForward(embed_dim, intermediate_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        y_in = self.norm1(x)
        if return_attn:
            y, weights = self.mha(y_in, mask=mask, return_all_weights=True)
        else:
            y = self.mha(y_in, mask=mask)
            weights = None
        x = x + self.drop(y)
        y = self.ffn(self.norm2(x))
        x = x + self.drop(y)
        if return_attn:
            return x, weights
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, intermediate_dim, max_positions=512, dropout=0.1):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim, max_positions=max_positions, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, intermediate_dim, dropout)
        for _ in range(num_layers)])

    def forward(self, input_ids, attention_mask=None, return_all_attn=False):
        x = self.embeddings(input_ids)
        attn_weights_all = []
        for i, layer in enumerate(self.layers):
            if return_all_attn:
                x, w = layer(x, mask=attention_mask, return_attn=True)
                attn_weights_all.append(w)  # [num_heads, B, T, T]
            else:
                x = layer(x, mask=attention_mask, return_attn=False)
        if return_all_attn:
            # list length = num_layers
            return x, attn_weights_all
        return x

class MiniTransformerForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, intermediate_dim, num_labels, max_positions=512, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            max_positions=max_positions,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, return_all_attn=False):
        # attention_mask here uses 1 for tokens to keep, 0 for pad; expand to [B, 1, T]
        if attention_mask is not None:
            # For encoder self-attn, we'll broadcast mask to [B, T, T] inside head via masked_fill on scores
            enc_mask = attention_mask  # [B, T]
        else:
            enc_mask = None

        if return_all_attn:
            hidden, attns = self.encoder(input_ids, attention_mask=enc_mask, return_all_attn=True)
        else:
            hidden = self.encoder(input_ids, attention_mask=enc_mask, return_all_attn=False)
            attns = None

        # CLS pooling: take token 0
        pooled = hidden[:, 0, :]
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "attentions": attns, "pooled": pooled}

# ------------------------------
# Initialize model
# ------------------------------
hf_config = AutoConfig.from_pretrained(MODEL_CKPT)
vocab_size = hf_config.vocab_size
max_pos = 512

model = MiniTransformerForSequenceClassification(
    vocab_size=vocab_size,
    embed_dim=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    intermediate_dim=INTERMEDIATE_SIZE,
    num_labels=NUM_LABELS,
    max_positions=max_pos,
    dropout=DROPOUT
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------------------
# Training / Evaluation helpers
# ------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)
        out = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = out["loss"]
        total_loss += loss.item() * input_ids.size(0)
        logits = out["logits"]
        pred = logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        gts.extend(labels.cpu().tolist())
    acc = accuracy_score(gts, preds)
    return total_loss / len(loader.dataset), acc, gts, preds

# ------------------------------
# Train
# ------------------------------
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc, gts, preds = evaluate(model, test_loader)
    dt = time.time() - t0
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

print("Classification report (test):")
print(classification_report(gts, preds, target_names=["neg","pos"], digits=4))

# ------------------------------
# Attention Visualization
# ------------------------------
@torch.no_grad()
def visualize_attention_for_sample(model, dataset, idx=0, layer_idx=0, head_idx=0, save_path="attention_map.png"):
    model.eval()
    sample = dataset[idx]
    input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)         # [1, T]
    attn_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)    # [1, T]
    out = model(input_ids, attention_mask=attn_mask, return_all_attn=True)
    attns = out["attentions"]   # list of len=num_layers, each: [num_heads, B, T, T]
    weights = attns[layer_idx][head_idx, 0].detach().cpu()  # [T, T]

    # Decode tokens for axis labels (truncate for readability)
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"].tolist())
    T = min(len(tokens), 30)
    tokens = tokens[:T]
    weights = weights[:T, :T]

    plt.figure(figsize=(7, 6))
    plt.imshow(weights, aspect="auto")
    plt.colorbar()
    plt.title(f"Layer {layer_idx} Head {head_idx} — Self-Attention")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    # Reduce clutter: show every 2nd token label if long
    step = 1 if T <= 20 else 2
    plt.xticks(range(0, T, step), tokens[::step], rotation=90)
    plt.yticks(range(0, T, step), tokens[::step])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved attention heatmap to: {save_path}")

# Visualize first test sample, first layer/head
visualize_attention_for_sample(
    model, test_small, idx=0, layer_idx=0, head_idx=0, save_path="attention_map.png"
)

print("Done.")

