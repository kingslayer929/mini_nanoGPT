import torch
import torch.nn as nn
from torch.nn import functional as F
import os

pretrain = "5000.pth"

batch_size = 64 
block_size = 256 
max_iters = 100
eval_interval = 5
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size): # head_size = n_embd / n_head
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 右上三角形都是0，左下三角形都是1

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # (batch_size, block_size, n_embd)
        k = self.key(x) # (batch_size, block_size, head_size)
        q = self.query(x) # (batch_size, block_size, head_size)

        # (batch_size, block_size, head_size) @ (batch_size, head_size, block_size) * head_size**-0.5
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch_size, block_size, block_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) # 轉成機率分佈
        wei = self.dropout(wei)
        
        v = self.value(x) # (batch_size, block_size, head_size)
        # (batch_size, block_size, block_size) @ (batch_size, block_size, head_size)
        out = wei @ v # (batch_size, block_size, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): # head_size = n_embd / n_head
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # num_heads個Head
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 將每個Head的結果cat起來
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # n_layer個block
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # (batch_size, block_size)
        
        tok_emb = self.token_embedding_table(idx) # (batch_size, block_size, n_embd)
        # torch.arange(T) = [0, 1, ... , T-1]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (batch_size, block_size, n_embd)
        x = tok_emb + pos_emb # (batch_size, block_size, n_embd)
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)
            
            # logits依舊是每個長度的idx_cond都會預測，但我們只要預測的最後一個
            logits = logits[:, -1, :] 
            
            probs = F.softmax(logits, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = GPTLanguageModel()
m = model.to(device)

checkpoints = torch.load(os.path.join("checkpoints", pretrain))
m.load_state_dict(checkpoints)

m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))