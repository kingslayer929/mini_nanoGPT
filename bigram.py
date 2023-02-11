import torch
import torch.nn as nn
from torch.nn import functional as F

# parameters
batch_size = 32 
block_size = 8 
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# 設定亂數種子
torch.manual_seed(2187)

# open dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 看字元種類數
chars = sorted(list(set(text)))
vocab_size = len(chars) # vocab_size = 65，在使用input.txt時

# encoder and decoder
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# 拆成train and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# 依照batch size、block size取得batch
# batch size = 2, block size = 4
# x:
# tensor([[43, 54, 53, 57],
#         [39, 50, 50,  1]])
# y:
# tensor([[54, 53, 57, 43],
#         [50, 50,  1, 39]])
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 只計算loss，沒有要train
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    # idx is x in get_batch()
    # targets is y in get_batch()
    def forward(self, idx, targets=None):    
        logits = self.token_embedding_table(idx) # (B, T, C), C = 65，在使用input.txt時

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            logits, loss = self(idx)
            
            logits = logits[:, -1, :] 
            
            probs = F.softmax(logits, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# 優化器使用AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 生成結果
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
