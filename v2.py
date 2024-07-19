import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBED = 32
HEAD_SIZE = 16

# For reproducability
torch.manual_seed(1337)

# Load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating the mapping (kinda) 'operation'
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[i] for i in l]

# Split the dataset into training and testing
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split=='train' else val_data

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split=split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)

    def forward(self, x):
        # Output of the self attention itself
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # And the applying the projection
        # Here is projecting back into the residual pathway
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity:

    So as the self-attention is doing the communication, and gather
    information from each other this feed forward layer will be come 
    afterwards to make the tokens to think about themselves. As this 
    layer is per token level, tokens do that independently.
    
    """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # Adding 4 is because of the attention is 
            # all you need paper mention
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        # Coomunication
        self.sa = MultiHeadAttention(n_head, head_size)
        # Computation
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        # check out "Deep Residual Learning for Image Recognition" on
        # gpt-dev.ipynb
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        # positional embedding
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        # self.sa_head = MultiHeadAttention(4, N_EMBED//4)
        # self.ffwd = FeedForward(N_EMBED)
        self.blocks = nn.Sequential(
            Block(N_EMBED, n_head=4),
            Block(N_EMBED, n_head=4),
            Block(N_EMBED, n_head=4),
        )
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        x = tok_embed + pos_emb # (B, T, C)
        # x = self.sa_head(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshaping the logits and targets as the cross_entropy function
            # on pytorch expects it as shown below
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
        
            idx_cond = idx[:, -BLOCK_SIZE:]

            logits, _ = self(idx_cond) # this will run forward func above
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# Create the model
model = BigramLanguageModel()
m = model.to(DEVICE)

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):

    # For evvaluation part
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    # Get a batch
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(''.join(decode(m.generate(context, max_new_tokens=300).tolist()[0])))
