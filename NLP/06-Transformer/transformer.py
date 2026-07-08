"""Transformer.ipynb


"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class PositionalEncoding(nn.Module):
  def __init__(self,d_model,max_len=5000):
    super().__init__()
    pe = torch.zeros(max_len,d_model)
    position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:,0::2]=torch.sin(position * div_term)
    pe[:,1::2]=torch.cos(position * div_term)
    self.register_buffer("pe",pe.unsqueeze(0))

  def forward(self,x):
    return x + self.pe[:, : x.size(1), :]

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model,num_heads) -> None:
    super().__init__()
    assert d_model % num_heads ==0
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k=d_model // num_heads

    self.W_q=nn.Linear(d_model,d_model)
    self.W_k=nn.Linear(d_model,d_model)
    self.W_v=nn.Linear(d_model,d_model)
    self.W_o=nn.Linear(d_model,d_model)

  def split_heads(self,x):
    batch_size,seq_len,_=x.size()
    return x.transpose(1,2).contiguous().view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)

  def combine_heads(self, x):
      batch_size, _, seq_len, _ = x.size()
      return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

  def forward(self,query,key,value,mask=None):
    Q=self.split_heads(self.W_q(query))
    K=self.split_heads(self.W_k(key))
    V=self.split_heads(self.W_v(value))

    score=torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
    if mask is not None:
      score=score.masked_fill(mask==0,float("-inf"))


    attn=F.softmax(score,dim=-1)
    context =torch.matmul(attn , V)
    output=self.W_o(self.combine_heads(context))
    return output ,attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)  # Maskeli self-attention (geleceği görme)
        x = self.norm1(x + self.dropout(attn_out))   # Residual + Dropout + LayerNorm

        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)  # Cross-attention (Q:decoder, K=V:encoder)
        x = self.norm2(x + self.dropout(cross_out))  # Residual + Dropout + LayerNorm

        ff_out = self.feed_forward(x)                # FFN'den geçir
        x = self.norm3(x + self.dropout(ff_out))     # Residual + Dropout + LayerNorm
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        x = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)

src_vocab_size = 1000
tgt_vocab_size = 1000

model = Transformer(
    src_vocab_size, tgt_vocab_size,
    d_model=128, num_heads=4, num_layers=2, d_ff=256, max_len=50
)

src = torch.randint(1, src_vocab_size, (2, 10))
tgt = torch.randint(1, tgt_vocab_size, (2, 12))

out = model(src, tgt)
print("Cikis sekli:", out.shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Toplam parametre sayisi: {total_params:,}")

VOCAB_SIZE = 30
SEQ_LEN = 8
SOS, EOS, PAD = 1, 2, 0
def make_batch(batch_size=32, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
    seq = torch.randint(3, vocab_size, (batch_size, seq_len))
    src = seq
    tgt_full = torch.cat(
        [torch.full((batch_size, 1), SOS), seq, torch.full((batch_size, 1), EOS)], dim=1
    )
    tgt_input = tgt_full[:, :-1]
    tgt_output = tgt_full[:, 1:]
    return src, tgt_input, tgt_output


toy_model = Transformer(
    VOCAB_SIZE, VOCAB_SIZE, d_model=64, num_heads=4, num_layers=2, d_ff=128, max_len=20
)
optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

losses = []
for epoch in range(300):
    src, tgt_input, tgt_output = make_batch()
    optimizer.zero_grad()
    logits = toy_model(src, tgt_input)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 50 == 0 or epoch == 299:
        print(f"Epoch {epoch:3d} | Kayip (loss): {loss.item():.4f}")

@torch.no_grad()
def greedy_decode(model, src, max_len=SEQ_LEN + 2):
    model.eval()
    src_mask, _ = model.generate_mask(src, src)
    enc_output = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), SOS, dtype=torch.long)
    for _ in range(max_len - 1):
        _, tgt_mask = model.generate_mask(src, ys)
        out = model.decode(ys, enc_output, src_mask, tgt_mask)
        logits = model.fc_out(out[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
    model.train()
    return ys


test_src, _, _ = make_batch(batch_size=1)
prediction = greedy_decode(toy_model, test_src)

print("Girdi dizisi   :", test_src.tolist()[0])
print("Model ciktisi  :", prediction.tolist()[0])