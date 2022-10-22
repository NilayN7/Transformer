import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)
        
class Positional_encoding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(Positional_encoding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        return x + torch.Tensor(self.pe[:,:seq_len], requires_grad=False)

class Multi_head_Attention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(Multi_head_attention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim/self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)

        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        query = query.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        k_adjust = k.transpose(-1,-2)

        product = torch.matmul(q, k_adjust)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product/math.sqrt(self.single_head_dim)

        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length, self.n_heads*self.single_head_dim)

        output = self.out(concat)

        return output

class Transformerblock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(Transformerblock, self).__init__()

        self.attention = Multi_head_Attention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
                                        nn.Linear(embed_dim, expansion_factor*embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(expansion_factor*embed_dim, embed_dim)
                                        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feedforward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, ):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = Positional_encoding(seq_len, embed_dim)

        self.layers = nn.ModuleList([Transformerblock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(Decoder, self).__init__()
        self.attention = Multi_head_Attention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = Transformerblock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        attention = self.attention(x, x, x, mask=mask)

        value = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value)

        return out
class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.positional_embedding = Positional_encoding(seq_len, embed_dim)

        self.layers = nn.ModuleList([DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x = self.word_embedding(x)
        x = positional_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)
        out = F.softmax(self.fc(x))
         
        return out

class Transformer(nn.Module):
    def __init__(self, ):
        super(Transformer, self):
        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)

        def make_trg_mask(self, trg):
            batch_size, trg_len = trg.shape
            trg_mask =  torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)

            return trg_mask

        def forward(self, src, trg):
            trg_mask = self.make_trg_mask(trg)
            enc_out = self.encoder(src)

            batch_size, seq_len = trg.shape[0], trg.shape[1]
            outputs = self.decoder(trg, enc_out, trg_mask)
            return outputs