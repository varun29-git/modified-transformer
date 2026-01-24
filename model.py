import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#####################################################################################
 
class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Removed redundant self assignments
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # Not scaled by root of d_model
        return self.embedding(x)
    
#####################################################################################

class RMSNorm(nn.Module):

    def __init__(self, d_model, epsilon=10**-8):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon 

        # RMSNorm normalizes scale only (no mean centering), so no bias term is needed
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):

        # Compute the mean of the squared values along the last dimension.
        mean = x.square()
        mean = mean.mean(dim=-1, keepdim=True)

        # Add epsilon and take the square root to get the RMS denominator.
        mean += self.epsilon
        RMS = mean.sqrt()
        
        # Divide x by the RMS
        x = x/RMS
        x = x * self.gamma

        return x

#####################################################################################

class FeedForward(nn.Module):

    def __init__(self, d_model, dropout, d_ff):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.Linear2 = nn.Linear(d_ff, d_model, bias=False) # No need of bias 

    def forward(self, x):

        # FFN(x) = silu(W1x + b1) W2
        return self.Linear2(self.dropout(F.silu(self.Linear1(x))))

#####################################################################################

class RotaryMultiHeadAttention(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0, "D_model has to be divisible by number of heads"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def Attention(query, key, values, mask, Dropout: nn.Dropout):

        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e20)
        attention_scores = attention_scores.softmax(dim = -1) # ((Q @ K.T)/ d_model  ** 0.5)

        if Dropout is not None:
            attention_scores = Dropout(attention_scores)
        
        return (attention_scores @ values), attention_scores
    
    @staticmethod
    def apply_rope(x, sin, cos):
        # x: (B, h, T, d_k)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_even * cos - x_odd * sin
        x_rot[..., 1::2] = x_even * sin + x_odd * cos

        return x_rot
    
    @staticmethod
    def get_rope_sin_cos(T, d_k, device):
        assert d_k % 2 == 0

        pos = torch.arange(T, device=device)          # (T,)
        dim = torch.arange(0, d_k, 2, device=device)  # (d_k/2,)

        inv_freq = 1.0 / (10000 ** (dim / d_k))
        angles = pos[:, None] * inv_freq[None, :]     # (T, d_k/2)

        sin = angles.sin()[None, None, :, :]  # (1, 1, T, d_k/2)
        cos = angles.cos()[None, None, :, :]  # (1, 1, T, d_k/2)

        return sin, cos

    def forward(self, x, mask):

        B, T, _ = x.shape

        query = self.w_q(x) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)
        key = self.w_k(x) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)
        value = self.w_v(x) # (Batch, Seq_len, D_model) -> (Batch, Seq_len, D_model)

        # Reshape to form Heads
        query = query.reshape(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.reshape(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.reshape(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Get sin and cos
        sin, cos = RotaryMultiHeadAttention.get_rope_sin_cos(T, self.d_k, x.device)

        # Applying Rotational Positional Embeddings 
        query = RotaryMultiHeadAttention.apply_rope(query, sin, cos)
        key = RotaryMultiHeadAttention.apply_rope(key, sin, cos)

        out, attn = self.Attention(query, key, value, mask, self.dropout)
        out = out.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(out)


#---------------------------------------------------------------------------------------------------------------------------

class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#####################################################################################

class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        self_attention: RotaryMultiHeadAttention,
        feed_forward: FeedForward,
        dropout
    ):
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x, mask):
        # Self-attention + residual
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, mask)
        )

        # Feed-forward + residual
        x = self.residual_connections[1](
            x, self.feed_forward
        )

        return x


#---------------------------------------------------------------------------------------------------------------------------
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


#---------------------------------------------------------------------------------------------------------------------------
# Projection Class

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


#---------------------------------------------------------------------------------------------------------------------------

class Transformer(nn.Module):

    def __init__(
        self,
        decoder: Decoder,
        token_embedding: InputEmbedding,
        projection: ProjectionLayer
    ):
        super().__init__()
        self.decoder = decoder
        self.token_embedding = token_embedding
        self.projection = projection

    def forward(self, x, mask):
        x = self.token_embedding(x)
        x = self.decoder(x, mask)
        return self.projection(x)

#---------------------------------------------------------------------------------------------------------------------------

def build_transformer(vocab_size, d_model=512, h=8, N=6, d_ff=2048, dropout=0.1):
    
    # Create embedding and projection layers
    token_embedding = InputEmbedding(vocab_size, d_model)
    projection = ProjectionLayer(d_model, vocab_size)
    
    # Create N decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention = RotaryMultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, dropout, d_ff)
        decoder_block = DecoderBlock(d_model, self_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create decoder with all blocks
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)
    
    # Create the complete transformer
    model = Transformer(decoder, token_embedding, projection)
    
    # Initialize weights using Xavier uniform initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return model

    
