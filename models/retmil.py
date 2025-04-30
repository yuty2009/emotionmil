
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.xpos import XPOS


class RoPEEmbedding1d(nn.Module):
    """ 1D Rotary Positional Embedding
    d_model: output dimension for each position
    max_len: maximum position index
    """
    def __init__(self, d_model, max_len=2048, base=10000, cls_token=False):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        pos = torch.arange(max_len).float().unsqueeze(1)
        pos *= 1 / 4
        freqs = pos * inv_freq
        pe = torch.cat((freqs, freqs), dim=-1)
        pe_sin = pe.sin()
        pe_cos = pe.cos()
        if cls_token:
            pe_sin = torch.cat([torch.zeros([1, d_model]), pe_sin], axis=0)
            pe_cos = torch.cat([torch.zeros([1, d_model]), pe_cos], axis=0)
        self.register_buffer('pe_sin', pe_sin)
        self.register_buffer('pe_cos', pe_cos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume x.shape is (batch, t, d_model)
        # refer to https://mp.weixin.qq.com/s/NN5skAwIhuwIMgSbsu6Guw
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x2 = x2.view(x.size())
        out = x * self.pe_cos[:x.size(1)] + x2 * self.pe_sin[:x.size(1)]
        return out


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        
        self.xpos = XPOS(head_size)

    def forward(self, X, mask=None, return_attn=False, cls_token=False):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length, cls_token=cls_token).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        scores = Q @ K.permute(0, 2, 1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'),)

        p_attn = F.softmax(scores, dim=-1)

        p_attn = p_attn * D.unsqueeze(0)
        
        if return_attn:
            return p_attn @ V, p_attn
        else:
            return p_attn @ V
        
    def _get_D(self, sequence_length, causal=False, cls_token=False):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        if causal:
            # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
            D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        else:
            # Bidirectional like RMT
            D = self.gamma ** abs(n - m)
        # fill the NaN with 0
        D[D != D] = 0
        # cls_token
        if cls_token:
            D[0, :] = self.gamma
            D[:, 0] = self.gamma

        return D
    

class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X, mask=None, return_attn=False, cls_token=False):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y, Ret = [], []
        for i in range(self.heads):
            Y_i, Ret_i = self.retentions[i](X, mask, return_attn=return_attn, cls_token=cls_token)
            Y.append(Y_i)
            Ret.append(Ret_i.unsqueeze(1))
        attn_weights = torch.cat(Ret, dim=1)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        if return_attn:
            return (self.swish(X @ self.W_G) * Y) @ self.W_O, attn_weights
        else:
            return (self.swish(X @ self.W_G) * Y) @ self.W_O
        

class RetLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = MultiScaleRetention(
            hidden_size = dim,
            heads = 8,
            double_v_dim = False,
        )

    def forward(self, x, mask=None):
        x1, attn = self.attn(self.norm(x), mask, return_attn=True, cls_token=True)
        x = x + x1
        return x, attn
    

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj0 = nn.Conv1d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        B, L, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2) # B x C x L
        x = cnn_feat + self.proj0(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.transpose(1, 2) # B x L x C
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
    
    
class RetMIL(nn.Module):
    def __init__(self, encoder, num_classes=2,
                 embed_dim=128, norm_layer=nn.LayerNorm):
        super(RetMIL, self).__init__()

        self.encoder = encoder
        encoder_dim = encoder.feature_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.pos_layer = PPEG(dim=embed_dim)
        self.pos_layer = RoPEEmbedding1d(d_model=embed_dim, max_len=1024, cls_token=True)
        self.trans_layer1 = RetLayer(dim=embed_dim)
        self.trans_layer2 = RetLayer(dim=embed_dim)
        self.norm = norm_layer(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None, extra_data=None):
        """ 
        x: (B, N, C, H, W) batch_size x num_instances x channels x height x width
        """
        inshape = x.shape
        x = x.view(-1, *inshape[2:])

        if extra_data is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, extra_data)
        x = x.view(x.size(0), -1)
        x = self.feature_projector(x)  # B*N x P
        x = x.view(inshape[0], inshape[1], -1)  # B x N x P

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        mask_token = torch.ones(x.size(0), 1, dtype=torch.bool).to(x.device)
        mask = torch.cat((mask_token, mask), dim=1) # (bs, seq_length+1)

        # apply Transformer blocks
        x = self.pos_layer(x)
        x, attn1 = self.trans_layer1(x, mask=~mask)
        x, attn2 = self.trans_layer2(x, mask=~mask)

        x = self.norm(x)
        x = x[:, 0] # pooling by cls token

        output = self.classifier(x)

        return output, attn2
    

class RetMIL1(nn.Module):
    def __init__(self, encoder, num_classes=2,
                 embed_dim=128, norm_layer=nn.LayerNorm):
        super(RetMIL1, self).__init__()

        self.encoder = encoder
        encoder_dim = encoder.feature_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_layer = PPEG(dim=embed_dim)
        self.trans_layer1 = RetLayer(dim=embed_dim)
        self.trans_layer2 = RetLayer(dim=embed_dim)
        self.norm = norm_layer(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None, extra_data=None):
        """ 
        x: (B, N, C, H, W) batch_size x num_instances x channels x height x width
        """
        inshape = x.shape
        x = x.view(-1, *inshape[2:])

        if extra_data is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, extra_data)
        x = x.view(x.size(0), -1)
        x = self.feature_projector(x)  # B*N x P
        x = x.view(inshape[0], inshape[1], -1)  # B x N x P

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        mask_token = torch.ones(x.size(0), 1, dtype=torch.bool).to(x.device)
        mask = torch.cat((mask_token, mask), dim=1) # (bs, seq_length+1)

        # apply Transformer blocks
        x, attn1 = self.trans_layer1(x, mask=~mask)
        x = self.pos_layer(x)
        x, attn2 = self.trans_layer2(x, mask=~mask)

        x = self.norm(x)
        x = x[:, 0] # pooling by cls token

        output = self.classifier(x)

        return output, attn2
