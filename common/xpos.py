# Copyright (c) 2022 Microsoft
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)
import torch
import torch.nn as nn

class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        
        # Register a buffer for scale values used in position encoding
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def fixed_pos_embedding(self, scale):
        """
        Generate fixed position embeddings (sine and cosine) based on the scale.
        """
        seq_len, dim = scale.shape
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
        sinusoid_inp = (
            torch.einsum("i, j -> ij", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(scale)
        )
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        """
        Rotate every two dimensions of the input tensor.
        """
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        rotated = torch.stack((-x2, x1), dim=-1)
        
        # Handle the case when the last dimension is odd
        if x.shape[-1] % 2 == 1:
            x2 = torch.cat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
        
        return rotated.flatten(-2)

    def duplicate_interleave(self, m):
        """
        Duplicate and interleave the elements of the matrix.
        """
        dim0 = m.shape[0]
        m = m.view(-1, 1)  # Flatten the matrix
        m = m.repeat(1, 2)  # Repeat all elements in the second dimension
        m = m.view(dim0, -1)  # Reshape back to a matrix, interleaving the copies
        return m

    def apply_rotary_pos_emb(self, x, sin, cos, scale=1):
        """
        Apply rotary position embedding to the input tensor.
        """
        sin, cos = map(lambda t: self.duplicate_interleave(t * scale), (sin, cos))
        return (x * cos[:, :x.shape[-1]]) + (self.rotate_every_two(x) * sin)[:, :, :x.shape[-1]]

    def forward(self, x, offset=0, downscale=False):
        """
        Forward pass to apply position encoding to the input tensor.
        """
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        
        # Compute scale for position encoding
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = self.fixed_pos_embedding(scale)

        # Adjust scale, sine, and cosine if necessary
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = self.apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        """
        Forward pass to apply reverse position encoding to the input tensor.
        """
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        
        # Compute scale for reverse position encoding
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = self.fixed_pos_embedding(scale)

        # Adjust scale, sine, and cosine if necessary
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = self.apply_rotary_pos_emb(x, -sin, cos, scale)
        return x

    
# test
if __name__ == "__main__":
    
    x = torch.eye(4).unsqueeze(0)
    xpos = XPOS(4)
    x_rot = xpos(x)
    # apply reverse
    x_rot_rev = xpos.forward(x)

    print(x_rot @ x_rot_rev.transpose(-1, -2))
    