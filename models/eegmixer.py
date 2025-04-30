
import torch
import torch.nn as nn

class EEGMixer(nn.Module):
    def __init__(
            self, n_classes=0, patch_size=10, n_channels=32, embed_dim=64, pooling=False,
            n_layers=3, kernel_size=5, droprate_t=0.25, droprate_s=0.5, droprate_fc=0.0,
        ):
        super().__init__()
        # temporal encoder
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.encoder_t = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size, groups=embed_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(embed_dim)
                )),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(droprate_t),
            ) for _ in range(n_layers)],
            nn.AdaptiveAvgPool1d((1)),
        )
        # spatial encoder
        # self.encoder_s = ConvPooling(embed_dim=embed_dim, n_channels=n_channels, droprate=droprate_s)
        self.encoder_s = MixerPooling(embed_dim=n_channels, ratio=1, droprate=droprate_s, pooling=pooling)
        # classfier head
        self.drop_fc = nn.Dropout(droprate_fc)
        self.feature_dim = embed_dim
        self.fc = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()      

    def forward(self, x):          
        # x: (bs, 1, n_timepoints, n_channels)
        inshape = x.shape
        x = x.permute(0, 3, 1, 2) # (bs, n_channels, 1, n_timepoints)
        x = x.reshape(-1, *x.shape[2:]) # (bs*n_channels, 1, n_timepoints)
        # embed patches
        x = self.patch_embed(x) # (bs*n_channels, embed_dim, n_patches)
        x = self.encoder_t(x) # (bs*n_channels, embed_dim, 1)
        x = x.reshape(inshape[0], inshape[-1], -1) # (bs, n_channels, embed_dim)
        # spatial pooling
        x = self.encoder_s(x)
        # classifier head
        x = self.drop_fc(x)
        x = self.fc(x)
        return x
    
class ConvPooling(nn.Module):
    def __init__(self, embed_dim=64, n_channels=32, droprate=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=n_channels, groups=embed_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(droprate)
        )
    def forward(self, x):
        # x: (B, N, P)
        # output: (B, P)
        x = x.permute(0, 2, 1) # (B, P, N)
        x = self.conv(x)
        x = x.squeeze(2) # (B, P)
        return x
    
class MixerPooling(nn.Module):
    def __init__(self, embed_dim, ratio=1, droprate=0., pooling=True):
        super().__init__()
        self.pooling = pooling
        inner_dim = int(embed_dim * ratio)
        # channel mixing
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, inner_dim, kernel_size=1, groups=embed_dim, bias=False),
            nn.BatchNorm1d(inner_dim),
            nn.GELU(),
            nn.Dropout(droprate),
            nn.Conv1d(inner_dim, embed_dim, kernel_size=1, bias=False),
            nn.Dropout(droprate),
        )

    def forward(self, x):
        # x: (B, N, P)
        # output: (B, N, P) or (B, N*P) determined by flatten
        x = self.conv(x)
        if self.pooling:
            # mean pooling on channels
            x = x.mean(dim=1)
        else:
            x = x.flatten(1)  # (B, N*P)
        return x
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    

if __name__ == '__main__':

    x = torch.randn(5, 1, 512, 30)

    model = EEGMixer(n_classes=4, patch_size=16, n_channels=30)
    output = model(x)
    print(output.shape)
