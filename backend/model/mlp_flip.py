import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, hidden_dim)
        
    def forward(self, x):
        # Transpose x to shape (batch_size, seq_length, in_channels)
        x = x.transpose(1, 2)  # Now x is (batch_size, input_dim, channels)
        # Perform the unfolding operation
        x = x.unfold(1, self.patch_size, self.patch_size).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x = self.proj(x)  # Project patches to hidden_dim
        return x

class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(hidden_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim)
        )
    
    def forward(self, x):
        # Token mixing
        x = x + self.token_mixer(self.norm1(x))
        # Channel mixing
        x = x + self.channel_mixer(self.norm2(x))
        return x

class MLPMixer1D_flip(nn.Module):
    def __init__(self, input_dim, patch_size, in_channels, hidden_dim, depth, token_dim, channel_dim, output_dim):
        super().__init__()
        num_patches = input_dim // patch_size
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, hidden_dim)
        self.mixer_blocks = nn.Sequential(*[MixerBlock(hidden_dim, token_dim, channel_dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * num_patches),
            nn.Linear(hidden_dim * num_patches, 128),  # Adjusted to match the new first layer dimension
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)  # Assuming output_dim is 5
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mixer_blocks(x)
        x = self.head(x)
        return x