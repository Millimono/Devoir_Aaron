from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size 

        # Convolution 2D qui sert de projection linéaire des patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) 

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch, num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."

        # Appliquer la convolution pour obtenir les embeddings des patches
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]

        # Réarranger pour obtenir [B, N, C] où N = num_patches et C = embed_dim
        x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, N] -> [B, N, embed_dim]

        return x



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self, dim, seq_len, token_mixing_mlp_width, channel_mixing_mlp_width,
        activation='gelu', drop=0., drop_path=0.
    ):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        
        # Token Mixing MLP
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = Mlp(seq_len, token_mixing_mlp_width, act_layer=act_layer, drop=drop)
        
        # Channel Mixing MLP
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = Mlp(dim, channel_mixing_mlp_width, act_layer=act_layer, drop=drop)

    def forward(self, x):
            """
            :param x: Tensor de forme (batch, seq_len, dim)
            :return: Tensor de même forme (batch, seq_len, dim)
            """
            # Token Mixing
            print("Avant Token Mixing :", x.shape)
            x = x + self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            print("Après Token Mixing :", x.shape)

            # Channel Mixing
            print("Avant Channel Mixing :", x.shape)
            x = x + self.mlp_channels(self.norm2(x))
            print("Après Channel Mixing :", x.shape)

            return x
       


# class MixerBlock(nn.Module):
#     """ Residual Block w/ token mixing and channel MLPs
#     Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
#     """
#     def __init__(
#             self, dim, seq_len, mlp_ratio=(0.5, 4.0),
#             activation='gelu', drop=0., drop_path=0.):
#         super(MixerBlock, self).__init__()
#         act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
#         tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        
#         self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
#         self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
#         self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         """
#         :param x: Tensor de forme (batch, seq_len, dim)
#         :return: Tensor de même forme (batch, seq_len, dim)
#         """
#         # Token Mixing
#         print("Avant Token Mixing :", x.shape)
#         x = x + self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
#         print("Après Token Mixing :", x.shape)

#         # Channel Mixing
#         print("Avant Channel Mixing :", x.shape)
#         x = x + self.mlp_channels(self.norm2(x))
#         print("Après Channel Mixing :", x.shape)

#         return x


class MLPMixer(nn.Module):
    def __init__(
        self, num_classes, img_size, patch_size, embed_dim, num_blocks,
        token_mixing_mlp_width, channel_mixing_mlp_width,
        drop_rate=0., activation='gelu'
    ):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches,
                token_mixing_mlp_width=token_mixing_mlp_width,
                channel_mixing_mlp_width=channel_mixing_mlp_width,
                activation=activation, drop=drop_rate
            )
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """
        Forward pass du MLPMixer
        :param images: Tensor de shape [batch, 3, img_size, img_size]
        :return: Tensor de shape [batch, num_classes]
        """
        # Étape 1 : Patch Embedding
        x = self.patchemb(images)  # [batch, num_patches, embed_dim]

        # Étape 2 : Passer dans les Mixer Blocks
        x = self.blocks(x)  # [batch, num_patches, embed_dim]

        # Étape 3 : Normalisation
        x = self.norm(x)  # [batch, num_patches, embed_dim]

        # Étape 4 : Global Average Pooling sur les patches
        x = x.mean(dim=1)  # [batch, embed_dim]

        # Étape 5 : Classification
        logits = self.head(x)  # [batch, num_classes]

        return logits

    def visualize(self, logdir):
        """
        Fonction à implémenter pour visualiser les couches du modèle.
        Peut être utilisé avec TensorBoard ou Matplotlib.
        """
        raise NotImplementedError
    

# class MLPMixer(nn.Module):
#     def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks,
#                  drop_rate=0., activation='gelu'):
#         super(MLPMixer, self).__init__()

#         self.patchemb = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim
#         )

#         self.blocks = nn.Sequential(*[
#             MixerBlock(dim=embed_dim, seq_len=self.patchemb.num_patches,
#                        activation=activation, drop=drop_rate)
#             for _ in range(num_blocks)
#         ])

#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)

#         self.apply(self.init_weights)

#     def init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Conv2d):
#             nn.init.kaiming_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.ones_(module.weight)
#             nn.init.zeros_(module.bias)

#     def forward(self, images):
#         """
#         Forward pass du MLPMixer
#         :param images: Tensor de shape [batch, 3, img_size, img_size]
#         :return: Tensor de shape [batch, num_classes]
#         """
#         # Étape 1 : Patch Embedding
#         x = self.patchemb(images)  # [batch, num_patches, embed_dim]

#         # Étape 2 : Passer dans les Mixer Blocks
#         x = self.blocks(x)  # [batch, num_patches, embed_dim]

#         # Étape 3 : Normalisation
#         x = self.norm(x)  # [batch, num_patches, embed_dim]

#         # Étape 4 : Global Average Pooling sur les patches
#         x = x.mean(dim=1)  # [batch, embed_dim]

#         # Étape 5 : Classification
#         logits = self.head(x)  # [batch, num_classes]

#         return logits

#     def visualize(self, logdir):
#         """
#         Fonction à implémenter pour visualiser les couches du modèle.
#         Peut être utilisé avec TensorBoard ou Matplotlib.
#         """
#         raise NotImplementedError
