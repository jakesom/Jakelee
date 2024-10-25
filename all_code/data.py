import torch
from torch import nn
import model2
import model1
import os
from timm.models.layers import trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MODEL(nn.Module):
    def __init__(self, num_classes, vit_pretrained_weights, mae_pretrained_weights, norm_layer, embed_dim,
                 use_attention=True, use_center_mask=True):
        super(MODEL, self).__init__()
        self.use_center_mask = use_center_mask
        self.use_attention = use_attention
        self.num_classes = num_classes
        self.vit_pretrained_weights = vit_pretrained_weights
        self.mae_pretrained_weights = mae_pretrained_weights
        self.fc_cat_norm = norm_layer(int(embed_dim + embed_dim / 2))
        self.fc_mae_norm = norm_layer(int(embed_dim / 2))
        self.attn = Attention(
            dim=int(embed_dim / 2), num_heads=16, qkv_bias=True, qk_scale=None, attn_drop=0.0,
            proj_drop=0.0)  # 参考timm中vit_large
        self.head = nn.Linear(int(embed_dim + embed_dim / 2),
                              self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.vit_model = model2.__dict__["vit_large_patch16"](num_classes=num_classes, drop_path_rate=0.1,
                                                                  global_pool=True)
        self.mae_model = model1.__dict__['mae_vit_large_patch16'](norm_pix_loss=False,
                                                                      use_center_mask=self.use_center_mask)

        if self.vit_pretrained_weights is not None:
            assert os.path.exists(self.vit_pretrained_weights), "weights file: '{}' not exist.".format(
                self.vit_pretrained_weights)
            vit_weights_dict = torch.load(self.vit_pretrained_weights, map_location='cpu')
            del_keys = ['head.weight', 'head.bias']
            for k in del_keys:
                del vit_weights_dict[k]
            print("Loading the vit pre-trained weights from {}!".format(self.vit_pretrained_weights))
            print(self.vit_model.load_state_dict(vit_weights_dict, strict=False))
        else:
            # print("No VIT weights found, starting from scratch!")
            pass

        if self.mae_pretrained_weights is not None:
            assert os.path.exists(self.mae_pretrained_weights), "weights file: '{}' not exist.".format(
                self.mae_pretrained_weights)
            mae_weights_dict = torch.load(self.mae_pretrained_weights, map_location='cpu')
            print("Loading the mae pre-trained weights from {}!".format(self.mae_pretrained_weights))
            print(self.mae_model.load_state_dict(mae_weights_dict['model'], strict=False))
        else:
            # print("No MAE weights found, starting from scratch!")
            pass

    def forward(self, normal_images, whole_images, mask_ratio):
        texture_features = self.vit_model(normal_images)
        pred, mae_loss = self.mae_model(whole_images, mask_ratio)
        if self.use_attention is True:
            pred = self.attn(pred)
            pred = self.fc_mae_norm(pred)
        pred = pred.mean(dim=1)
        features = torch.cat((pred, texture_features), dim=1)
        features = self.fc_cat_norm(features)
        x = self.head(features)
        return x, mae_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
