import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        ##kdkd
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim * 1, bias = False)
        ##kdkd
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        if type(x) != tuple:
            b, n, _, h = *x.shape, self.heads
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        else:
            x,x_input = x
            b, n, _, h = *x.shape, self.heads
            qk = self.to_qk(x).chunk(2, dim = -1)
            # v = self.to_v(x_input).chunk(1, dim = -1)
            v = self.to_v(x_input)
            q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qk)
            v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
 

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(dim)
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        count = 0
        x_input = x
        for attn, ff in self.layers:
            if count == self.depth - 1 : ## last depth
                x_input = self.LayerNorm(x_input)
                x = self.LayerNorm(x)
                x = attn((x,x_input)) ## +x
                x = ff(x) + x
            else:
                x = attn(x) + x
                x = ff(x) + x
            count += 1

            
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT_Generator(nn.Module):
    def __init__(self, *, image_size_w,image_size_h, patch_size_w, patch_size_h, num_classes=1024, dim=1024, depth=10, heads=2, mlp_dim=256, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,gen_out_pixel):
        super().__init__()
        assert image_size_h % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size_w // patch_size_w)*(image_size_h // patch_size_h)
        patch_dim = channels * patch_size_w * patch_size_h
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + gen_out_pixel*image_size_h//patch_size_w//patch_size_h, dim))
        self.cls_token = nn.Parameter(torch.randn(1,2*image_size_h//patch_size_h, patch_size_h*patch_size_w*3))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, image_size),
        #     # Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2) ', p1 = patch_size_h, p2 = patch_size_w)
        # )
        if patch_size_h < image_size_h: ## columnwise 아니면
            self.generate_img = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, channels * patch_size_w * patch_size_h), ## 이거 고치는중
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size_h, \
                    p2 = patch_size_w, h = image_size_h//patch_size_h , w = image_size_w//patch_size_w+ 2 ) ## (1 , 14 ,16*224*3) --> (1, 3, 224, 14 * 16)
                # nn.Linear(patch_size_w*(num_patches+2),image_size_w)
            )
        else:
            self.generate_img = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, channels * patch_size_w * patch_size_h), ## 이거 고치는중
                Rearrange('b w (p1 p2 c) -> b c p1 (w p2)', p1 = patch_size_h, p2 = patch_size_w), ## (1 , 14 ,16*224*3) --> (1, 3, 224, 14 * 16)
                
                ## kdkd 끝단에 Conv2d 추가
                nn.Conv2d(in_channels = 3,
                                      out_channels = 3,
                                      kernel_size = 3,
                                      stride = 1,
                                      padding = 1)
                # nn.Linear(patch_size_w*(num_patches+2),image_size_w)
            )
        
        self.linear = nn.Linear(patch_dim,dim)

    def forward(self, img):

        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens=self.linear(cls_tokens)
        # print(x.shape)
        # print(cls_tokens.shape)
        # print(self.pos_embedding[:, :(n+2)].shape)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embedding = self.pos_embedding
        x += self.pos_embedding##kdkd cls token 없애서 n+1 --> n 으로 바꿈
        x = self.dropout(x)

        x = self.transformer(x)
        
        # x = self.mlp_head(x)
        # print(x.shape)
        x = self.generate_img(x)
        # print(x.shape)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x
