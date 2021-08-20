import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as models
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
    def __init__(self, dim, heads = 16, dim_head = 64, dropout = 0., config = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1) ## 여기를 바꾸면 됨

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.config = config
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale ## 32, 16, 80, 80

        attn = self.attend(dots) ## 32, 16, 80, 80
        # print('softmax:',attn[0,0,0,:])
        # print('softmax_sum:',torch.sum(attn[0,0,0,:]))

        # kdkd temperature 추가
        # if self.config.TEMP:
        #     # print('self.config.TEMP:',self.config.TEMP)
        #     temperature = self.config.TEMP
        #     attn = attn**(1/temperature)
        #     attn = attn / attn.sum(dim=-1, keepdim=True)
            # print('temp_softmax:',attn[0,0,0,:])
            # print('temp_softmax_sum:',torch.sum(attn[0,0,0,:]))
        # kdkd temperature 추가
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., config = None):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,config = config)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_Generator(nn.Module):
    def __init__(self, *, image_size_w,image_size_h, patch_size_w, patch_size_h, num_classes=1024,\
         dim=1024, depth=10, heads=2, mlp_dim=256, pool = 'cls', channels = 3, dim_head = 64, \
             dropout = 0., emb_dropout = 0.,gen_out_pixel, config = None):
             
        super().__init__()
        assert image_size_h % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size_w // patch_size_w)*(image_size_h // patch_size_h) ## 64
        patch_dim = channels * patch_size_w * patch_size_h
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        config.image_size = image_size_w
        self.config = config
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2 * gen_out_pixel*image_size_h//patch_size_w//patch_size_h, dim))
        self.cls_token_l = nn.Parameter(torch.randn(1,gen_out_pixel*image_size_h//patch_size_h//patch_size_w, patch_dim)) ## 양 사이드여서 2 * gen_out_pixel...
        self.cls_token_r = nn.Parameter(torch.randn(1,gen_out_pixel*image_size_h//patch_size_h//patch_size_w, patch_dim)) ## 양 사이드여서 2 * gen_out_pixel...
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,config=self.config)

        self.pool = pool
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, image_size),
        #     # Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2) ', p1 = patch_size_h, p2 = patch_size_w)
        # )
        if patch_size_h < image_size_h: ## columnwise 아니면
            if self.config.PATCH_ATTENTION:
                print('patch_attention')
                self.generate_img = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, channels * patch_size_w * patch_size_h), # (32,3,1024)--> 32 ,8*10, 3 * 16*16 (32,80,768)
                    Rearrange('b (h w) (p1 p2 c) -> b (w h) (p1 p2 c)', p1 = patch_size_h, \
                        p2 = patch_size_w, h = image_size_h//patch_size_h , w = image_size_w//patch_size_w+ 2*self.config.GEN_PIXEL_SIZE//patch_size_w ),
                    PATCH_Attention(config = self.config),
                    Rearrange('b (w h) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size_h, \
                        p2 = patch_size_w, h = image_size_h//patch_size_h , w = image_size_w//patch_size_w+ 2*self.config.GEN_PIXEL_SIZE//patch_size_w )
                    # Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size_h, \
                    #     p2 = patch_size_w, h = image_size_h//patch_size_h , w = image_size_w//patch_size_w+ 2*self.config.GEN_PIXEL_SIZE//patch_size_w )                    
                    ) # 32 ,8*10, 3 * 16*16 (32,80,768) --> 32, 3, 16*8, 16*10 (32,3,128,160) == (32,3,128,16 + 128 + 16)
                    # b (w h(p1 p2 c)
            else:
                self.generate_img = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, channels * patch_size_w * patch_size_h), ## 128 , 96 ,1024 --> 128, 96 , 3*16*16 = << 128 , 96 , 768 >>
                    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size_h, \
                        p2 = patch_size_w, h = image_size_h//patch_size_h , w = image_size_w//patch_size_w+ 2*self.config.GEN_PIXEL_SIZE//patch_size_w )
                        # 32 ,8*10, 3 * 16*16 --> 32, 3, 16*8, 16*10
                    
                )
        else:
            
            self.generate_img = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, channels * patch_size_w * patch_size_h), 
                Rearrange('b w (p1 p2 c) -> b c p1 (w p2)', p1 = patch_size_h, p2 = patch_size_w) ## (1 , 10 , 3*16*16) --> (1, 3, 128, 16+128+16)
                # nn.Linear(patch_size_w*(num_patches+2),image_size_w)
            )
        
        self.linear = nn.Linear(patch_dim,dim)

    def forward(self, img): ##img: 
        
        ## initialization_with_copied input_image
        # padded_img = torch.cat((img[:,:,:,:self.config.GEN_PIXEL_SIZE],img),dim=3)

        ##cls_token randn으로 생성할때 (original)
        x = self.to_patch_embedding(img) ## .shape = 128,64,1024 ## patch개수 : (128/16)의 제곱 == 64 ## paris : 14^2
        ## cls_token randn으로 생성할때 (original)

        # x = self.to_patch_embedding(padded_img)
        b, n, d = x.shape
        
        ##cls_token randn으로 생성할때 (original)
        cls_tokens_l = repeat(self.cls_token_l, '() n d -> b n d', b = b) ## 32,8,768  ## 8 pixel  일 때, 32, 4, 768
        cls_tokens_l = self.linear(cls_tokens_l) ## 128,16,1024 (16개 patch)
        cls_tokens_r = repeat(self.cls_token_r, '() n d -> b n d', b = b) 
        cls_tokens_r = self.linear(cls_tokens_r) 

        x = torch.cat((cls_tokens_l, x, cls_tokens_r), dim=1) ## 8 + 98 + 8 = 114 ## 8 pixel 일 때, 4 + 98 + 4
        ##cls_token randn으로 생성할때 (original)

        x += self.pos_embedding ##kdkd cls token n + 1 --> n + 2m
        x = self.dropout(x)

        x = self.transformer(x)
        
        # x = self.mlp_head(x)
        # print(x.shape)
        x = self.generate_img(x) ## 128 96 1024 ## 
        # print(x.shape)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x

class Resnet_Generator(nn.Module):
    def __init__(self, image_size = 128, config = None):
        super(Resnet_Generator, self).__init__()
        self.config = config
        self.resnet = models.resnet50()
        self.resnet.fc = nn.Linear(2048,80*768)
        self.generate_img = nn.Sequential(Rearrange('b (n d) -> b n d', n = 80),
        Rearrange('b (w h) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 16, p2 = 16, w =10)
        )

    def forward(self, img): ##img: 

        x = self.resnet(img) ## .shape = 128,64,1024 ## patch개수 : (128/16)의 제곱 == 64 ## paris : 14^2
        x = self.generate_img(x)
        
        return x
# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,config):
#         super().__init__()
#         config.image_size = image_size
#         self.config = config
#         assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = channels * patch_size ** 2
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#             nn.Linear(patch_dim, dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
#         return self.mlp_head(x)

class PATCH_Attention(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.attend = nn.Softmax(dim = -1) ## 여기를 바꾸면 됨
        self.config = config
        self.side = self.config.image_size//self.config.GEN_PIXEL_SIZE ## 8
    def forward(self, x):
        l,c,r = x[:,    :self.side    ,:], x[:,   self.side:-self.side   ,:], x[:,  -self.side: ,:] ## 32 80 768
        side = torch.cat((l,r),dim=1)
        side_norm = torch.nn.functional.normalize(side, p=2.0, dim=-1, eps=1e-12, out=None)
        c_norm = torch.nn.functional.normalize(c, p=2.0, dim=-1, eps=1e-12, out=None)
        dots = einsum('b i d, b j d -> b i j', side_norm, c_norm)  ## 32 16 768 , 32 64 768
        attn = self.attend(dots) ## 32, 16, 64
        # print(torch.sum(attn,dim=-1))
        zeros = torch.zeros_like(attn)
        for b in range(len(attn)):
            for p in range(len(attn[b])):
                max_index = torch.argmax(attn[b][p],dim=-1)
                zeros[b][p][max_index] = 1
        attn = zeros
        
        # kdkd temperature 추가
        if self.config.TEMP:
            print('self.config.TEMP:',self.config.TEMP)
            temperature = self.config.TEMP
            attn = attn**(1/temperature)
            attn = attn / attn.sum(dim=-1, keepdim=True)
        # kdkd temperature 추가

        out = einsum('b i j, b j d -> b i d', attn, c)
        out = torch.cat((out[:,:self.side,:],c,out[:,-self.side:,:]),dim=1)
        return out