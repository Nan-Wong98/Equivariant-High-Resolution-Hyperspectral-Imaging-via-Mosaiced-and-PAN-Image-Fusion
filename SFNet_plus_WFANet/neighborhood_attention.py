import torch
from torch import nn
from torch.nn import functional as F

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class NeighborhoodAttention(nn.Module): #It can only use static size as input,but you can define a new input size if you wish.
    def __init__(self, dim, num_heads,window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert window_size%2 == 1,'windowsize must be odd.'
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size//2)
        self.relative_bias = nn.Parameter(torch.zeros((2*self.window_size-1)**2,num_heads))
        nn.init.trunc_normal_(self.relative_bias, std=.02)
        self.idx_h = torch.arange(0,window_size)
        self.idx_w = torch.arange(0,window_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w).view(-1)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def attention(self,x):
        B,C,H,W = x.shape
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        attn_idx, bias_idx = self.set_input_size((H, W))
        qkv = self.qkv(x).view(B, 3,self.num_heads,C//self.num_heads,H*W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q.unsqueeze(3) @ k[:,:, attn_idx].transpose(-1,-2) #B,nh,L,1,K^2
        attn = attn + self.relative_bias[bias_idx].permute(2, 0, 1).unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v[:,:, attn_idx]).squeeze(3).transpose(-1,-2).contiguous().view(B,C,H,W)
        return x
        
    def get_bias_idx(self,H,W):
        num_repeat_h = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_h[self.window_size//2] = H-(self.window_size-1)
        num_repeat_w[self.window_size//2] = W-(self.window_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        return bias_idx.view(-1,self.window_size**2)
    
    def get_attn_idx(self,H,W):
        H_ = H - (self.window_size - 1)
        W_ = W - (self.window_size - 1)
        attn_idx = torch.arange(0,H_*W_,dtype=torch.float).view(1,1,H_,W_)
        attn_idx = self.pad_idx(attn_idx).view(-1).type(torch.long)
        attn_idx = self.get_unfold_idx(H,W)[attn_idx]
        return attn_idx
    
    def get_unfold_idx(self,H,W):
        H_ = H-(self.window_size-1)
        W_ = W-(self.window_size-1)
        h_idx = torch.arange(W_).repeat(H_)
        w_idx = torch.arange(H_).repeat_interleave(W_) * W
        k_idx_1 = torch.arange(self.window_size).repeat(self.window_size)
        k_idx_2 = torch.arange(self.window_size).repeat_interleave(self.window_size) * W
        k_idx = k_idx_1 + k_idx_2
        hw_idx = h_idx + w_idx
        unfold_idx = hw_idx[:,None] + k_idx
        return unfold_idx
    
    def set_input_size(self, input_size):
        H,W = input_size
        self.H, self.W = H,W
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        attn_idx = self.get_attn_idx(H,W)
        bias_idx = self.get_bias_idx(H,W)
        return attn_idx, bias_idx
        
class NATLayer(nn.Module):
    def __init__(self, dim, num_heads,window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, num_heads, window_size,qkv_bias, qk_scale, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
        
def test():
    print('it is cpu')
    model = NATLayer(128,4)
    img = torch.rand(2,128,56,56)
    try:
        print(model(img).shape)
    except:
        print('error')
        model.set_input_size((56,56))
        print(model(img).shape)
    print('cpu_success\n')

def test_cuda():
    print('it is cuda')
    model = NATLayer(128,4).cuda()
    img = torch.rand(2,128,56,56).cuda()
    try:
        print(model(img).shape)
    except:
        print('error')
        model.set_input_size((56,56))
        print(model(img).shape)
    print('success')
    print('cuda_success\n')
        
if __name__ == '__main__' :
    test()
    if torch.cuda.is_available():
        test_cuda()