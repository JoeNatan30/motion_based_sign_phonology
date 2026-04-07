import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torchvision.models as models # type: ignore
import torchvision.transforms.functional as TF # type: ignore


def rgb_to_luma_bt709(x: torch.Tensor) -> torch.Tensor:
    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    b = x[:, :, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y

def sobel_mag(img: torch.Tensor) -> torch.Tensor:
    """
    img: [N,1,H,W]
    returns: [N,1,H,W] Sobel magnitude
    """
    device = img.device
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=img.dtype, device=device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=img.dtype, device=device).view(1,1,3,3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return gx*gx + gy*gy + 1e-8

def build_diff_luma_variations(x: torch.Tensor, blur_ks: int = 3, blur_sigma: float = 1.0, detach_diff: bool = False) -> torch.Tensor:
    """
    x: [B, T, 3, H, W] en [0,1]
    return: x_cat [B, 4, T-1, H, W] listo para Conv3D
            canales = [d_fine, y_norm(t), diff_luma, d_edge]
    """
    assert x.dim() == 5 and x.size(2) == 3, "x debe ser [B,T,3,H,W]"
    B, T, C, H, W = x.shape
    assert T >= 2, "Necesitas al menos 2 frames para diff"

    # 1) Blur opcional (suaviza antes de luma/diff)
    if blur_ks and blur_ks > 1:
        x_ = x.view(B * T, C, H, W)
        x_ = TF.gaussian_blur(x_, kernel_size=blur_ks, sigma=blur_sigma)
        x = x_.view(B, T, C, H, W)

    # 2) Luma
    y = rgb_to_luma_bt709(x)  # [B,T,1,H,W]  para imágenes HD

    # 3) Normalización por frame (quita brillo global)
    mu = y.mean(dim=(3, 4), keepdim=True)      # [B,T,1,1,1]
    y_norm = y - mu                            # [B,T,1,H,W]
    y_norm_t = y_norm[:, :-1]                  # [B,T-1,1,H,W]

    # 4) Diff fino 
    d_fine = torch.abs(y[:, 1:] - y[:, :-1])   # [B,T-1,1,H,W]
    #d_fine = torch.abs(y_norm[:, 1:] - y_norm[:, :-1])   # [B,T-1,1,H,W]
    if detach_diff:
        d_fine = d_fine.detach()

    # 5) Bordes del movimiento
    df_flat = d_fine.view(B * (T - 1), 1, H, W)
    d_edge = sobel_mag(df_flat).view(B, T - 1, 1, H, W)

    # 6) Concat en canales (dim=2)
    x_cat = torch.cat([d_fine, y_norm_t, d_fine, d_edge], dim=2) # [B,T-1,4,H,W]
    #TODO Probar que no es necesario y_norm_t, y duplicar d_fine
    #x_cat = torch.cat([d_fine, d_edge], dim=2) # [B,T-1,4,H,W]

    # 7) Formato Conv3D: [B, C, T, H, W]
    x_cat = x_cat.permute(0, 2, 1, 3, 4).contiguous()  # [B,4,T-1,H,W]
    return x_cat

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu') # init.xavier_uniform_(m.weight, gain=0.5) 
        if m.bias is not None:
            m.bias.data.zero_()

class TransformerTemporalAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.pe = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.score_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Paso 1: Pooling espacial por frame → [B, T, C]
        x_pooled = x.to(self.input_proj.weight.device)
        #TODO Probar con GeM pooling 
        x_pooled = F.adaptive_avg_pool2d(x_pooled.view(B * T, C, H, W), 1).view(B, T, C)  # [B, T, C]
        
        # Paso 2: Proyección y Transformer temporal
        x_embed = self.input_proj(x_pooled)  # [B, T, embed_dim]
        x_embed = x_embed + self.pe[:, :T, :] # Añadir Positional Encoding
        x_transformed = self.transformer_encoder(x_embed)  # [B, T, embed_dim]

        # Paso 3: Calcular pesos de atención
        scores = self.score_proj(x_transformed).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(scores, dim=1)  # [B, T]
        attn_weights = attn_weights.view(B, T, 1, 1, 1)
        
        return attn_weights

class Motion3DBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels,
                      kernel_size=(3, 7, 7),
                      stride=(1, 2, 2),          # T; H/W -> /2
                      padding=(1, 3, 3),
                      bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )  # -> [B, base, T, H/2, W/2]

        self.layer2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2,
                      kernel_size=(3, 3, 3),
                      stride=(1, 2, 2),          # T; H/W -> /4
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )  # -> [B, 2*base, T, H/4, W/4]

        self.layer3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4,
                      kernel_size=(3, 3, 3),
                      stride=(2, 1, 1),          #  T -> T/2; H/W /4
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )  # -> [B, 4*base, T/2, H/4, W/4]

        self.out_channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Is is 3DCNN with temporal attention
class Encoder2D1(nn.Module):
    def __init__(self, config, device_map=None):
        super().__init__()
        self.devices = device_map
        self.embedding_channels = config["embedding_channels"]
        base_channels = self.embedding_channels // 4

        self.backbone3d = Motion3DBackbone(
            in_channels=4,
            base_channels=base_channels
        ).to(self.devices[0])   # <- IMPORTANT

        assert self.backbone3d.out_channels == self.embedding_channels

        self.attn_conv = TransformerTemporalAttention(
            embed_dim=self.embedding_channels,
            num_heads=8
        ).to(self.devices[0])   # <- IMPORTANT

    def forward(self, x):

        # x = x.to(self.devices[0])

        B, T, C, H, W = x.shape
        assert C == 3, "check RGB channels in input, it should be 3"

        x_cat = build_diff_luma_variations(x, blur_ks=3, blur_sigma=1.0) 
        feat = self.backbone3d(x_cat)       
        
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        # print(feat.shape, "<<< feat shape in encoder2d1")
        attn_weights = self.attn_conv(feat)

        embedding = (feat * attn_weights).sum(dim=1)

        mhi = embedding.mean(dim=1, keepdim=True)
        # print(embedding.shape, "<<< embedding shape in encoder2d1")
        return mhi, embedding, attn_weights


# -------------------------
# 2. Mapping Head para proyectar el embedding a imagen (para compararlo con la MHI)
# -------------------------
class MappingHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, out_channels, kernel_size=1),

        )
    def forward(self, x):
        return self.net(x)  # logits


def init_weights_projector(m):
    import torch.nn as nn
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

@torch.no_grad()
def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # Mide diferencia promedio entre tensores (entrada cruda)
    return (a - b).abs().mean().item()

@torch.no_grad()
def collapse_probe(z_a, z_p, z_n, thr_cos=0.999):
    cos_ap = F.cosine_similarity(z_a, z_p, dim=1).mean().item()
    cos_an = F.cosine_similarity(z_a, z_n, dim=1).mean().item()
    return cos_ap > thr_cos and cos_an > thr_cos, cos_ap, cos_an


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], device=None):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- VGG preentrenado ---
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(self.device).eval()
        self.slices = nn.ModuleList()
        prev_idx = 0
        for l in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_idx, l)]).to(self.device)
            self.slices.append(block)
            prev_idx = l

        for param in self.parameters():
            param.requires_grad = False

        # --- Normalización Imagenet ---
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean.to(self.device))
        self.register_buffer('std', std.to(self.device))

    def forward(self, x, y):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1: y = y.repeat(1, 3, 1, 1)

        # Resize + normaliza
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False).to(self.device)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False).to(self.device)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        for slice in self.slices:
            x = slice(x)
            y = slice(y)
            loss += F.l1_loss(x, y)
        return loss
    
