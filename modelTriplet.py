import torch.nn as nn
import torch
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu') # init.xavier_uniform_(m.weight, gain=0.5) 
        if m.bias is not None:
            m.bias.data.zero_()

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


# -------------------------
# GeM Pooling
# -------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=1e-3)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x.flatten(1)  # (B, C)


# -------------------------
# Conv Block
# -------------------------
class ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_groups=8):
        super().__init__()

        num_groups = min(num_groups, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# Improved Projector CNN
# -------------------------
class ProjectorCNN(nn.Module):
    """
    Projector CNN multi-scale para entradas de forma:
        (B, 32, 128, 128)

    Mejoras respecto al original:
    - menos canales extremos
    - multi-scale pooling con GeM
    - gating softmax entre escalas
    - projection head más estable para triplet loss
    - salida normalizada L2
    """
    def __init__(
        self,
        in_ch=32,
        channels=(64, 128, 256, 512),
        emb_dim=256,
        proj_hidden=512,
        dropout=0.15,
        use_gem=True,
        l2_norm=True
    ):
        super().__init__()

        self.use_gem = use_gem
        self.l2_norm = l2_norm

        # Backbone CNN
        # Para input 128x128:
        # 128 -> 64 -> 32 -> 16 -> 16
        blocks = []
        prev = in_ch
        for i, ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            blocks.append(ConvGNReLU(prev, ch, stride=stride, num_groups=8))
            prev = ch
        self.blocks = nn.ModuleList(blocks)

        # Pooling por escala
        if use_gem:
            self.pool = GeM(p=3.0, learn_p=True)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        total_dim = sum(channels)

        # Gating entre escalas
        self.scale_logits = nn.Parameter(torch.zeros(len(channels)))

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(total_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, emb_dim, bias=False),
        )

        self.neck = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _pool_2d(self, x):
        if self.use_gem:
            return self.pool(x)  # (B, C)
        return self.pool(x).flatten(1)

    def forward(self, x, return_aux=False):
        """
        x: (B, 32, 128, 128)
        """
        feats = []
        out = x

        for block in self.blocks:
            out = block(out)
            pooled = self._pool_2d(out)   # (B, C_scale)
            feats.append(pooled)

        # Pesos de escalas
        scale_weights = torch.softmax(self.scale_logits, dim=0)

        weighted_feats = [f * scale_weights[i] for i, f in enumerate(feats)]
        multi_scale = torch.cat(weighted_feats, dim=1)

        emb = self.proj(multi_scale)
        emb = self.neck(emb)

        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=1)

        if return_aux:
            return {
                "embedding": emb,
                "scale_weights": scale_weights.detach()
            }

        return emb


