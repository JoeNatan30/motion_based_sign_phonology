
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

IN_CH = 32


def fix_shape_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 3 and x.shape[-1] == IN_CH:
        x = np.transpose(x, (2, 0, 1))
    assert x.ndim == 3 and x.shape[0] == IN_CH, f"Forma inesperada: {x.shape}"
    return x


class MetricLearningDataset(Dataset):
    def __init__(self, dataestType, base_folder, dataset="", geoEmb_folder="geoEmb_icnic105Peru"):
        if dataestType == "training":
            df = pd.read_csv(base_folder + "train_metadata_signerInd.csv")#"/../splits/train.csv")#"/train_metadata_signerInd.csv")
        elif dataestType == "validation":
            df = pd.read_csv(base_folder + "val_metadata_signerInd.csv")#"/../splits/val.csv")#"/val_metadata_signerInd.csv")
        elif dataestType == "testing":
            df = pd.read_csv(base_folder + "test_metadata_signerInd.csv")#"/../splits/test.csv")#"/test_metadata_signerInd.csv")
        else:
            raise ValueError(f"dataestType no válido: {dataestType}")

        if dataset == "aslCitizen":
            df["Gloss"] = df["Gloss"].astype(str).str.strip()
            df["label"] = df["Gloss"]
            df["npy_path"] = df["Video file"].apply(
                lambda s: os.path.join(
                    base_folder.replace("videos", geoEmb_folder).replace("_preprocessed", ""),
                    s.replace(".mp4", ".npy")
                )
            )
        else:
            df["category"] = df["category"].astype(str).str.strip()
            df["label"] = df["category"]
            df["npy_path"] = df["file_path"].apply(
                lambda s: os.path.join(
                    base_folder.replace("videos", geoEmb_folder),#.replace("_processed", "_aslCitizen"),
                    s.replace(".mp4", ".npy")
                )
            )

        print(df["npy_path"].head())
        df = df[df["npy_path"].apply(os.path.exists)].reset_index(drop=True)

        df = df.groupby("label").filter(lambda d: len(d) >= 2).reset_index(drop=True)

        if df.empty or df["label"].nunique() < 2:
            raise ValueError("Se requieren >=2 clases con >=2 muestras c/u.")

        classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        df["label_id"] = df["label"].map(self.class_to_idx)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["npy_path"], allow_pickle=False, mmap_mode="r")
        x = torch.tensor(fix_shape_np(x), dtype=torch.float32)
        y = int(row["label_id"])
        return x, y


def pairwise_cosine_distance(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.t()
    dist = 1.0 - sim
    return dist.clamp_min(0.0)


class BatchTripletLoss(nn.Module):
    """
    mode:
      - "hard": positive más lejano + negative más cercano
      - "semi-hard": positive más lejano + negative semi-hard si existe,
                     si no existe usa hard negative
    """
    def __init__(self, margin=0.2, mode="semi-hard", reduction="mean"):
        super().__init__()
        assert mode in ("hard", "semi-hard")
        assert reduction in ("mean", "sum", "none")
        self.margin = margin
        self.mode = mode
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        device = embeddings.device
        labels = labels.view(-1)
        dist = pairwise_cosine_distance(embeddings)   # [B, B]
        B = dist.size(0)

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=device)

        positive_mask = label_eq & (~eye)
        negative_mask = ~label_eq

        losses = []
        d_ap_list = []
        d_an_list = []

        valid_triplets = 0
        semi_hard_count = 0
        hard_fallback_count = 0
        positive_loss_count = 0

        for i in range(B):
            pos_idx = torch.where(positive_mask[i])[0]
            neg_idx = torch.where(negative_mask[i])[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # hard positive: el positivo más lejano
            pos_dists = dist[i, pos_idx]
            d_ap, ap_j = torch.max(pos_dists, dim=0)

            neg_dists = dist[i, neg_idx]

            if self.mode == "hard":
                d_an, an_j = torch.min(neg_dists, dim=0)
                hard_fallback_count += 1
            else:
                # semi-hard: d_ap < d_an < d_ap + margin
                semi_mask = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)

                if semi_mask.any():
                    semi_dists = neg_dists[semi_mask]
                    d_an, an_j = torch.min(semi_dists, dim=0)
                    semi_hard_count += 1
                else:
                    d_an, an_j = torch.min(neg_dists, dim=0)
                    hard_fallback_count += 1

            loss_i = F.relu(d_ap - d_an + self.margin)

            losses.append(loss_i)
            d_ap_list.append(d_ap.detach())
            d_an_list.append(d_an.detach())
            valid_triplets += 1

            if loss_i.detach().item() > 0:
                positive_loss_count += 1

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss": zero,
                "d_ap": 0.0,
                "d_an": 0.0,
                "num_triplets": 0,
                "num_semi_hard": 0,
                "num_hard_fallback": 0,
                "num_positive_loss": 0,
                "frac_positive_loss": 0.0,
            }

        losses = torch.stack(losses)

        if self.reduction == "mean":
            loss = losses.mean()
        elif self.reduction == "sum":
            loss = losses.sum()
        else:
            loss = losses

        d_ap_mean = torch.stack(d_ap_list).mean().item()
        d_an_mean = torch.stack(d_an_list).mean().item()

        return {
            "loss": loss,
            "d_ap": d_ap_mean,
            "d_an": d_an_mean,
            "num_triplets": valid_triplets,
            "num_semi_hard": semi_hard_count,
            "num_hard_fallback": hard_fallback_count,
            "num_positive_loss": positive_loss_count,
            "frac_positive_loss": positive_loss_count / max(valid_triplets, 1),
        }
    
def batch_variance_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalize_centroids: bool = False,
    reduction: str = "mean",
):
    """
    Penaliza la dispersión intra-clase dentro del batch.

    embeddings: [B, D]
    labels:     [B]
    normalize_centroids:
        si True, renormaliza cada centroide a norma 1.
        útil si trabajas estrictamente en espacio coseno.
    reduction:
        "mean" o "sum"
    """
    assert embeddings.ndim == 2, f"embeddings debe ser [B,D], llegó {embeddings.shape}"
    labels = labels.view(-1)

    unique_labels = torch.unique(labels)
    class_losses = []
    per_class_var = []

    for c in unique_labels:
        mask = labels == c
        zc = embeddings[mask]   # [K, D]

        if zc.size(0) < 2:
            continue

        centroid = zc.mean(dim=0, keepdim=True)  # [1, D]

        if normalize_centroids:
            centroid = F.normalize(centroid, p=2, dim=1)

        # Distancia cuadrática al centroide
        d2 = ((zc - centroid) ** 2).sum(dim=1)   # [K]
        loss_c = d2.mean()

        class_losses.append(loss_c)
        per_class_var.append(loss_c.detach())

    if len(class_losses) == 0:
        zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return {
            "loss": zero,
            "num_classes_used": 0,
            "mean_class_var": 0.0,
        }

    class_losses = torch.stack(class_losses)

    if reduction == "mean":
        loss = class_losses.mean()
    elif reduction == "sum":
        loss = class_losses.sum()
    else:
        raise ValueError(f"reduction no soportado: {reduction}")

    mean_class_var = torch.stack(per_class_var).mean().item()

    return {
        "loss": loss,
        "num_classes_used": len(class_losses),
        "mean_class_var": mean_class_var,
    }