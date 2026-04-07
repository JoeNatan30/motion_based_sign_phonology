import os
import json
import random

import wandb
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import silhouette_score, pairwise_distances
from collections import defaultdict
from torch.utils.data import Sampler

from dataloaderTripletSemiHard import MetricLearningDataset, BatchTripletLoss, batch_variance_loss
from modelTriplet import ProjectorCNN


# ---------------------------------------------------
# UTILS
# ---------------------------------------------------
def set_all_seeds(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def include_fn(path):
    return os.path.basename(path) in [
        "model_works.py",
        "1_5_train_phonetic_semi_hard.py",
        "dataloaderTripletSemiHard.py",
        "utils.py",
    ]


class PKSampler(Sampler):
    """
    Cada batch tiene:
      - P clases
      - K muestras por clase
    batch_size = P * K
    """
    def __init__(self, labels, p_classes=8, k_samples=4, steps_per_epoch=200, seed=42):
        self.labels = list(labels)
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[y].append(idx)

        valid_classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= 2]
        if len(valid_classes) < self.p_classes:
            raise ValueError(
                f"No hay suficientes clases con al menos 2 muestras para PKSampler. "
                f"Disponibles={len(valid_classes)}, requeridas={self.p_classes}"
            )

        self.classes = valid_classes
        self.batch_size = self.p_classes * self.k_samples

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            chosen_classes = self.rng.sample(self.classes, self.p_classes)

            for c in chosen_classes:
                idxs = self.class_to_indices[c]

                if len(idxs) >= self.k_samples:
                    sampled = self.rng.sample(idxs, self.k_samples)
                else:
                    # con reemplazo si la clase tiene menos de K
                    sampled = [self.rng.choice(idxs) for _ in range(self.k_samples)]

                batch.extend(sampled)

            yield batch

    def __len__(self):
        return self.steps_per_epoch

# ---------------------------------------------------
# MÉTRICAS DE EVALUACIÓN DEL ESPACIO EMBEBIDO
# ---------------------------------------------------
def compute_class_centroids(embeddings, labels, normalize=True):
    classes = np.unique(labels)
    centroids = {}

    for c in classes:
        zc = embeddings[labels == c]
        centroid = zc.mean(axis=0)

        if normalize:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

        centroids[c] = centroid

    return centroids


def compute_intra_class_stats(embeddings, labels, centroids, metric="cosine"):
    per_class_rows = []
    intra_means = []

    for c in np.unique(labels):
        zc = embeddings[labels == c]
        rc = centroids[c][None, :]
        dists = pairwise_distances(zc, rc, metric=metric).ravel()

        row = {
            "class_id": int(c),
            "n_samples": int(len(zc)),
            "intra_mean": float(np.mean(dists)),
            "intra_std": float(np.std(dists)),
            "intra_p90": float(np.percentile(dists, 90)),
            "intra_p95": float(np.percentile(dists, 95)),
            "intra_max": float(np.max(dists)),
        }
        per_class_rows.append(row)
        intra_means.append(row["intra_mean"])

    return {
        "per_class": per_class_rows,
        "global_intra_mean": float(np.mean(intra_means)),
        "global_intra_std": float(np.std(intra_means)),
    }


def compute_inter_class_stats(centroids, metric="cosine"):
    classes = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)

    D = pairwise_distances(C, metric=metric)
    mask = ~np.eye(len(classes), dtype=bool)
    inter_vals = D[mask]

    return {
        "classes": classes,
        "distance_matrix": D,
        "inter_mean": float(np.mean(inter_vals)),
        "inter_std": float(np.std(inter_vals)),
        "inter_min": float(np.min(inter_vals)),
        "inter_p10": float(np.percentile(inter_vals, 10)),
    }


def compute_inter_intra_ratio(intra_stats, inter_stats, eps=1e-12):
    intra = intra_stats["global_intra_mean"]
    inter = inter_stats["inter_mean"]

    return {
        "intra_global_mean": float(intra),
        "inter_global_mean": float(inter),
        "inter_intra_ratio": float(inter / (intra + eps)),
        "inter_minus_intra": float(inter - intra),
    }


def nearest_centroid_accuracy(embeddings, labels, centroids, metric="cosine"):
    classes = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)

    D = pairwise_distances(embeddings, C, metric=metric)
    pred_idx = np.argmin(D, axis=1)
    pred_labels = np.array([classes[i] for i in pred_idx])

    acc = (pred_labels == labels).mean()
    return float(acc)


def centroid_margin_stats(embeddings, labels, centroids, metric="cosine"):
    classes = sorted(centroids.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    C = np.stack([centroids[c] for c in classes], axis=0)

    D = pairwise_distances(embeddings, C, metric=metric)

    margins = []
    for i in range(len(embeddings)):
        true_c = labels[i]
        true_idx = class_to_idx[true_c]

        d_pos = D[i, true_idx]

        d_neg = D[i].copy()
        d_neg[true_idx] = np.inf
        d_neg_min = np.min(d_neg)

        margins.append(d_neg_min - d_pos)

    margins = np.array(margins)

    return {
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "margin_p10": float(np.percentile(margins, 10)),
        "fraction_margin_gt_0": float(np.mean(margins > 0)),
    }


def nearest_classes_per_class(centroids, topk=5, metric="cosine"):
    classes = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)
    D = pairwise_distances(C, metric=metric)

    results = {}
    for i, c in enumerate(classes):
        order = np.argsort(D[i])
        order = [j for j in order if j != i][:topk]

        results[int(c)] = [
            {
                "neighbor_class": int(classes[j]),
                "distance": float(D[i, j])
            }
            for j in order
        ]
    return results


def evaluate_embedding_space(embeddings, labels, metric="cosine", topk_neighbors=5):
    centroids = compute_class_centroids(embeddings, labels, normalize=True)
    intra_stats = compute_intra_class_stats(embeddings, labels, centroids, metric=metric)
    inter_stats = compute_inter_class_stats(centroids, metric=metric)
    ratio_stats = compute_inter_intra_ratio(intra_stats, inter_stats)
    nc_acc = nearest_centroid_accuracy(embeddings, labels, centroids, metric=metric)
    margin_stats = centroid_margin_stats(embeddings, labels, centroids, metric=metric)
    neighbors = nearest_classes_per_class(centroids, topk=topk_neighbors, metric=metric)

    summary = {
        "intra_global_mean": ratio_stats["intra_global_mean"],
        "inter_global_mean": ratio_stats["inter_global_mean"],
        "inter_intra_ratio": ratio_stats["inter_intra_ratio"],
        "inter_minus_intra": ratio_stats["inter_minus_intra"],
        "nearest_centroid_acc": nc_acc,
        "margin_mean": margin_stats["margin_mean"],
        "margin_std": margin_stats["margin_std"],
        "margin_p10": margin_stats["margin_p10"],
        "fraction_margin_gt_0": margin_stats["fraction_margin_gt_0"],
    }

    return {
        "summary": summary,
        "centroids": centroids,
        "intra_stats": intra_stats,
        "inter_stats": inter_stats,
        "neighbors": neighbors,
    }


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

P_CLASSES = 16 #32 #16
K_SAMPLES = 8 #8 #4
EPOCHS = 10000
LR = 7e-5
MARGIN = 0.2
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1
PATIENCE = 20
MIN_DELTA = 1e-4
MINING_MODE = "semi-hard"
SEED = 42
ENTITY = "joenatan30"

EMBEDDING_FOLDER = "../../../../data/dataset_asl/dataset_videos/" #"../../../../data/dataset_lsp/dataset_videos/"#"../../../../data/ASL-Citizen/ASL_Citizen/videos_preprocessed" #"../../../../data/dataset_lsp/dataset_videos"
GEO_EMB_FOLDER = "geoEmb_iconic105USA" #"geoEmb_aslCitizen" #geoEmb_iconic105Peru

MHI_MODEL_DATASET = "Iconic105USA" #ASLcitizen #Iconic105Peru
TRIPLET_MODEL_DATASET = "Iconic105USA" #aslCitizen

MODEL_PATH = f"./MHIv9_MHI{MHI_MODEL_DATASET}_TripletSemiHard{TRIPLET_MODEL_DATASET}_GeoEmb{GEO_EMB_FOLDER.split('_')[-1]}_26032026.pt"
TOPK_NEIGHBORS = 5

TAG = [
    "triplet",
    "GeM_weighted",
    "32x128x128",
    f"TripletSemiHard{TRIPLET_MODEL_DATASET}_GeoEmb{GEO_EMB_FOLDER.split('_')[-1]}",
    "embedding-eval-centroids",
]

set_all_seeds(SEED)

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "triplet-learning-semi-hard"),
    entity=ENTITY,
    tags=TAG,
    config={
        "retrieve_model": False,
        "seed": SEED,
        "device": str(device),
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "margin": MARGIN,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "save_path": MODEL_PATH,
        "embedding_folder": EMBEDDING_FOLDER,
        "mining_mode": MINING_MODE,
        "p_classes": P_CLASSES,
        "k_samples": K_SAMPLES,
        "in_ch": 32,
        "channels": [64, 128, 256, 512],
        "emb_dim": 256,
        "proj_hidden": 512,
        "dropout": 0.15,
        "use_gem": True,
        "l2_norm": True,
    }
)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------
dataset = MetricLearningDataset(
    dataestType="training",
    base_folder=EMBEDDING_FOLDER,
    dataset=TRIPLET_MODEL_DATASET,
    geoEmb_folder=GEO_EMB_FOLDER
)

dataset_eval = MetricLearningDataset(
    dataestType="validation",
    base_folder=EMBEDDING_FOLDER,
    dataset=TRIPLET_MODEL_DATASET,
    geoEmb_folder=GEO_EMB_FOLDER
)

labels_for_sampler = dataset.df["label_id"].tolist()
steps_per_epoch = max(1, len(dataset) // (P_CLASSES * K_SAMPLES))

sampler = PKSampler(
    labels=labels_for_sampler,
    p_classes=P_CLASSES,
    k_samples=K_SAMPLES,
    steps_per_epoch=steps_per_epoch,
    seed=SEED
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=0,
    pin_memory=True
)

# loader para recorrer validation una sola vez completa
loader_eval = DataLoader(
    dataset_eval,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# loader adicional solo si quieres una val_loss comparable a train
labels_eval_for_sampler = dataset_eval.df["label_id"].tolist()
eval_steps = max(1, len(dataset_eval) // (P_CLASSES * K_SAMPLES))

sampler_eval_triplet = PKSampler(
    labels=labels_eval_for_sampler,
    p_classes=P_CLASSES,
    k_samples=K_SAMPLES,
    steps_per_epoch=eval_steps,
    seed=SEED + 1
)

loader_eval_triplet = DataLoader(
    dataset_eval,
    batch_sampler=sampler_eval_triplet,
    num_workers=0,
    pin_memory=True
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
model = ProjectorCNN(
    in_ch=32,
    channels=(64, 128, 256, 512),
    emb_dim=256,     
    proj_hidden=512,
    dropout=0.15,
    use_gem=True,
    l2_norm=True
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

criterion = BatchTripletLoss(
    margin=MARGIN,
    mode=MINING_MODE,
    reduction="mean"
)

run.log_code(root=".", include_fn=include_fn)

# ---------------------------------------------------
# INIT VARIABLES
# ---------------------------------------------------
lambda_var = 0.01 #Lambda del variational loss
start_epoch = 0
best_score = -float("inf")
best_silhouette = -float("inf")

if run.config["retrieve_model"]:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_score = checkpoint["best_score"]
    best_silhouette = checkpoint["best_silhouette"]
    summary = checkpoint["summary"]
    idx_to_class = checkpoint["idx_to_class"]

# ---------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()

    running_loss = 0.0
    running_dap = 0.0
    running_dan = 0.0
    running_triplets = 0.0
    running_semi_hard = 0.0
    running_hard_fallback = 0.0
    running_frac_pos_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        z = model(x)
        if z.ndim != 2:
            z = z.view(z.size(0), -1)
        z = F.normalize(z, p=2, dim=1)

        out = criterion(z, y)

        #ar_out = batch_variance_loss(z, y, normalize_centroids=False, reduction="mean")
        
        loss = out["loss"] #+ lambda_var * var_out["loss"]
        #loss = out["loss"]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dap += out["d_ap"]
        running_dan += out["d_an"]
        running_triplets += out["num_triplets"]
        running_semi_hard += out["num_semi_hard"]
        running_hard_fallback += out["num_hard_fallback"]
        running_frac_pos_loss += out["frac_positive_loss"]

    n_batches = len(loader)
    epoch_loss = running_loss / max(n_batches, 1)

    # ---------------------------------------------------
    # VALIDACIÓN 1: val_loss comparable a train (PK batches)
    # ---------------------------------------------------
    model.eval()
    val_triplet_loss = 0.0
    val_triplets = 0.0
    val_semi_hard = 0.0
    val_hard_fallback = 0.0
    val_frac_pos_loss = 0.0
    val_total_loss = 0.0
    val_var_loss = 0.0
    val_mean_class_var = 0.0

    with torch.no_grad():
        for x_val, y_val in loader_eval_triplet:
            x_val = x_val.to(device, non_blocking=True).float()
            y_val = y_val.to(device, non_blocking=True)

            z_val = model(x_val)
            if z_val.ndim != 2:
                z_val = z_val.view(z_val.size(0), -1)
            z_val = F.normalize(z_val, p=2, dim=1)

            out_val = criterion(z_val, y_val)
            triplet_loss_val = out_val["loss"]

            #var_out = batch_variance_loss(z_val, y_val, normalize_centroids=False, reduction="mean")
            #var_loss_val = var_out["loss"]

            total_loss_val = triplet_loss_val #+ lambda_var * var_loss_val

            val_triplet_loss += triplet_loss_val.item()
            #val_var_loss += var_loss_val.item()
            val_total_loss += total_loss_val.item()
            #val_mean_class_var += var_out["mean_class_var"]

            val_triplets += out_val["num_triplets"]
            val_semi_hard += out_val["num_semi_hard"]
            val_hard_fallback += out_val["num_hard_fallback"]
            val_frac_pos_loss += out_val["frac_positive_loss"]

    n_val_triplet_batches = len(loader_eval_triplet)
    val_triplet_loss /= max(n_val_triplet_batches, 1)
    #val_var_loss /= max(n_val_triplet_batches, 1)
    val_total_loss /= max(n_val_triplet_batches, 1)
    # val_mean_class_var /= max(n_val_triplet_batches, 1)

    val_triplets /= max(n_val_triplet_batches, 1)
    val_semi_hard /= max(n_val_triplet_batches, 1)
    val_hard_fallback /= max(n_val_triplet_batches, 1)
    val_frac_pos_loss /= max(n_val_triplet_batches, 1)

    # ---------------------------------------------------
    # VALIDACIÓN 2: evaluación global del espacio embebido
    # ---------------------------------------------------
    all_val_embeddings = []
    all_val_labels = []

    with torch.no_grad():
        for x_val, y_val in loader_eval:
            x_val = x_val.to(device, non_blocking=True).float()
            y_val = y_val.to(device, non_blocking=True)

            z_val = model(x_val)
            if z_val.ndim != 2:
                z_val = z_val.view(z_val.size(0), -1)
            z_val = F.normalize(z_val, p=2, dim=1)

            all_val_embeddings.append(z_val.cpu())
            all_val_labels.append(y_val.cpu())

    all_val_embeddings = torch.cat(all_val_embeddings, dim=0).numpy()
    all_val_labels = torch.cat(all_val_labels, dim=0).numpy()

    sil_score = silhouette_score(all_val_embeddings, all_val_labels, metric="cosine")

    eval_stats = evaluate_embedding_space(
        all_val_embeddings,
        all_val_labels,
        metric="cosine",
        topk_neighbors=TOPK_NEIGHBORS
    )
    summary = eval_stats["summary"]

    # score principal para seleccionar el mejor modelo
    current_score = (
        1.0 * summary["inter_intra_ratio"]
        + 0.5 * summary["margin_mean"]
        + 0.5 * summary["nearest_centroid_acc"]
    )

    print(
        f"Epoch {epoch+1:03d} | "
        f"train_loss={epoch_loss:.4f} | "
        # f"train_var_loss={val_toatl_loss:.4f} | "
        f"d_ap={running_dap / max(n_batches,1):.4f} | "
        f"d_an={running_dan / max(n_batches,1):.4f} | "
        f"triplets/batch={running_triplets / max(n_batches,1):.1f} | "
        f"semi-hard/batch={running_semi_hard / max(n_batches,1):.1f} | "
        f"hard-fallback/batch={running_hard_fallback / max(n_batches,1):.1f} | "
        f"frac_pos_loss={running_frac_pos_loss / max(n_batches,1):.4f} | "
        f"val_triplet_loss={val_triplet_loss:.8f} | "
        f"silhouette={sil_score:.4f} | "
        f"intra={summary['intra_global_mean']:.4f} | "
        f"inter={summary['inter_global_mean']:.4f} | "
        f"ratio={summary['inter_intra_ratio']:.4f} | "
        f"nc_acc={summary['nearest_centroid_acc']:.4f} | "
        f"margin={summary['margin_mean']:.4f} | "
        f"score={current_score:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,

        "train/loss": epoch_loss,
        "train/d_ap": running_dap / max(n_batches, 1),
        "train/d_an": running_dan / max(n_batches, 1),
        "train/triplets_per_batch": running_triplets / max(n_batches, 1),
        "train/semi_hard_per_batch": running_semi_hard / max(n_batches, 1),
        "train/hard_fallback_per_batch": running_hard_fallback / max(n_batches, 1),
        "train/frac_positive_loss": running_frac_pos_loss / max(n_batches, 1),

        "val/triplet_loss": val_triplet_loss,
        "val/triplets_per_batch": val_triplets,
        "val/semi_hard_per_batch": val_semi_hard,
        "val/hard_fallback_per_batch": val_hard_fallback,
        "val/frac_positive_loss": val_frac_pos_loss,

        "val/silhouette": sil_score,
        "val/intra_global_mean": summary["intra_global_mean"],
        "val/inter_global_mean": summary["inter_global_mean"],
        "val/inter_intra_ratio": summary["inter_intra_ratio"],
        "val/inter_minus_intra": summary["inter_minus_intra"],
        "val/nearest_centroid_acc": summary["nearest_centroid_acc"],
        "val/margin_mean": summary["margin_mean"],
        "val/margin_std": summary["margin_std"],
        "val/margin_p10": summary["margin_p10"],
        "val/fraction_margin_gt_0": summary["fraction_margin_gt_0"],

        "val/model_selection_score": current_score,
    })

    # guardar tabla intra-clase
    df_intra = pd.DataFrame(eval_stats["intra_stats"]["per_class"])
    df_intra["class_name"] = df_intra["class_id"].map(dataset_eval.idx_to_class)

    # opcional: subir la tabla cada cierto número de épocas
    if (epoch + 1) % 10 == 0:
        wandb.log({"val/intra_table": wandb.Table(dataframe=df_intra)})

    # guardar vecinos por clase cada cierto número de épocas
    if (epoch + 1) % 10 == 0:
        neighbors_named = {}
        for cid, neighs in eval_stats["neighbors"].items():
            class_name = dataset_eval.idx_to_class[cid]
            neighbors_named[class_name] = [
                {
                    "neighbor_class": dataset_eval.idx_to_class[n["neighbor_class"]],
                    "distance": n["distance"]
                }
                for n in neighs
            ]

        neighbors_json_path = "nearest_neighbors_val.json"
        with open(neighbors_json_path, "w", encoding="utf-8") as f:
            json.dump(neighbors_named, f, ensure_ascii=False, indent=2)

        artifact = wandb.Artifact(f"neighbors-epoch-{epoch+1}", type="analysis")
        artifact.add_file(neighbors_json_path)
        run.log_artifact(artifact)

    # mejor modelo por score compuesto
    if current_score > best_score:
        best_score = current_score
        best_silhouette = max(best_silhouette, sil_score)

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "best_silhouette": best_silhouette,
            "summary": summary,
            "idx_to_class": dataset_eval.idx_to_class,
        }, MODEL_PATH)

        wandb.save(MODEL_PATH, policy="now")
        print(f"  -> Mejor modelo guardado por score={best_score:.4f}")

wandb.summary["best_score"] = best_score
wandb.summary["best_silhouette"] = best_silhouette
wandb.finish()