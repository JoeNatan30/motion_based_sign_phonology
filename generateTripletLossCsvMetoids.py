# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import pairwise_distances

from dataloaderCsv import videoAndImageDataset
from modelTriplet import ProjectorCNN


# =========================================================
# Utils
# =========================================================
SEED = 42

def load_excluded_paths(
    outliers_csv_path: str,
    mode: str = "label",
    label_value: str = "strong_outlier",
    flag_col: str = "is_outlier_p95",
):
    """
    Lee un CSV de outliers y devuelve un set de PATHs a excluir.

    mode:
      - "label": usa columna 'outlier_label' == label_value
      - "flag": usa una columna booleana como flag_col == True
      - "all_flags": excluye si cualquiera de las columnas típicas de outlier es True
    """
    df_out = pd.read_csv(outliers_csv_path)

    if "PATH" not in df_out.columns:
        raise ValueError("El CSV de outliers debe tener columna 'PATH'.")

    if mode == "label":
        if "outlier_label" not in df_out.columns:
            raise ValueError("No existe la columna 'outlier_label' en el CSV de outliers.")
        mask = df_out["outlier_label"].astype(str).str.strip().str.lower() == label_value.strip().lower()

    elif mode == "flag":
        if flag_col not in df_out.columns:
            raise ValueError(f"No existe la columna '{flag_col}' en el CSV de outliers.")
        mask = df_out[flag_col].fillna(False).astype(bool)

    elif mode == "all_flags":
        candidate_cols = [
            "is_outlier_p80", "is_outlier_p90", "is_outlier_p95", "is_outlier_p99",
            "is_outlier_z20", "is_outlier_z25", "is_outlier_z30",
            "is_outlier_iqr15"
        ]
        existing = [c for c in candidate_cols if c in df_out.columns]
        if not existing:
            raise ValueError("No se encontraron columnas de flags de outlier en el CSV.")
        mask = df_out[existing].fillna(False).astype(bool).any(axis=1)

    else:
        raise ValueError("mode debe ser 'label', 'flag' o 'all_flags'.")

    excluded_paths = set(df_out.loc[mask, "PATH"].astype(str).tolist())
    return excluded_paths

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def standardize(imgs: torch.Tensor) -> torch.Tensor:
    """
    OJO:
    Usa esto solo si también lo usaste en entrenamiento del modelo triplet.
    Si NO lo usaste al entrenar, quítalo del loop.
    """
    imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1e4, neginf=-1e4)
    mu = imgs.mean(dim=(2, 3), keepdim=True)
    sd = imgs.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (imgs - mu) / sd


def read_word_map(words_csv_path: str) -> dict:

    required = {"EnglishGloss", "SpanishGloss"}
    df = pd.read_csv(words_csv_path, usecols=list(required))
    assert required.issubset(df.columns), "CSV debe tener 'EnglishGloss' y 'SpanishGloss'."
    assert not df.empty, "El CSV de palabras no debe estar vacío."

    df["SpanishGloss"] = df["SpanishGloss"].astype(str).str.upper().str.strip()
    df["EnglishGloss"] = df["EnglishGloss"].astype(str).str.upper().str.strip()

    return dict(zip(df["EnglishGloss"], df["SpanishGloss"])), dict(zip(df["SpanishGloss"], df["EnglishGloss"]))


# =========================================================
# Medoids
# =========================================================
def compute_class_medoids(df_embeddings, label_col="LSP_GLOSS", metric="cosine"):
    """
    Devuelve:
      - df_medoids: un row por clase, correspondiente a una muestra real (el medoid)
      - df_out: dataframe original con columna is_medoid
    """
    ignore_cols = {"PATH", "ASL_GLOSS", "LSP_GLOSS", "is_medoid", "medoid_rank", "class_size"}
    embedding_cols = [c for c in df_embeddings.columns if c not in ignore_cols]

    df_out = df_embeddings.copy()
    df_out["is_medoid"] = False
    df_out["medoid_rank"] = -1

    medoid_rows = []

    for class_name, group in df_out.groupby(label_col, sort=True):
        X = group[embedding_cols].to_numpy(dtype=np.float32)

        if len(group) == 1:
            medoid_local_idx = 0
        else:
            D = pairwise_distances(X, metric=metric)
            total_dist = D.sum(axis=1)
            medoid_local_idx = int(np.argmin(total_dist))

        group_idx = group.index.to_list()
        medoid_global_idx = group_idx[medoid_local_idx]

        df_out.loc[medoid_global_idx, "is_medoid"] = True
        df_out.loc[medoid_global_idx, "medoid_rank"] = 0

        medoid_row = df_out.loc[[medoid_global_idx]].copy()
        medoid_row["class_size"] = len(group)
        medoid_rows.append(medoid_row)

    df_medoids = pd.concat(medoid_rows, axis=0).reset_index(drop=True)
    return df_medoids, df_out


def per_class_dispersion_to_medoid(df_embeddings, label_col="LSP_GLOSS", metric="cosine"):
    """
    Calcula, por clase, la dispersión de las muestras hacia el medoid.
    """
    ignore_cols = {"PATH", "ASL_GLOSS", "LSP_GLOSS", "is_medoid", "medoid_rank", "class_size"}
    embedding_cols = [c for c in df_embeddings.columns if c not in ignore_cols]

    rows = []

    for class_name, group in df_embeddings.groupby(label_col, sort=True):
        X = group[embedding_cols].to_numpy(dtype=np.float32)

        if len(group) == 1:
            dists_to_medoid = np.array([0.0], dtype=np.float32)
        else:
            D = pairwise_distances(X, metric=metric)
            total_dist = D.sum(axis=1)
            medoid_idx = int(np.argmin(total_dist))
            dists_to_medoid = D[:, medoid_idx]

        rows.append({
            "class_name": class_name,
            "n": int(len(group)),
            "mean": float(np.mean(dists_to_medoid)),
            "std": float(np.std(dists_to_medoid)),
            "max": float(np.max(dists_to_medoid)),
        })

    df_disp = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df_disp


def nearest_classes_by_medoid(df_medoids, label_col="LSP_GLOSS", metric="cosine", topk=5):
    """
    Vecinos más cercanos entre clases usando el medoid de cada clase.
    """
    ignore_cols = {"PATH", "ASL_GLOSS", "LSP_GLOSS", "is_medoid", "medoid_rank", "class_size"}
    embedding_cols = [c for c in df_medoids.columns if c not in ignore_cols]

    X = df_medoids[embedding_cols].to_numpy(dtype=np.float32)
    labels = df_medoids[label_col].tolist()

    D = pairwise_distances(X, metric=metric)

    results = {}
    for i, label in enumerate(labels):
        order = np.argsort(D[i])
        order = [j for j in order if j != i][:topk]

        results[label] = [
            {"neighbor": labels[j], "distance": float(D[i, j])}
            for j in order
        ]
    return results


def print_dispersion_table(df_disp):
    print("\nPer-class dispersion to medoid\n")
    for _, row in df_disp.iterrows():
        print(
            f"{row['class_name']}\t"
            f"n={int(row['n'])}\t"
            f"mean={row['mean']:.6f}\t"
            f"std={row['std']:.6f}\t"
            f"max={row['max']:.6f}"
        )


def print_neighbors(neighbors_dict):
    print("\nNearest classes by medoid distance\n")
    for cls, neighs in neighbors_dict.items():
        print(cls)
        for item in neighs:
            print(f"   -> {item['neighbor']}   dist={item['distance']:.6f}")
        print()


# =========================================================
# Config
# =========================================================
seed_everything(SEED)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

SAVE_PATH = "MHIv9_27032026_MHIIconic105USA_TripletSemiHardIconic105USA_GeoEmbiconic105USA.pt" #"MHIv8_24032026_MHIaslCitizen_TripletSemiHardaslcitizen_GeoEmbASLCitizen_WithVarLoss.pt" #"MHIv8_06032026_iconic105Peru_paperClusters.pt" #f"./MHIv8_{GEO_EMB_FOLDER.split('_')[-1]}_bestMetric_corrections_06032026.pt"

config = {
    "triplet_model_path": SAVE_PATH, # "../../../../data/joe/models/triplet/triplet_MHIv5.pt",
    "words_csv": "./words.csv",
    "batch_size": 1,
    "num_workers": 0,
    "metric": "cosine", 
    "topk_neighbors": 5,

    "exclude_outliers": True,
    "outliers_csv_path": "./audit_outliers_medoid_cosine_all_outliers_p95.csv",   # cambia esta ruta
    "outlier_exclusion_mode": "label",               # "label", "flag", "all_flags"
    "outlier_label_value": "strong_outlier",         # si mode="label"
    "outlier_flag_col": "is_outlier_p95",            # si mode="flag"

}

GEO_EMB_FOLDER = "geoEmb_iconic105USA"#"geoEmb_ASLCitizen" #"geoEmb_iconic105Peru"

video_folder_path = "../../../../data/dataset_asl/dataset_videos/"
mhiAI_folder_path = f"../../../../data/dataset_asl/dataset_geoEmb_iconic105USA/" #"../../../../data/dataset_asl/datasetgeoEmb_iconic105USA/"
# iconic_folder_path = "../../../../data/WebScrapedData/Images/"

dataset_type = "all"   # "training", "validation", "test", "preview" "all"
language = "asl" #"aslCitizen" 
embName = "MHIv9"
tripletOn = "iconic105USA"
method = "medoids"  # "median"
text_language = "ingles"

base_name = f"phonoEmbedding_{method}_{language}_{embName}_{tripletOn}triplet_{dataset_type}"


# =========================================================
# DataLoader
# =========================================================
def worker_init_fn(wid):
    np.random.seed(SEED + wid)


mk_loader = lambda ds: DataLoader(
    ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    worker_init_fn=worker_init_fn,
)

if dataset_type == "all":
    dataset_train = videoAndImageDataset("training", video_folder_path, mhiAI_folder_path)
    dataset_val   = videoAndImageDataset("validation", video_folder_path, mhiAI_folder_path)
    dataset_test  = videoAndImageDataset("test", video_folder_path, mhiAI_folder_path)

    dataset = ConcatDataset([dataset_train, dataset_val, dataset_test])
else:
    dataset = videoAndImageDataset(dataset_type, video_folder_path, mhiAI_folder_path)

loader = mk_loader(dataset)

print("loaders size:", len(loader))


# =========================================================
# Modelo
# =========================================================
# IMPORTANTE:
# Debes reconstruir EXACTAMENTE la arquitectura usada al entrenar.
# Si tu modelo entrenado no es ProjectorCNN(use_gem=True), cambia esto.
phonologicModel = ProjectorCNN(
    in_ch=32,
    channels=(64, 128, 256, 512),
    emb_dim=256,
    proj_hidden=512,
    dropout=0.15,
    use_gem=True,
    l2_norm=True
).to(device)

ckpt = torch.load(config["triplet_model_path"], map_location=device)
phonologicModel.load_state_dict(ckpt["model_state_dict"], strict=True)
phonologicModel.eval()

with torch.no_grad():
    w = next(phonologicModel.parameters())
    print("weight_mean:", w.abs().mean().item())

print("ok - phonologic model instantiated.")


# =========================================================
# Mapping EN -> ES y ES -> EN
# =========================================================
en_to_es, es_to_en = read_word_map(config["words_csv"])


# =========================================================
# Extraer embeddings por muestra
# =========================================================
embs, labels_en, labels_es, paths_list = [], [], [], []

with torch.inference_mode():
    for embedding, label, path in tqdm(loader, desc=f"Embeddings ({dataset_type})"):
        embedding = embedding.to(device).float()

        # Si NO lo usaste en entrenamiento, comenta la siguiente línea
        #embedding = standardize(embedding)

        z = phonologicModel(embedding)

        if z.ndim != 2:
            z = z.view(z.size(0), -1)

        z = F.normalize(z, p=2, dim=1)

        z_np = z.detach().cpu().numpy()

        for i in range(z_np.shape[0]):
            embs.append(z_np[i])

        # Si batch_size=1, esto normalmente será una sola etiqueta/path
        if isinstance(path, (list, tuple)):
            batch_paths = list(path)
        else:
            batch_paths = [path]

        if isinstance(label, (list, tuple)):
            batch_labels = list(label)
        else:
            batch_labels = [label]

        for lbl, pth in zip(batch_labels, batch_paths):

            temp_name = str(lbl).strip().upper()
          
            if text_language == "ingles":
                label_es = en_to_es.get(temp_name, temp_name)
                labels_es.append(str(label_es).upper())
                labels_en.append(str(temp_name).upper())

            else:
                label_en = es_to_en.get(temp_name, temp_name)
                labels_en.append(str(label_en).upper())
                labels_es.append(str(temp_name).upper())  
                
            
            paths_list.append(str(pth))

print("Total embeddings extraídos:", len(embs))


# =========================================================
# Guardar CSV por muestra
# =========================================================
df = pd.DataFrame(embs)
df.insert(0, "PATH", paths_list)
df.insert(1, "ASL_GLOSS", labels_en)
df.insert(2, "LSP_GLOSS", labels_es)

if config.get("exclude_outliers", False):
    excluded_paths = load_excluded_paths(
        outliers_csv_path=config["outliers_csv_path"],
        mode=config.get("outlier_exclusion_mode", "label"),
        label_value=config.get("outlier_label_value", "strong_outlier"),
        flag_col=config.get("outlier_flag_col", "is_outlier_p95"),
    )

    n_before = len(df)
    df = df[~df["PATH"].astype(str).isin(excluded_paths)].reset_index(drop=True)
    n_after = len(df)

    print(f"Outliers excluidos: {n_before - n_after}")
    print(f"Muestras restantes: {n_after}")
# samples_csv_path = Path(f"{base_name}_samples.csv")
# df.to_csv(samples_csv_path, index=False)

# print(f"ok - CSV por muestra guardado en: {samples_csv_path}")
# print(f"Dimensiones del CSV por muestra: {df.shape}")
# print("Número de clases:", df["LSP_GLOSS"].nunique())


# =========================================================
# Calcular medoids por clase
# =========================================================
df_medoids, df_marked = compute_class_medoids(
    df,
    label_col="LSP_GLOSS",
    metric=config["metric"]
)

# medoids_csv_path = Path(f"{base_name}_class_medoids.csv")
# df_medoids.to_csv(medoids_csv_path, index=False)

# print(f"ok - CSV de medoids guardado en: {medoids_csv_path}")
print(f"Dimensiones del CSV de medoids: {df_medoids.shape}")

ignore_cols_for_final = {"PATH", "is_medoid", "medoid_rank", "class_size"}
final_cols = [c for c in df_medoids.columns if c not in ignore_cols_for_final]

df_medoids_final = df_medoids[final_cols].copy()

final_csv_path = Path(f"{base_name}.csv")
df_medoids_final.to_csv(final_csv_path, index=False)

print(f"ok - CSV final de representantes por clase guardado en: {final_csv_path}")
print(f"Dimensiones del CSV final: {df_medoids_final.shape}")

# # =========================================================
# # Dispersión al medoid
# # =========================================================
# df_disp = per_class_dispersion_to_medoid(
#     df,
#     label_col="LSP_GLOSS",
#     metric=config["metric"]
# )

# disp_csv_path = Path(f"{base_name}_dispersion_to_medoid.csv")
# df_disp.to_csv(disp_csv_path, index=False)

# print(f"ok - CSV de dispersión guardado en: {disp_csv_path}")
# print_dispersion_table(df_disp)


# =========================================================
# Vecinos más cercanos entre clases usando medoids
# =========================================================
neighbors = nearest_classes_by_medoid(
    df_medoids,
    label_col="LSP_GLOSS",
    metric=config["metric"],
    topk=config["topk_neighbors"]
)

neighbors_json_path = Path(f"{base_name}_nearest_neighbors.json")
with open(neighbors_json_path, "w", encoding="utf-8") as f:
    json.dump(neighbors, f, ensure_ascii=False, indent=2)

print(f"ok - JSON de vecinos guardado en: {neighbors_json_path}")
print_neighbors(neighbors)


# =========================================================
# Resumen final
# =========================================================
print("\nResumen final")
# print("Samples CSV:", samples_csv_path)
# print("Medoids CSV:", medoids_csv_path)
# print("Dispersion CSV:", disp_csv_path)
print("Neighbors JSON:", neighbors_json_path)