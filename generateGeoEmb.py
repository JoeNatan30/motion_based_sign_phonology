import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataloaderEmb import videoAndImageDataset
from modelMhi import Encoder2D1 #model import Encoder2D1

# =========================
# Configuración de dispositivo
# =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

runType = "train" #train #test #val #preview

config = {
    'num_channels': 4,           # 3 (frames) + 3 (motion)
    'embedding_size': 128,       # tamaño espacial del embedding
    'image_size': 1024,
    'sem_embed_dim': 256, 
    "retrieve_model": True,
    "z_dim": 64,
    "embedding_channels": 32,
    # Si quieres usar GPU, cambia por ["cuda:0","cuda:0","cuda:0","cuda:0"]
    "device_map": ["cpu","cpu","cpu","cpu"] #["cpu","cpu","cpu","cpu"]
}

video_folder_path = "../../../../data/dataset_lsp/dataset_videos/"#"../../../../data/dataset_lsp/dataset_seña_videos/" #'../../../../data/ASL-Citizen/ASL_Citizen/videos_preprocessed/' #"../../../../data/dataset_asl/dataset_asl_videos/"
mhi_folder_path   = "../../../../data/dataset_lsp/dataset_mhi/" #"../../../../data/ASL-Citizen/ASL_Citizen/mhi/" #"../../../../data/dataset_asl/dataset_asl_mhi/"

# =========================
# Modelo (Encoder)
# =========================
encoder = Encoder2D1(config, device_map=config["device_map"])  # el propio Encoder reparte dispositivos
# Carga de pesos
# state_dict = torch.load("./models/encoder/epoch_modelo con posEnc y GroupNorm_1324.pth", map_location="cpu", weights_only=True)#"./models/encoder/epoch_posEncGroupNorm_904.pth", map_location="cpu")

state_dict = torch.load("./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_Iconic105Peru_base.pth", map_location="cpu", weights_only=True)
# state_dict = torch.load("./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_ASLcitizen_base.pth", map_location="cpu", weights_only=True)#"./models/encoder/epoch_posEncGroupNorm_904.pth", map_location="cpu") #"./models/encoder/model-newMHI-mot3DFrame_ASL_16dim.pth"
#"./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_ASLcitizen_base_inference.pth" #"./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_Iconic105Peru_base.pth"
encoder.load_state_dict(state_dict, strict=False)
encoder.eval()  # importante para BN/IN

if runType=="train":
    dataset_train = videoAndImageDataset("training",  video_folder_path, mhi_folder_path, generate_mhi=True, addPath=True)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
elif runType=="val":
    dataset_val = videoAndImageDataset("validation", video_folder_path, mhi_folder_path, generate_mhi=True, addPath=True)
    val_loader  = DataLoader(dataset_val,   batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
elif runType=="test":
    dataset_test = videoAndImageDataset("test", video_folder_path, mhi_folder_path, generate_mhi=True, addPath=True)
    test_loader  = DataLoader(dataset_test,  batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
elif runType=="preview":
    dataset_preview = videoAndImageDataset("preview", video_folder_path, mhi_folder_path, generate_mhi=True, addPath=True)
    preview_loader  = DataLoader(dataset_preview,  batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
else:
    assert "Seleccion un runType correcto"

# =========================
# Utilidades
# =========================
def ensure_dir_for(path_np):
    os.makedirs(os.path.dirname(path_np), exist_ok=True)

@torch.no_grad()
def export_embeddings(loader, split_name: str, threshold: float = 0.05):
    """
    Recorre un loader y genera embeddings .npy.
    Asume que 'loader.dataset.current_video_path' se actualiza en __getitem__.
    - video esperado en [B, T, H, W, C] uint8
    - se normaliza a [0,1], se calcula motion y se concatena con frames: C=6
    """
    total = len(loader)
    for video, _, path in tqdm(loader, total=total, desc=f"Export {split_name}"):

        path = path[0]

        # Cambia 'videos' por 'geoEmb' y extensión a .npy
        out_path = path.replace("videos", "geoEmb_Iconic105Peru").replace("mp4", "npy")
        #"geoEmb_iconic105Peru" #"geoEmb_ASLCitizen"

        if os.path.exists(out_path):
            # ya existe
            print(f"Ya existe {out_path}, saltando...")
            continue
            

        # Transferencias asíncronas si tu DataLoader usa pin_memory=True
        video = video.float()       # [B, T, 3, 1024, 1024]
        print(f"Procesando {path} → {out_path}")

        video = video.permute(0, 1, 4, 2, 3).contiguous()   # [B,T,3,H,W]

        # print("video shape after permute:", video.shape)

        # --- Preprocesado: intenta hacerlo en FP16 para bajar huella ---
        #with torch.amp.autocast(dtype=torch.float32, device_type='cuda'):

        B, T, C, H, W = video.shape
        assert C == 3

        video = video / 255

        # ---- forward del encoder ----
        mhi, emb, _ = encoder(video)  # emb: [B, C_e, H_e, W_e]

        # ---- guardar ----
        ensure_dir_for(out_path)
        print(out_path)
        np.save(out_path, emb.cpu().numpy())


if runType=="train":
    export_embeddings(train_loader, split_name="train", threshold=0.05)
elif runType=="val":
    export_embeddings(val_loader, split_name="val", threshold=0.05)
elif runType=="test":
    export_embeddings(test_loader,  split_name="test", threshold=0.05)
elif runType=="preview":
    export_embeddings(preview_loader,  split_name="preview", threshold=0.05)
else:
    assert "Seleccion un runType correcto"