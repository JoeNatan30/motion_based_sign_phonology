import os
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import numpy as np

from modelMhi import MappingHead, init_weights, Encoder2D1
from dataloaderEmb import videoAndImageDataset

torch.backends.cudnn.benchmark = True
cudnn.benchmark = False

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

config = {
    'num_channels': 4,
    'embedding_channels': 32,  # Dimensión del embedding en canales
    'image_size': 1024,         # Resolución de los videos de entrada (1024x1024)'
    "retrieve_model": False,
    "device_map":["cuda:4","cuda:4","cuda:4","cuda:4"],
    'sem_embed_dim': 256, 
    "patience": 30,
    "dataset": "Iconic105", #ASLcitizen #Iconic105
    "dataset_language": "Peru", #USA #Peru
}

folderName = "ASL-Citizen" if config["dataset"] == "ASLcitizen" else "Iconic105"
if folderName == "Iconic105" and config["dataset_language"] == "Peru":
    folderName = "dataset_lsp"
    subfolderName = "dataset_seña_videos"
elif folderName == "Iconic105" and config["dataset_language"] == "USA":
    folderName = "dataset_asl"
    subfolderName = "dataset_videos"

description = f"modelWithMapping_lumaWithVariation_smoothL1_{config['dataset']}_base" 

wandb_config = dict(
    lambda_embedding=1.0,
    lr=1e-5,
    weight_decay=1e-7,
    scheduler_step_size=1000,
    scheduler_gamma=0.5,
    num_epochs=5000,
    batch_size=1,
    threshold=0.05,
    device=str(device),
    desc=description,
    **config,  # mezcla tu config (num_channels, embedding_channels, etc.)
)

ENTITY    = "joenatan30"
run = wandb.init(project="mhi-AI", entity=ENTITY, name=f"run-{description}", config=wandb_config)

def include_fn(path):
    return os.path.basename(path) in ["model_new.py", "1_trainingVideoEnbed_new.py", "dataloader_emb.py", "utils.py"]

run.log_code(root=".", include_fn=include_fn)

video_folder_path = f"../../../../data/{folderName}/{subfolderName}/"
image_folder_path = f"../../../../data/{folderName}/{subfolderName.replace('videos','mhi')}/"


dataset_train = videoAndImageDataset("training", video_folder_path, image_folder_path, dataset=config["dataset"])
dataset_val = videoAndImageDataset("validation", video_folder_path, image_folder_path, dataset=config["dataset"])
print(len(dataset_train))
train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=3)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# Instanciar módulos del modelo
encoder = Encoder2D1(config, device_map=config["device_map"]).to(device)
encoder.apply(init_weights)
mapping_head = MappingHead(in_channels=config['embedding_channels'], out_channels=1).to(device).to(memory_format=torch.channels_last)
finalEpoch = 0
if config["retrieve_model"]:

    finalEpoch = 10
    epoch = 10
    print(f"Recuperando modelo de epoch {finalEpoch}")
    # Cargar pesos del encoder y mapping head
    encoder.load_state_dict(torch.load(f"./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_{config['dataset']}_base.pth")) #f"./models/encoder/model-newMHI-mot3DFrame_ASL_16dim.pth"
    mapping_head.load_state_dict(torch.load(f"./models/mapping_head/mappingHead_lumaWithVariation_smoothL1_{config['dataset']}_base_last.pth")) #f"./models/mapping_head/mapping_head_epoch_{finalEpoch}.pth"

print(encoder)
print(mapping_head)

# Optimizer para todos los parámetros entrenables
optimizer = optim.Adam(
    list(encoder.parameters()) + list(mapping_head.parameters()),
    lr=1e-5,
    weight_decay=1e-7
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
num_epochs = 5000

scaler = torch.amp.GradScaler(enabled=True)

log_every = 500
wandb.watch(models=encoder, log="all", log_freq=log_every)
wandb.watch(models=mapping_head, log="all", log_freq=log_every)

global_step = 0
min_val_loss = float('inf')

for epoch in range(finalEpoch, num_epochs):
    encoder.train()
    mapping_head.train()
    
    total_epoch_loss = 0.0

    for video, mhi_ref in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad(set_to_none=True)

        # Transferencias asíncronas si tu DataLoader usa pin_memory=True
        video = video.to(device, non_blocking=True).float()       # [B, T, 3, 1024, 1024]
        mhi_ref = mhi_ref.to(device, non_blocking=True).float()   # [B, 3, 1024, 1024]

        video = video.permute(0, 1, 4, 2, 3).contiguous()   # [B,T,3,H,W]
        mhi_ref = mhi_ref.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        # print("video shape after permute:", video.shape)

        # --- Preprocesado: intenta hacerlo en FP16 para bajar huella ---
        #with torch.amp.autocast(dtype=torch.float32, device_type='cuda'):

        B, T, C, H, W = video.shape
        assert C == 3

        video = video / 255
        mhi_ref01 = mhi_ref / 255

        # print("video_in shape:", video_in.shape)
        mhi_coarse, embedding, _  = encoder(video)
        mhi_pred = mapping_head(embedding)

        # Prepara MHI objetivo también reducido una sola vez
        h2, w2 = mhi_pred.shape[-2], mhi_pred.shape[-1]
        target_small = F.interpolate(mhi_ref01, size=(h2, w2), mode='bilinear', align_corners=False)
        r, g, b = target_small[:, 0:1], target_small[:, 1:2], target_small[:, 2:3]
        target_small = 0.2126 * r + 0.7152 * g + 0.0722 * b

        loss = F.smooth_l1_loss(mhi_pred, target_small)
        total_loss = wandb_config["lambda_embedding"] * loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_epoch_loss += float(total_loss.detach().item())

        if (global_step % log_every) == 0:
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "train/step_loss": float(total_loss.detach().item()),
                "train/lr": current_lr,
                "train/grad_scaler_scale": float(scaler.get_scale()),
                "meta/epoch": epoch,
                "meta/step": global_step
            }, step=global_step)

        global_step += 1

        # Libera intermediarios grandes explicitamente
        del video, mhi_ref, mhi_ref01, embedding, mhi_coarse, mhi_pred, target_small, total_loss, loss, r, g, b
        #torch.cuda.synchronize()  # opcional; evita acumulación de buffers en colas

    avg_epoch_loss = total_epoch_loss / len(train_loader)
    # avg_epoch_mse = total_epoch_mse / len(train_loader)
    # avg_epoch_perc = total_epoch_perc / len(train_loader)
    wandb.log({
        "train/epoch_loss": avg_epoch_loss,
        "meta/epoch": epoch
    }, step=global_step)
    scheduler.step()

    encoder.eval()
    mapping_head.eval()

    total_val_loss = 0.0
    last_pred = None
    last_tgt  = None


    #with torch.no_grad(), torch.amp.autocast(dtype=torch.float32, device_type='cuda'):
    with torch.no_grad():
        for video, mhi_ref in tqdm(val_loader, total=len(val_loader)):
            video = video.to(device, non_blocking=True).float()
            mhi_ref = mhi_ref.to(device, non_blocking=True).float()

            video = video.permute(0, 1, 4, 2, 3).contiguous()   # [B,T,3,H,W]
            mhi_ref = mhi_ref.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]

            video = video / 255
            mhi_ref01 = mhi_ref / 255

            mhi_coarse, embedding, _ = encoder(video)
            mhi_pred = mapping_head(embedding)  # [B,1,h2,w2]

            h2, w2 = mhi_pred.shape[-2], mhi_pred.shape[-1]
            target_small = F.interpolate(mhi_ref01, size=(h2, w2), mode='bilinear', align_corners=False)
            r, g, b = target_small[:, 0:1], target_small[:, 1:2], target_small[:, 2:3]
            target_small = 0.2126 * r + 0.7152 * g + 0.0722 * b

            loss = F.smooth_l1_loss(mhi_pred, target_small)
            val_loss = wandb_config["lambda_embedding"] * loss
            total_val_loss += float(val_loss.detach().item())

            last_pred = mhi_pred
            last_tgt  = target_small

            del video, mhi_ref, mhi_ref01, embedding, mhi_coarse, mhi_pred, target_small, loss, val_loss, r, g, b

    if last_pred is not None and last_tgt is not None:
        pred_img = (last_pred[0, 0].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        tgt_img  = (last_tgt[0, 0].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

        img_pred = Image.fromarray(pred_img, mode='L')
        img_tgt  = Image.fromarray(tgt_img,  mode='L')

        wandb.log({
            "val/pred_vs_target": [
                wandb.Image(img_pred, caption="pred"),
                wandb.Image(img_tgt,  caption="target")
            ]
        }, step=global_step)

    # Guardado (si quieres versionar por epoch, agrega _{epoch})
    encoder_path = f"./models/encoder/modelWithMapping_lumaWithVariation_smoothL1_{config['dataset']}_base.pth"
    head_path    = f"./models/mapping_head/mappingHead_lumaWithVariation_smoothL1_{config['dataset']}_base_last.pth"

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(mapping_head.state_dict(), head_path)

    wandb.save(encoder_path, policy="now")
    wandb.save(head_path, policy="now")


    avg_val_loss = total_val_loss / max(1, len(val_loader))
    wandb.log({"val/loss": avg_val_loss, "meta/epoch": epoch}, step=global_step)
    
    print(
    f"Epoch [{epoch+1}/{num_epochs}] "
    f"- Train: {avg_epoch_loss:.6f} "
    f"- Val: {avg_val_loss:.6f} "
    f"- LR: {scheduler.get_last_lr()[0]:.2e}")

run.finish()