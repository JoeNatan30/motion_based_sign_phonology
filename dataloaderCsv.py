import os

import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from utils import load_image


def deleteItemsWithoutIconic(video_paths, iconic_paths):
    for i in range(len(video_paths)-1, -1, -1):
        # si no se puede cargar a imagen de iconic path
        try:
            iconic = load_image(iconic_paths[i])
        except:
            iconic = None
        if iconic is None:
            # eliminar video path y iconic path
            del video_paths[i]
            del iconic_paths[i]
    return video_paths, iconic_paths


class videoAndImageDataset(torch.utils.data.Dataset):

    def __init__(self, datasetType="training", videoFolderPath="", mhiAiFolderPath=""):
        super(videoAndImageDataset, self).__init__()

        self.datasetType = datasetType

        # imagesMetadata = get_video_paths_from_csv(iconicFolderPath+"cropped_images.csv")

        # videoMetadata = pd.read_csv(videoFolderPath + "train_metadata.csv")
        # df_common = pd.merge(imagesMetadata, videoMetadata, on='category', how='inner')
        # df_common["file_name_x"] = df_common["file_name_x"].apply(lambda x: iconicFolderPath + x)

        # df_common["file_path"] = df_common["file_path"].apply(lambda x: videoFolderPath + x)
        # df_common["mhiAi_path"] = df_common["file_path"].apply(lambda x: x.replace(videoFolderPath, mhiAiFolderPath).replace("mp4", "npy"))
        # df_common["mhiAi_exists"] = df_common["mhiAi_path"].apply(os.path.exists)
        # df_common = df_common[df_common["mhiAi_exists"] == True]

        if datasetType == "training":
            videoMetadata = pd.read_csv(videoFolderPath + "train_metadata_signerInd.csv")
            mhiAiFolderPath = mhiAiFolderPath
        elif datasetType == "validation":
            videoMetadata = pd.read_csv(videoFolderPath + "val_metadata_signerInd.csv")
            mhiAiFolderPath = mhiAiFolderPath
        elif datasetType == "preview":
            videoMetadata = pd.read_csv(videoFolderPath + "metadata.csv")
            mhiAiFolderPath = mhiAiFolderPath
        else:
            videoMetadata = pd.read_csv(videoFolderPath + "test_metadata_signerInd.csv")
            mhiAiFolderPath = mhiAiFolderPath
        
        df_common = videoMetadata

        df_common["mhiAi_path"] = df_common["file_path"].apply(lambda x: mhiAiFolderPath + x.replace("mp4", "npy"))
        df_common["file_path"] = df_common["file_path"].apply(lambda x: videoFolderPath + x)
        df_common["mhiAi_exists"] = df_common["mhiAi_path"].apply(os.path.exists)
        df_common = df_common[df_common["mhiAi_exists"] == True]
        
        del df_common["mhiAi_exists"]

        self.get_current_label_batch = None

        self.df = df_common
        self.global_min, self.global_max = self.get_global_min_max()

    def get_global_min_max(self):

        global_min = 1000
        global_max = -1000

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            mhi = np.load(row["mhiAi_path"])
            
            global_min = min(global_min, mhi.min())
            global_max = max(global_max, mhi.max())
          
        return global_min, global_max

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        label = row["category"]

        embedding = np.load(row["mhiAi_path"])
        embedding = torch.from_numpy(embedding)
        embedding = embedding.squeeze(0)


        return embedding, label, row["file_path"]
