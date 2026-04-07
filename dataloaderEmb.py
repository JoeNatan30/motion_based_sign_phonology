import os
import torch # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from utils import crop_center_image, crop_center_video, resize_video_numpy, load_video, load_image

class videoAndImageDataset(torch.utils.data.Dataset):

    def __init__(self, datasetType="training", videoFolderPath="", mhiFolderPath="", dataset="Iconic105", transform=None, generate_mhi=False, addPath=False):
        super(videoAndImageDataset, self).__init__()

        self.generate_mhi = generate_mhi
        self.dataset = dataset

        videoFolderPath_prev = videoFolderPath
        if datasetType == "training":
            if dataset == "ASLcitizen":
                videoMetadata = pd.read_csv(videoFolderPath.replace("videos_preprocessed","splits") + "train.csv", encoding="utf-8")
            if dataset == "Iconic105":
                videoMetadata = pd.read_csv(videoFolderPath + "train_metadata_signerInd.csv", encoding="utf-8")
        elif datasetType == "validation": 
            if dataset == "ASLcitizen":
                videoMetadata = pd.read_csv(videoFolderPath.replace("videos_preprocessed","splits") + "val.csv", encoding="utf-8")
            if dataset == "Iconic105":
                videoMetadata = pd.read_csv(videoFolderPath + "val_metadata_signerInd.csv", encoding="utf-8")
        elif datasetType == "test":
            if dataset == "ASLcitizen":
                videoMetadata = pd.read_csv(videoFolderPath.replace("videos_preprocessed","splits") + "test.csv", encoding="utf-8")
            if dataset == "Iconic105":
                videoMetadata = pd.read_csv(videoFolderPath + "test_metadata_signerInd.csv",
                                                                       encoding="utf-8")
        else:                             
            videoMetadata = pd.read_csv(videoFolderPath + "metadata.csv", encoding="utf-8")

        if "ASLcitizen" == dataset:
            print(videoMetadata["Video file"][0], "<<<")
            videoMetadata["Video file"] = videoMetadata["Video file"].apply(lambda x: videoFolderPath + x)
            print(videoMetadata["Video file"][0], "<<<")
            videoMetadata["mhi_path"] = videoMetadata["Video file"].apply(lambda x: x.replace(videoFolderPath_prev, mhiFolderPath).replace("mp4", "png"))
            print(videoMetadata["mhi_path"][0], "<<<")
            videoMetadata["mhi_exists"] = videoMetadata["mhi_path"].apply(os.path.exists)
        if "Iconic105" == dataset:
            print(videoMetadata["file_path"][0], "<<<")
            videoMetadata["file_path"] = videoMetadata["file_path"].apply(lambda x: videoFolderPath + x)
            print(videoMetadata["file_path"][0], "<<<")
            videoMetadata["mhi_path"] = videoMetadata["file_path"].apply(lambda x: x.replace(videoFolderPath_prev, mhiFolderPath).replace("mp4", "png"))
            print(videoMetadata["mhi_path"][0], "<<<")
            videoMetadata["mhi_exists"] = videoMetadata["mhi_path"].apply(os.path.exists)

        if dataset not in ["ASLcitizen", "Iconic105"]:
            raise ValueError(f"Dataset no reconocido: {dataset}. Debe ser 'ASLcitizen' o 'Iconic105'.")
      
        if not generate_mhi:
            videoMetadata = videoMetadata[videoMetadata["mhi_exists"] == True]
            del videoMetadata["mhi_exists"]
        
        self.current_video_path = None
        self.addPath = addPath
        self.df = videoMetadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        if self.dataset == "ASLcitizen":
            self.current_video_path = row["Video file"]
        if self.dataset == "Iconic105":
            self.current_video_path = row["file_path"]

        if not self.generate_mhi:
            mhi = load_image(row["mhi_path"])
            mhi = crop_center_image(mhi)
            mhi = cv2.resize(mhi, (512, 512), interpolation=cv2.INTER_LINEAR)
        else:
            mhi = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.dataset == "ASLcitizen":
            video = load_video(row["Video file"])
        if self.dataset == "Iconic105":
            video = load_video(row["file_path"])

        video = np.array(video)
        if video is None:
            raise ValueError(f"Video is None. path={row['file_path']}")

        if not isinstance(video, np.ndarray):
            raise ValueError(f"Video no es np.ndarray. type={type(video)} path={row['file_path']}")

        if video.ndim != 4:
            raise ValueError(f"Video ndim != 4. ndim={video.ndim} shape={video.shape} path={row['file_path']}")
        # print(row["file_path"])
        video = crop_center_video(video)
        video = resize_video_numpy(video, size=(int(512), int(512)))    
        
        if self.addPath:
            return video, mhi, self.current_video_path
        else:
            return video, mhi

    