import cv2

import numpy as np


def imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image


def crop_center_video(video):
    """
    Recorta el cuadrado central de cada frame de un video.
    video: array con forma (B, T, H, W, C)
    """
    T, H, W, C = video.shape
    new_size = min(H, W)
    start_y = (H - new_size) // 2
    start_x = (W - new_size) // 2
    return video[:, start_y:start_y+new_size, start_x:start_x+new_size, :]


def crop_center_image(image):
    # Suponiendo que image es un array de NumPy con forma (altura, anchura, canales)
    h, w, c = image.shape
    new_size = min(h, w)
    start_x = (w - new_size) // 2
    start_y = (h - new_size) // 2
    cropped = image[start_y:start_y+new_size, start_x:start_x+new_size, :]
    return cropped


def resize_video_numpy(video, size=(1024, 1024)):
    """
    Redimensiona un video en formato NumPy.
    
    Parámetros:
      video: numpy array de forma [B, T, H, W, C]
      size: tuple (ancho, alto) deseado, p.ej. (1024, 1024)
    
    Retorna:
      video_resized: numpy array de forma [B, T, size[1], size[0], C]
    """
    T, H, W, C = video.shape
    resized_video = np.zeros(( T, size[1], size[0], C), dtype=video.dtype)
    

    for t in range(T):
        # cv2.resize recibe (ancho, alto)
        resized_video[t] = cv2.resize(video[t], size, interpolation=cv2.INTER_LINEAR)

    return resized_video

def load_image(image_path):
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen desde la ruta: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    threshold_value = 50  # Ajusta según lo que consideres "ruido"
    _, mhi_thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO)
    return mhi_thresh


def load_video(video_path):
    #print(f"Cargando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("Error al abrir el video:", video_path)
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
