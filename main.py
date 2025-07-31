# 视频相似度检查系统
# 包含三种技术：
# 1. 感知哈希（pHash）图像帧比对
# 2. 音频频谱相似度比对（音频哈希）
# 3. CNN 视频帧语义特征提取（使用预训练模型）

import cv2
import numpy as np
import os
import imagehash
from PIL import Image
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# =============== 基础配置 ===============
FRAME_INTERVAL = 1  # 每秒抽取一帧
HASH_SIZE = 8       # pHash 尺寸
CNN_IMAGE_SIZE = (224, 224)

# =============== 工具函数 ===============
def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (fps * interval)) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames

def phash_image(img):
    return imagehash.phash(img, hash_size=HASH_SIZE)

def compare_hashes(hashes1, hashes2):
    distances = []
    for h1, h2 in zip(hashes1, hashes2):
        distances.append(h1 - h2)  # 汉明距离
    return np.mean(distances)

def extract_audio_hash(video_path):
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path, logger=None)
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())
    freq = np.fft.fft(samples)
    spectrum_hash = np.sign(np.real(freq[:512]))  # 取前512位频谱符号为hash
    os.remove(audio_path)
    return spectrum_hash

def compare_audio_hash(a1, a2):
    return np.sum(a1 != a2) / len(a1)  # 比例差异

# CNN 模型用于提取图像语义特征
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_cnn_feature(img):
    img = img.resize(CNN_IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x, verbose=0)
    return features.flatten()

def compare_cnn_features(f1_list, f2_list):
    distances = []
    for f1, f2 in zip(f1_list, f2_list):
        distances.append(cosine(f1, f2))
    return np.mean(distances)

# =============== 主流程 ===============
def analyze_video_similarity(video_path_1, video_path_2):
    # 图像帧抽取
    frames_1 = extract_frames(video_path_1, FRAME_INTERVAL)
    frames_2 = extract_frames(video_path_2, FRAME_INTERVAL)
    min_len = min(len(frames_1), len(frames_2))
    frames_1 = frames_1[:min_len]
    frames_2 = frames_2[:min_len]

    # 图像感知哈希对比
    hashes_1 = [phash_image(f) for f in frames_1]
    hashes_2 = [phash_image(f) for f in frames_2]
    hash_score = compare_hashes(hashes_1, hashes_2)

    # 图像CNN特征对比
    features_1 = [extract_cnn_feature(f) for f in frames_1]
    features_2 = [extract_cnn_feature(f) for f in frames_2]
    cnn_score = compare_cnn_features(features_1, features_2)

    # 音频对比
    audio_hash_1 = extract_audio_hash(video_path_1)
    audio_hash_2 = extract_audio_hash(video_path_2)
    audio_diff = compare_audio_hash(audio_hash_1, audio_hash_2)

    # 输出结果
    return {
        "pHash_distance": hash_score,
        "cnn_cosine_distance": cnn_score,
        "audio_difference_ratio": audio_diff
    }

# =============== 用法 ===============
if __name__ == '__main__':
    video1 = "example_1.mp4"
    video2 = "example_2.mp4"
    result = analyze_video_similarity(video1, video2)
    print("视频相似度分析结果：")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")