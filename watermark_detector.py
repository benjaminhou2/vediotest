#!/usr/bin/env python3
# TikTok & 抖音隐形水印检测器
# 分析短视频中的各种隐形水印信息

import cv2
import numpy as np
import os
import json
import sys
from datetime import datetime
import hashlib
import struct
from PIL import Image, ImageEnhance
import wave
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.spatial.distance import cosine
from moviepy.editor import VideoFileClip
import exifread
from pathlib import Path

class WatermarkDetector:
    """TikTok & 抖音隐形水印检测器"""
    
    def __init__(self):
        self.detection_methods = [
            'metadata_analysis',      # 元数据分析
            'dct_watermark',         # DCT频域水印
            'lsb_steganography',     # LSB隐写术
            'dwt_watermark',         # 小波变换水印
            'spread_spectrum',       # 扩频水印
            'frame_correlation',     # 帧间相关性
            'audio_watermark',       # 音频水印
            'temporal_pattern',      # 时域模式
            'pixel_analysis',        # 像素级分析
            'color_channel_analysis', # 颜色通道分析
            'edge_detection_anomaly', # 边缘检测异常
            'fourier_analysis'       # 傅里叶分析
        ]
        
        self.watermark_info = {}
        
    def analyze_video(self, video_path: str) -> dict:
        """
        全面分析视频中的隐形水印
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            检测结果字典
        """
        print(f"🔍 开始分析视频: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        results = {
            'file_info': self._get_file_info(video_path),
            'analysis_time': datetime.now().isoformat(),
            'watermarks_detected': {},
            'confidence_scores': {},
            'summary': {}
        }
        
        # 1. 元数据分析
        print("📋 分析视频元数据...")
        results['watermarks_detected']['metadata'] = self._analyze_metadata(video_path)
        
        # 2. 视频帧分析
        print("🎬 分析视频帧...")
        frame_results = self._analyze_video_frames(video_path)
        results['watermarks_detected'].update(frame_results)
        
        # 3. 音频分析
        print("🎵 分析音频轨道...")
        audio_results = self._analyze_audio(video_path)
        results['watermarks_detected']['audio'] = audio_results
        
        # 4. 文件结构分析
        print("🔧 分析文件结构...")
        structure_results = self._analyze_file_structure(video_path)
        results['watermarks_detected']['file_structure'] = structure_results
        
        # 5. 时域分析
        print("⏱️ 分析时域特征...")
        temporal_results = self._analyze_temporal_patterns(video_path)
        results['watermarks_detected']['temporal'] = temporal_results
        
        # 6. 生成置信度分数
        results['confidence_scores'] = self._calculate_confidence_scores(results['watermarks_detected'])
        
        # 7. 生成分析摘要
        results['summary'] = self._generate_summary(results)
        
        print("✅ 水印分析完成")
        return results
    
    def _get_file_info(self, video_path: str) -> dict:
        """获取文件基本信息"""
        stat = os.stat(video_path)
        file_hash = self._calculate_file_hash(video_path)
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except:
            fps = frame_count = width = height = duration = 0
        
        return {
            'filename': os.path.basename(video_path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'md5_hash': file_hash,
            'video_info': {
                'fps': fps,
                'frame_count': frame_count,
                'resolution': f"{width}x{height}",
                'duration_seconds': duration
            }
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _analyze_metadata(self, video_path: str) -> dict:
        """分析视频元数据中的水印信息"""
        metadata = {}
        
        try:
            # 使用moviepy提取元数据
            clip = VideoFileClip(video_path)
            if hasattr(clip, 'reader') and hasattr(clip.reader, 'infos'):
                metadata['moviepy_info'] = clip.reader.infos
            clip.close()
        except Exception as e:
            metadata['moviepy_error'] = str(e)
        
        try:
            # 使用OpenCV提取元数据
            cap = cv2.VideoCapture(video_path)
            
            # 检查编码器信息
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = struct.pack('<I', fourcc).decode('utf-8', errors='ignore')
            
            metadata['opencv_info'] = {
                'codec': codec,
                'backend': cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown'
            }
            cap.release()
        except Exception as e:
            metadata['opencv_error'] = str(e)
        
        # 检查文件头部特征
        metadata['file_header'] = self._analyze_file_header(video_path)
        
        # 检查潜在的平台标识
        metadata['platform_signatures'] = self._detect_platform_signatures(video_path)
        
        return metadata
    
    def _analyze_file_header(self, video_path: str) -> dict:
        """分析文件头部信息"""
        header_info = {}
        
        try:
            with open(video_path, 'rb') as f:
                # 读取前1KB作为文件头
                header = f.read(1024)
                
                # 检查文件魔数
                if header[:4] == b'\x00\x00\x00\x20':
                    header_info['format'] = 'MP4'
                elif header[:3] == b'FLV':
                    header_info['format'] = 'FLV'
                elif header[:4] == b'RIFF':
                    header_info['format'] = 'AVI'
                
                # 查找可疑的字符串模式
                suspicious_patterns = [
                    b'tiktok', b'douyin', b'bytedance',
                    b'TikTok', b'Douyin', b'ByteDance',
                    b'watermark', b'signature', b'uid'
                ]
                
                found_patterns = []
                for pattern in suspicious_patterns:
                    if pattern in header:
                        found_patterns.append(pattern.decode('utf-8', errors='ignore'))
                
                header_info['suspicious_patterns'] = found_patterns
                header_info['header_hex'] = header[:100].hex()  # 前100字节的十六进制
                
        except Exception as e:
            header_info['error'] = str(e)
        
        return header_info
    
    def _detect_platform_signatures(self, video_path: str) -> dict:
        """检测平台特有的签名"""
        signatures = {
            'tiktok_indicators': [],
            'douyin_indicators': [],
            'generic_watermarks': []
        }
        
        try:
            # 读取文件的多个部分
            file_size = os.path.getsize(video_path)
            sample_positions = [0, file_size//4, file_size//2, file_size*3//4, max(0, file_size-1024)]
            
            with open(video_path, 'rb') as f:
                for pos in sample_positions:
                    f.seek(pos)
                    chunk = f.read(1024)
                    
                    # TikTok特征
                    tiktok_patterns = [
                        b'tiktok', b'tt_', b'musical.ly',
                        b'douyin', b'bytedance'
                    ]
                    
                    for pattern in tiktok_patterns:
                        if pattern in chunk.lower():
                            signatures['tiktok_indicators'].append({
                                'pattern': pattern.decode('utf-8', errors='ignore'),
                                'position': pos,
                                'context': chunk[max(0, chunk.find(pattern)-10):chunk.find(pattern)+30].hex()
                            })
                    
                    # 抖音特征  
                    douyin_patterns = [
                        b'\xe6\x8a\x96\xe9\x9f\xb3',  # "抖音" UTF-8编码
                        b'douyin', b'aweme'
                    ]
                    
                    for pattern in douyin_patterns:
                        if pattern in chunk.lower():
                            signatures['douyin_indicators'].append({
                                'pattern': pattern.hex() if len(pattern) > 10 else pattern.decode('utf-8', errors='ignore'),
                                'position': pos,
                                'context': chunk[max(0, chunk.find(pattern)-10):chunk.find(pattern)+30].hex()
                            })
                            
        except Exception as e:
            signatures['error'] = str(e)
        
        return signatures
    
    def _analyze_video_frames(self, video_path: str) -> dict:
        """分析视频帧中的隐形水印"""
        frame_results = {
            'dct_watermark': {},
            'lsb_analysis': {},
            'dwt_analysis': {},
            'pixel_anomalies': {},
            'color_channel_analysis': {},
            'edge_anomalies': {},
            'fourier_analysis': {}
        }
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 采样帧进行分析（避免处理所有帧）
        sample_frames = min(20, total_frames)
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        frames_data = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            frames_data.append(frame)
            
            # 每隔几帧进行详细分析
            if i % 5 == 0:
                print(f"  分析第 {frame_idx} 帧...")
                
                # DCT频域分析
                dct_result = self._analyze_dct_watermark(frame)
                if dct_result['confidence'] > 0.3:
                    frame_results['dct_watermark'][f'frame_{frame_idx}'] = dct_result
                
                # LSB隐写术检测
                lsb_result = self._analyze_lsb_steganography(frame)
                if lsb_result['suspicious_pixels'] > 100:
                    frame_results['lsb_analysis'][f'frame_{frame_idx}'] = lsb_result
                
                # 小波变换分析
                dwt_result = self._analyze_dwt_watermark(frame)
                if dwt_result['watermark_strength'] > 0.2:
                    frame_results['dwt_analysis'][f'frame_{frame_idx}'] = dwt_result
                
                # 像素异常检测
                pixel_result = self._analyze_pixel_anomalies(frame)
                frame_results['pixel_anomalies'][f'frame_{frame_idx}'] = pixel_result
        
        cap.release()
        
        # 帧间相关性分析
        if len(frames_data) > 1:
            frame_results['frame_correlation'] = self._analyze_frame_correlation(frames_data)
        
        # 颜色通道分析
        if frames_data:
            frame_results['color_channel_analysis'] = self._analyze_color_channels(frames_data[0])
        
        return frame_results
    
    def _analyze_dct_watermark(self, frame: np.ndarray) -> dict:
        """DCT频域水印检测"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 分块DCT变换
            block_size = 8
            h, w = gray.shape
            
            watermark_blocks = []
            suspicious_blocks = 0
            
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # DCT变换
                    dct_block = cv2.dct(block)
                    
                    # 检查中频系数异常
                    mid_freq_coeffs = dct_block[2:6, 2:6]
                    
                    # 计算异常程度
                    anomaly_score = np.std(mid_freq_coeffs) / (np.mean(np.abs(mid_freq_coeffs)) + 1e-8)
                    
                    if anomaly_score > 2.0:  # 阈值可调
                        suspicious_blocks += 1
                        watermark_blocks.append({
                            'position': (i, j),
                            'anomaly_score': float(anomaly_score),
                            'coefficients': mid_freq_coeffs.flatten().tolist()
                        })
            
            total_blocks = ((h//block_size) * (w//block_size))
            confidence = suspicious_blocks / max(total_blocks, 1)
            
            return {
                'suspicious_blocks': suspicious_blocks,
                'total_blocks': total_blocks,
                'confidence': confidence,
                'watermark_blocks': watermark_blocks[:10]  # 只保存前10个
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 0}
    
    def _analyze_lsb_steganography(self, frame: np.ndarray) -> dict:
        """LSB隐写术检测"""
        try:
            h, w, c = frame.shape
            suspicious_pixels = 0
            lsb_patterns = []
            
            # 检查每个颜色通道的LSB
            for channel in range(c):
                channel_data = frame[:, :, channel]
                
                # 提取LSB
                lsb_plane = channel_data & 1
                
                # 计算LSB的随机性
                lsb_flat = lsb_plane.flatten()
                
                # 计算连续位的分布
                consecutive_ones = 0
                consecutive_zeros = 0
                max_consecutive = 0
                current_consecutive = 1
                
                for i in range(1, len(lsb_flat)):
                    if lsb_flat[i] == lsb_flat[i-1]:
                        current_consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, current_consecutive)
                        current_consecutive = 1
                
                # 如果LSB模式过于规律，可能存在隐写
                randomness = np.std(lsb_flat)
                if randomness < 0.45 or max_consecutive > 50:
                    suspicious_pixels += np.sum(lsb_plane)
                    lsb_patterns.append({
                        'channel': channel,
                        'randomness': float(randomness),
                        'max_consecutive': max_consecutive,
                        'lsb_mean': float(np.mean(lsb_flat))
                    })
            
            return {
                'suspicious_pixels': suspicious_pixels,
                'total_pixels': h * w,
                'lsb_patterns': lsb_patterns,
                'suspicion_ratio': suspicious_pixels / (h * w)
            }
            
        except Exception as e:
            return {'error': str(e), 'suspicious_pixels': 0}
    
    def _analyze_dwt_watermark(self, frame: np.ndarray) -> dict:
        """小波变换水印检测"""
        try:
            import pywt
            
            # 转换为灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # 多级小波分解
            coeffs = pywt.wavedec2(gray, 'haar', level=3)
            
            watermark_strength = 0
            suspicious_subbands = []
            
            # 检查各个子带的异常
            for level, (cH, cV, cD) in enumerate(coeffs[1:], 1):
                # 计算高频子带的能量分布
                energy_h = np.mean(cH ** 2)
                energy_v = np.mean(cV ** 2)
                energy_d = np.mean(cD ** 2)
                
                # 检查能量分布异常
                energy_ratio = energy_d / (energy_h + energy_v + 1e-8)
                
                if energy_ratio > 1.5 or energy_ratio < 0.3:
                    watermark_strength += energy_ratio * 0.1
                    suspicious_subbands.append({
                        'level': level,
                        'energy_ratio': float(energy_ratio),
                        'energy_h': float(energy_h),
                        'energy_v': float(energy_v),
                        'energy_d': float(energy_d)
                    })
            
            return {
                'watermark_strength': min(watermark_strength, 1.0),
                'suspicious_subbands': suspicious_subbands,
                'dwt_levels': len(coeffs) - 1
            }
            
        except ImportError:
            return {'error': 'PyWavelets not installed', 'watermark_strength': 0}
        except Exception as e:
            return {'error': str(e), 'watermark_strength': 0}
    
    def _analyze_pixel_anomalies(self, frame: np.ndarray) -> dict:
        """像素级异常检测"""
        try:
            h, w, c = frame.shape
            
            # 计算相邻像素差异
            diff_h = np.abs(frame[1:, :, :] - frame[:-1, :, :])
            diff_v = np.abs(frame[:, 1:, :] - frame[:, :-1, :])
            
            # 统计异常像素
            anomaly_threshold = 30
            anomalous_h = np.sum(diff_h > anomaly_threshold)
            anomalous_v = np.sum(diff_v > anomaly_threshold)
            
            # 检查像素值分布
            pixel_hist = np.histogram(frame.flatten(), bins=256)[0]
            distribution_entropy = -np.sum((pixel_hist / np.sum(pixel_hist)) * 
                                         np.log2(pixel_hist / np.sum(pixel_hist) + 1e-8))
            
            return {
                'anomalous_horizontal': int(anomalous_h),
                'anomalous_vertical': int(anomalous_v),
                'total_pixels': h * w,
                'distribution_entropy': float(distribution_entropy),
                'anomaly_ratio': (anomalous_h + anomalous_v) / (2 * h * w)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_frame_correlation(self, frames: list) -> dict:
        """帧间相关性分析"""
        try:
            correlations = []
            
            for i in range(len(frames) - 1):
                frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
                
                # 计算结构相似性
                correlation = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF_NORMED)[0][0]
                correlations.append(correlation)
            
            # 检查异常的相关性模式
            corr_mean = np.mean(correlations)
            corr_std = np.std(correlations)
            
            # 寻找周期性模式
            fft_corr = np.fft.fft(correlations)
            dominant_freq = np.argmax(np.abs(fft_corr[1:len(fft_corr)//2])) + 1
            
            return {
                'correlation_mean': float(corr_mean),
                'correlation_std': float(corr_std),
                'correlations': [float(c) for c in correlations],
                'dominant_frequency': int(dominant_freq),
                'periodic_watermark_suspected': corr_std < 0.05 and corr_mean > 0.8
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_color_channels(self, frame: np.ndarray) -> dict:
        """颜色通道分析"""
        try:
            b, g, r = cv2.split(frame)
            
            channel_analysis = {}
            
            for name, channel in [('blue', b), ('green', g), ('red', r)]:
                # 计算通道统计
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                
                # 检查通道间的异常关系
                hist = np.histogram(channel, bins=256)[0]
                entropy = -np.sum((hist / np.sum(hist)) * np.log2(hist / np.sum(hist) + 1e-8))
                
                channel_analysis[name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'entropy': float(entropy),
                    'peak_values': np.where(hist > np.max(hist) * 0.8)[0].tolist()
                }
            
            # 检查通道间相关性
            correlations = {
                'bg_correlation': float(np.corrcoef(b.flatten(), g.flatten())[0, 1]),
                'br_correlation': float(np.corrcoef(b.flatten(), r.flatten())[0, 1]),
                'gr_correlation': float(np.corrcoef(g.flatten(), r.flatten())[0, 1])
            }
            
            return {
                'channels': channel_analysis,
                'correlations': correlations,
                'suspicious_channel_imbalance': any(abs(c) < 0.7 for c in correlations.values())
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_audio(self, video_path: str) -> dict:
        """音频水印检测"""
        try:
            # 提取音频
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                return {'error': 'No audio track found'}
            
            audio_path = 'temp_audio_analysis.wav'
            clip.audio.write_audiofile(audio_path, logger=None, verbose=False)
            clip.close()
            
            # 使用librosa分析音频
            y, sr = librosa.load(audio_path, sr=None)
            
            # 频谱分析
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 检测隐藏频率
            freq_bins = librosa.fft_frequencies(sr=sr)
            
            # 寻找异常的频率峰值
            freq_energy = np.mean(magnitude, axis=1)
            peaks = np.where(freq_energy > np.mean(freq_energy) + 2 * np.std(freq_energy))[0]
            
            # 检测静音段中的隐藏信号
            silence_threshold = 0.01
            silent_frames = np.where(np.abs(y) < silence_threshold)[0]
            
            if len(silent_frames) > 0:
                silent_audio = y[silent_frames]
                silent_energy = np.mean(silent_audio ** 2)
            else:
                silent_energy = 0
            
            # 检测不可听见的高频信号
            high_freq_start = int(len(freq_bins) * 0.8)  # 高频部分
            high_freq_energy = np.mean(freq_energy[high_freq_start:])
            
            # 清理临时文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                'sample_rate': sr,
                'duration': len(y) / sr,
                'suspicious_frequencies': [float(freq_bins[p]) for p in peaks],
                'high_freq_energy': float(high_freq_energy),
                'silent_frame_energy': float(silent_energy),
                'spectral_peaks': len(peaks),
                'watermark_suspected': len(peaks) > 5 or high_freq_energy > 0.1
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_file_structure(self, video_path: str) -> dict:
        """文件结构分析"""
        try:
            file_size = os.path.getsize(video_path)
            
            # 分析文件的不同区域
            regions = {
                'header': (0, min(8192, file_size//10)),
                'middle': (file_size//2 - 4096, file_size//2 + 4096),
                'footer': (max(0, file_size - 8192), file_size)
            }
            
            structure_analysis = {}
            
            with open(video_path, 'rb') as f:
                for region_name, (start, end) in regions.items():
                    f.seek(start)
                    data = f.read(end - start)
                    
                    # 计算熵值
                    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
                    probabilities = byte_counts / len(data)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
                    
                    # 检查重复模式
                    patterns = {}
                    for i in range(len(data) - 4):
                        pattern = data[i:i+4]
                        patterns[pattern] = patterns.get(pattern, 0) + 1
                    
                    most_common = max(patterns.values()) if patterns else 0
                    
                    structure_analysis[region_name] = {
                        'entropy': float(entropy),
                        'size': len(data),
                        'most_common_pattern_count': most_common,
                        'unique_patterns': len(patterns),
                        'suspicious': entropy < 6.0 or most_common > len(data) * 0.1
                    }
            
            return structure_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_temporal_patterns(self, video_path: str) -> dict:
        """时域模式分析"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 采样帧的平均亮度
            brightness_timeline = []
            frame_indices = np.linspace(0, total_frames-1, min(100, total_frames), dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    brightness_timeline.append(brightness)
            
            cap.release()
            
            # 分析亮度变化模式
            if len(brightness_timeline) > 10:
                # FFT分析寻找周期性
                fft_brightness = np.fft.fft(brightness_timeline)
                frequencies = np.fft.fftfreq(len(brightness_timeline))
                
                # 寻找显著的周期性
                magnitude = np.abs(fft_brightness)
                peak_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                peak_frequency = frequencies[peak_freq_idx]
                
                # 计算变化的规律性
                brightness_std = np.std(brightness_timeline)
                brightness_mean = np.mean(brightness_timeline)
                
                return {
                    'brightness_timeline': [float(b) for b in brightness_timeline],
                    'brightness_mean': float(brightness_mean),
                    'brightness_std': float(brightness_std),
                    'peak_frequency': float(peak_frequency),
                    'periodic_pattern_detected': magnitude[peak_freq_idx] > np.mean(magnitude) * 3,
                    'regularity_score': float(1.0 / (brightness_std + 1e-8))
                }
            else:
                return {'error': 'Insufficient frames for temporal analysis'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_confidence_scores(self, watermarks: dict) -> dict:
        """计算各类水印检测的置信度分数"""
        scores = {}
        
        # 元数据置信度
        metadata = watermarks.get('metadata', {})
        platform_sigs = metadata.get('platform_signatures', {})
        tiktok_count = len(platform_sigs.get('tiktok_indicators', []))
        douyin_count = len(platform_sigs.get('douyin_indicators', []))
        scores['metadata_confidence'] = min((tiktok_count + douyin_count) * 0.2, 1.0)
        
        # 视频帧置信度
        dct_data = watermarks.get('dct_watermark', {})
        if dct_data and not isinstance(dct_data, dict) or 'error' in dct_data:
            scores['dct_confidence'] = 0.0
        else:
            dct_confidence = 0.0
            for frame_data in dct_data.values():
                dct_confidence = max(dct_confidence, frame_data.get('confidence', 0))
            scores['dct_confidence'] = dct_confidence
        
        # LSB置信度
        lsb_data = watermarks.get('lsb_analysis', {})
        lsb_confidence = 0.0
        for frame_data in lsb_data.values():
            if 'suspicion_ratio' in frame_data:
                lsb_confidence = max(lsb_confidence, frame_data['suspicion_ratio'])
        scores['lsb_confidence'] = min(lsb_confidence * 2, 1.0)
        
        # 音频置信度
        audio_data = watermarks.get('audio', {})
        if 'watermark_suspected' in audio_data:
            scores['audio_confidence'] = 0.8 if audio_data['watermark_suspected'] else 0.2
        else:
            scores['audio_confidence'] = 0.0
        
        # 综合置信度
        all_scores = [score for score in scores.values() if score > 0]
        scores['overall_confidence'] = np.mean(all_scores) if all_scores else 0.0
        
        return scores
    
    def _generate_summary(self, results: dict) -> dict:
        """生成分析摘要"""
        watermarks = results['watermarks_detected']
        confidence = results['confidence_scores']
        
        # 判断平台来源
        platform = 'unknown'
        metadata = watermarks.get('metadata', {})
        platform_sigs = metadata.get('platform_signatures', {})
        
        if platform_sigs.get('tiktok_indicators'):
            platform = 'tiktok'
        elif platform_sigs.get('douyin_indicators'):
            platform = 'douyin'
        
        # 检测到的水印类型
        detected_watermarks = []
        
        if confidence.get('metadata_confidence', 0) > 0.3:
            detected_watermarks.append('metadata_signatures')
        
        if confidence.get('dct_confidence', 0) > 0.3:
            detected_watermarks.append('dct_frequency_watermark')
        
        if confidence.get('lsb_confidence', 0) > 0.3:
            detected_watermarks.append('lsb_steganography')
        
        if confidence.get('audio_confidence', 0) > 0.5:
            detected_watermarks.append('audio_watermark')
        
        # 风险评估
        overall_conf = confidence.get('overall_confidence', 0)
        if overall_conf > 0.7:
            risk_level = 'high'
        elif overall_conf > 0.4:
            risk_level = 'medium'
        elif overall_conf > 0.1:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return {
            'suspected_platform': platform,
            'detected_watermark_types': detected_watermarks,
            'risk_level': risk_level,
            'overall_confidence': overall_conf,
            'watermark_count': len(detected_watermarks),
            'analysis_complete': True,
            'recommendations': self._generate_recommendations(detected_watermarks, risk_level)
        }
    
    def _generate_recommendations(self, watermarks: list, risk_level: str) -> list:
        """生成建议"""
        recommendations = []
        
        if 'metadata_signatures' in watermarks:
            recommendations.append("检测到元数据签名，建议清理视频元数据")
        
        if 'dct_frequency_watermark' in watermarks:
            recommendations.append("检测到DCT频域水印，可能需要重新编码或压缩")
        
        if 'lsb_steganography' in watermarks:
            recommendations.append("检测到LSB隐写术，建议进行像素级处理")
        
        if 'audio_watermark' in watermarks:
            recommendations.append("检测到音频水印，建议处理音频轨道")
        
        if risk_level == 'high':
            recommendations.append("高风险：建议避免商业使用此视频")
        elif risk_level == 'medium':
            recommendations.append("中等风险：建议进一步处理后使用")
        
        if not watermarks:
            recommendations.append("未检测到明显水印，但仍建议谨慎使用")
        
        return recommendations
    
    def save_results(self, results: dict, output_path: str = None) -> str:
        """保存分析结果"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"watermark_analysis_{timestamp}.json"
            output_path = os.path.join('reports', filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"💾 分析结果已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return ""
    
    def print_summary(self, results: dict):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("🔍 TikTok & 抖音隐形水印分析报告")
        print("="*60)
        
        file_info = results['file_info']
        summary = results['summary']
        confidence = results['confidence_scores']
        
        print(f"\n📁 文件信息:")
        print(f"  • 文件名: {file_info['filename']}")
        print(f"  • 大小: {file_info['size_mb']:.2f} MB")
        print(f"  • 时长: {file_info['video_info']['duration_seconds']:.1f} 秒")
        print(f"  • 分辨率: {file_info['video_info']['resolution']}")
        
        print(f"\n🎯 检测结果:")
        print(f"  • 疑似平台: {summary['suspected_platform'].upper()}")
        print(f"  • 风险级别: {summary['risk_level'].upper()}")
        print(f"  • 整体置信度: {summary['overall_confidence']:.2%}")
        print(f"  • 检测到水印类型: {len(summary['detected_watermark_types'])} 种")
        
        if summary['detected_watermark_types']:
            print(f"\n🔍 水印类型详情:")
            for wm_type in summary['detected_watermark_types']:
                print(f"  • {wm_type}")
        
        print(f"\n📊 详细置信度:")
        for conf_type, score in confidence.items():
            if score > 0:
                print(f"  • {conf_type}: {score:.2%}")
        
        if summary['recommendations']:
            print(f"\n💡 建议:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*60)

def main():
    """命令行主程序"""
    if len(sys.argv) != 2:
        print("使用方法: python watermark_detector.py <视频文件路径>")
        print("示例: python watermark_detector.py tiktok_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        sys.exit(1)
    
    # 创建检测器
    detector = WatermarkDetector()
    
    try:
        print("🎬 TikTok & 抖音隐形水印检测器 v1.0")
        print("-" * 50)
        
        # 执行分析
        results = detector.analyze_video(video_path)
        
        # 显示摘要
        detector.print_summary(results)
        
        # 保存详细结果
        output_path = detector.save_results(results)
        
        print(f"\n📄 详细报告: {output_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()