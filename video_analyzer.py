# 优化后的视频分析引擎
import cv2
import numpy as np
import os
import sys
import time
import traceback
import imagehash
from PIL import Image
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
from datetime import datetime

from config import config
from utils import (
    validate_video_file, get_file_info, calculate_file_hash,
    clean_temp_files, format_duration, ProgressTracker,
    similarity_to_percentage, calculate_overall_similarity,
    is_similar, safe_remove_file, create_memory_monitor
)

class VideoAnalyzer:
    """优化后的视频分析器"""
    
    def __init__(self):
        self.config = config
        self.memory_monitor = create_memory_monitor()
        self.progress_lock = Lock()
        
        # 加载CNN模型
        print("🔄 加载ResNet50模型...")
        self.resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        print("✅ 模型加载完成")
        
        # 清理旧的临时文件
        clean_temp_files(self.config.files.temp_dir)
    
    def extract_frames_parallel(self, video_path: str, interval: int = 1, max_frames: int = None) -> list:
        """
        并行提取视频帧
        
        Args:
            video_path: 视频文件路径
            interval: 帧间隔(秒)
            max_frames: 最大帧数限制
        
        Returns:
            提取的帧列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # 计算要提取的帧索引
        frame_indices = []
        frame_step = int(fps * interval)
        for i in range(0, total_frames, frame_step):
            frame_indices.append(i)
            if max_frames and len(frame_indices) >= max_frames:
                break
        
        print(f"📹 视频时长: {format_duration(duration)}, 将提取 {len(frame_indices)} 帧")
        
        frames = []
        progress = ProgressTracker(len(frame_indices), f"提取帧 ({os.path.basename(video_path)})")
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            
            progress.update(1, f"帧 {idx+1}")
            
            # 内存监控
            if idx % 10 == 0:
                memory_mb = self.memory_monitor()
                if memory_mb > 1000:  # 超过1GB警告
                    print(f"\n⚠️  内存使用: {memory_mb:.1f}MB")
        
        progress.finish()
        cap.release()
        return frames
    
    def phash_image(self, img: Image.Image) -> str:
        """计算图像感知哈希"""
        try:
            return str(imagehash.phash(img, hash_size=self.config.processing.hash_size))
        except Exception as e:
            print(f"⚠️  pHash计算失败: {e}")
            return "0" * (self.config.processing.hash_size ** 2)
    
    def compare_hashes_parallel(self, hashes1: list, hashes2: list) -> float:
        """并行比较哈希值"""
        def calculate_distance(h1, h2):
            try:
                # 将字符串哈希转换为imagehash对象进行比较
                hash1 = imagehash.hex_to_hash(h1)
                hash2 = imagehash.hex_to_hash(h2)
                return hash1 - hash2
            except:
                return 64  # 最大距离
        
        with ThreadPoolExecutor(max_workers=self.config.processing.max_workers) as executor:
            futures = [executor.submit(calculate_distance, h1, h2) 
                      for h1, h2 in zip(hashes1, hashes2)]
            
            distances = []
            for future in as_completed(futures):
                try:
                    distances.append(future.result())
                except Exception as e:
                    print(f"⚠️  哈希比较失败: {e}")
                    distances.append(64)
        
        return np.mean(distances) if distances else 64
    
    def extract_audio_hash_safe(self, video_path: str) -> np.ndarray:
        """安全提取音频哈希"""
        temp_audio_path = os.path.join(self.config.files.temp_dir, 
                                     f"temp_audio_{int(time.time())}.wav")
        
        try:
            # 提取音频
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                print("⚠️  视频无音频轨道")
                return np.zeros(512)
            
            clip.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
            clip.close()
            
            # 处理音频
            audio = AudioSegment.from_wav(temp_audio_path)
            samples = np.array(audio.get_array_of_samples())
            
            # 限制采样数避免内存问题
            if len(samples) > 1000000:  # 1M采样点
                samples = samples[:1000000]
            
            freq = np.fft.fft(samples)
            spectrum_hash = np.sign(np.real(freq[:512]))
            
            return spectrum_hash
            
        except Exception as e:
            print(f"⚠️  音频提取失败: {e}")
            return np.zeros(512)
        finally:
            safe_remove_file(temp_audio_path)
    
    def compare_audio_hash(self, a1: np.ndarray, a2: np.ndarray) -> float:
        """比较音频哈希"""
        try:
            if len(a1) == 0 or len(a2) == 0:
                return 1.0
            return np.sum(a1 != a2) / len(a1)
        except:
            return 1.0
    
    def extract_cnn_features_batch(self, images: list) -> list:
        """批量提取CNN特征"""
        features = []
        batch_size = 8  # 控制批处理大小避免内存问题
        
        progress = ProgressTracker(len(images), "提取CNN特征")
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_features = []
            
            try:
                # 预处理批量图像
                batch_arrays = []
                for img in batch:
                    img_resized = img.resize(self.config.processing.cnn_image_size)
                    x = image.img_to_array(img_resized)
                    x = preprocess_input(x)
                    batch_arrays.append(x)
                
                if batch_arrays:
                    batch_input = np.stack(batch_arrays)
                    batch_pred = self.resnet_model.predict(batch_input, verbose=0)
                    
                    for pred in batch_pred:
                        batch_features.append(pred.flatten())
                
                features.extend(batch_features)
                
            except Exception as e:
                print(f"⚠️  CNN特征提取失败: {e}")
                # 添加零特征向量作为备用
                for _ in batch:
                    features.append(np.zeros(2048))  # ResNet50输出特征维度
            
            progress.update(len(batch))
        
        progress.finish()
        return features
    
    def compare_cnn_features_parallel(self, f1_list: list, f2_list: list) -> float:
        """并行比较CNN特征"""
        def calculate_cosine(f1, f2):
            try:
                if len(f1) == 0 or len(f2) == 0:
                    return 1.0
                return cosine(f1, f2)
            except:
                return 1.0
        
        with ThreadPoolExecutor(max_workers=self.config.processing.max_workers) as executor:
            futures = [executor.submit(calculate_cosine, f1, f2) 
                      for f1, f2 in zip(f1_list, f2_list)]
            
            distances = []
            for future in as_completed(futures):
                try:
                    distances.append(future.result())
                except Exception as e:
                    print(f"⚠️  特征比较失败: {e}")
                    distances.append(1.0)
        
        return np.mean(distances) if distances else 1.0
    
    def analyze_video_similarity(self, video_path_1: str, video_path_2: str, 
                               progress_callback=None) -> dict:
        """
        分析两个视频的相似度
        
        Args:
            video_path_1: 第一个视频路径
            video_path_2: 第二个视频路径
            progress_callback: 进度回调函数
        
        Returns:
            分析结果字典
        """
        start_time = time.time()
        
        # 验证文件
        for i, video_path in enumerate([video_path_1, video_path_2], 1):
            valid, error = validate_video_file(video_path, self.config.files.max_file_size_mb)
            if not valid:
                raise ValueError(f"视频{i}验证失败: {error}")
        
        print(f"\n🎬 开始分析视频相似度")
        print(f"📁 视频1: {os.path.basename(video_path_1)}")
        print(f"📁 视频2: {os.path.basename(video_path_2)}")
        
        result = {
            'video1_info': get_file_info(video_path_1),
            'video2_info': get_file_info(video_path_2),
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_used': {
                'frame_interval': self.config.processing.frame_interval,
                'hash_size': self.config.processing.hash_size,
                'cnn_image_size': self.config.processing.cnn_image_size
            }
        }
        
        try:
            # 1. 提取视频帧
            if progress_callback:
                progress_callback("提取视频帧", 10)
            
            frames_1 = self.extract_frames_parallel(video_path_1, self.config.processing.frame_interval)
            frames_2 = self.extract_frames_parallel(video_path_2, self.config.processing.frame_interval)
            
            # 统一帧数
            min_len = min(len(frames_1), len(frames_2))
            if min_len == 0:
                raise ValueError("无法提取视频帧")
            
            frames_1 = frames_1[:min_len]
            frames_2 = frames_2[:min_len]
            
            print(f"📊 使用 {min_len} 帧进行分析")
            
            # 2. pHash分析
            if progress_callback:
                progress_callback("计算感知哈希", 30)
            
            print("🔍 计算感知哈希...")
            hashes_1 = [self.phash_image(f) for f in frames_1]
            hashes_2 = [self.phash_image(f) for f in frames_2]
            hash_distance = self.compare_hashes_parallel(hashes_1, hashes_2)
            
            # 3. CNN特征分析
            if progress_callback:
                progress_callback("提取CNN特征", 60)
            
            print("🧠 提取CNN特征...")
            features_1 = self.extract_cnn_features_batch(frames_1)
            features_2 = self.extract_cnn_features_batch(frames_2)
            cnn_distance = self.compare_cnn_features_parallel(features_1, features_2)
            
            # 4. 音频分析
            if progress_callback:
                progress_callback("分析音频", 80)
            
            print("🎵 分析音频...")
            audio_hash_1 = self.extract_audio_hash_safe(video_path_1)
            audio_hash_2 = self.extract_audio_hash_safe(video_path_2)
            audio_diff = self.compare_audio_hash(audio_hash_1, audio_hash_2)
            
            # 5. 计算相似度结果
            if progress_callback:
                progress_callback("生成报告", 95)
            
            # 基础距离指标
            result.update({
                'pHash_distance': float(hash_distance),
                'cnn_cosine_distance': float(cnn_distance),
                'audio_difference_ratio': float(audio_diff)
            })
            
            # 转换为相似度百分比
            result.update({
                'phash_similarity_percent': similarity_to_percentage(hash_distance, 'phash'),
                'cnn_similarity_percent': similarity_to_percentage(cnn_distance, 'cnn'),
                'audio_similarity_percent': similarity_to_percentage(audio_diff, 'audio')
            })
            
            # 计算综合相似度
            overall_similarity = calculate_overall_similarity(hash_distance, cnn_distance, audio_diff)
            result['overall_similarity_percent'] = overall_similarity
            
            # 相似性判断
            is_similar_result = is_similar(result, {
                'phash_threshold': self.config.thresholds.phash_threshold,
                'cnn_threshold': self.config.thresholds.cnn_threshold,
                'audio_threshold': self.config.thresholds.audio_threshold,
                'overall_threshold': self.config.thresholds.overall_threshold
            })
            result['is_similar'] = is_similar_result
            
            # 计算处理时间
            processing_time = time.time() - start_time
            result['processing_time_seconds'] = processing_time
            
            if progress_callback:
                progress_callback("完成", 100)
            
            print(f"\n✅ 分析完成 - 用时: {format_duration(processing_time)}")
            print(f"📊 综合相似度: {overall_similarity:.1f}%")
            print(f"🎯 判断结果: {'相似' if is_similar_result else '不相似'}")
            
            return result
            
        except Exception as e:
            error_msg = f"分析过程中出现错误: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"📋 错误详情:\n{traceback.format_exc()}")
            
            result.update({
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'processing_time_seconds': time.time() - start_time
            })
            return result
    
    def generate_detailed_report(self, result: dict, output_path: str = None) -> str:
        """生成详细分析报告"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.config.files.output_dir, 
                                     f"similarity_report_{timestamp}.json")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"📄 详细报告已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 报告保存失败: {e}")
            return ""

def main():
    """命令行主程序"""
    if len(sys.argv) != 3:
        print("用法: python video_analyzer.py <视频1> <视频2>")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    # 创建分析器
    analyzer = VideoAnalyzer()
    
    try:
        # 执行分析
        result = analyzer.analyze_video_similarity(video1_path, video2_path)
        
        # 显示结果
        print("\n" + "="*60)
        print("📊 视频相似度分析结果")
        print("="*60)
        
        if 'error' in result:
            print(f"❌ 分析失败: {result['error']}")
            return
        
        print(f"🎬 视频1: {result['video1_info']['name']}")
        print(f"🎬 视频2: {result['video2_info']['name']}")
        print()
        
        print("📈 相似度指标:")
        print(f"  • 视觉相似度 (pHash): {result['phash_similarity_percent']:.1f}%")
        print(f"  • 语义相似度 (CNN):   {result['cnn_similarity_percent']:.1f}%")
        print(f"  • 音频相似度:         {result['audio_similarity_percent']:.1f}%")
        print(f"  • 综合相似度:         {result['overall_similarity_percent']:.1f}%")
        print()
        
        print(f"🎯 判断结果: {'✅ 相似' if result['is_similar'] else '❌ 不相似'}")
        print(f"⏱️  处理时间: {format_duration(result['processing_time_seconds'])}")
        
        # 生成详细报告
        if config.reports.generate_detailed:
            report_path = analyzer.generate_detailed_report(result)
            if report_path:
                print(f"📄 详细报告: {report_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()