# 工具函数模块
import os
import time
import shutil
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

def validate_video_file(file_path: str, max_size_mb: int = 500) -> tuple:
    """
    验证视频文件是否有效
    
    Args:
        file_path: 视频文件路径
        max_size_mb: 最大文件大小限制(MB)
    
    Returns:
        (是否有效, 错误信息)
    """
    if not os.path.exists(file_path):
        return False, f"文件不存在: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"不是有效文件: {file_path}"
    
    # 检查文件扩展名
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in supported_formats:
        return False, f"不支持的文件格式: {file_ext}。支持格式: {', '.join(supported_formats)}"
    
    # 检查文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"文件过大: {file_size_mb:.1f}MB，超过限制 {max_size_mb}MB"
    
    # 检查文件是否损坏（简单检查）
    try:
        with open(file_path, 'rb') as f:
            # 读取文件头
            header = f.read(1024)
            if len(header) < 100:
                return False, "文件可能已损坏（文件头过短）"
    except Exception as e:
        return False, f"文件读取错误: {e}"
    
    return True, ""

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件详细信息"""
    try:
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path),
            'path': file_path,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'extension': Path(file_path).suffix.lower()
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"Error: {e}"

def clean_temp_files(temp_dir: str = './temp', max_age_hours: int = 24):
    """清理临时文件"""
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    cleaned_count = 0
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    cleaned_count += 1
        except Exception as e:
            print(f"清理文件失败 {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 清理了 {cleaned_count} 个临时文件")

def format_duration(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.1f}秒"

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def similarity_to_percentage(distance: float, distance_type: str) -> float:
    """
    将距离值转换为相似度百分比
    
    Args:
        distance: 距离值
        distance_type: 距离类型 ('phash', 'cnn', 'audio')
    
    Returns:
        相似度百分比 (0-100)
    """
    if distance_type == 'phash':
        # pHash距离范围通常是0-64，距离越小越相似
        max_distance = 64
        similarity = max(0, (max_distance - distance) / max_distance)
        return similarity * 100
    
    elif distance_type == 'cnn':
        # 余弦距离范围0-2，距离越小越相似
        similarity = max(0, (2 - distance) / 2)
        return similarity * 100
    
    elif distance_type == 'audio':
        # 音频差异比例0-1，比例越小越相似
        similarity = max(0, 1 - distance)
        return similarity * 100
    
    else:
        return 0.0

def calculate_overall_similarity(phash_dist: float, cnn_dist: float, audio_diff: float) -> float:
    """
    计算综合相似度
    
    Args:
        phash_dist: pHash距离
        cnn_dist: CNN余弦距离  
        audio_diff: 音频差异比例
    
    Returns:
        综合相似度百分比 (0-100)
    """
    # 转换为相似度百分比
    phash_sim = similarity_to_percentage(phash_dist, 'phash')
    cnn_sim = similarity_to_percentage(cnn_dist, 'cnn')
    audio_sim = similarity_to_percentage(audio_diff, 'audio')
    
    # 加权平均 (视觉权重更高)
    weights = {'phash': 0.3, 'cnn': 0.4, 'audio': 0.3}
    overall_similarity = (
        phash_sim * weights['phash'] + 
        cnn_sim * weights['cnn'] + 
        audio_sim * weights['audio']
    )
    
    return overall_similarity

def is_similar(result: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    """
    根据阈值判断视频是否相似
    
    Args:
        result: 分析结果
        thresholds: 阈值配置
    
    Returns:
        是否相似
    """
    phash_similar = result.get('pHash_distance', float('inf')) < thresholds.get('phash_threshold', 15)
    cnn_similar = result.get('cnn_cosine_distance', float('inf')) < thresholds.get('cnn_threshold', 0.3)
    audio_similar = result.get('audio_difference_ratio', float('inf')) < thresholds.get('audio_threshold', 0.4)
    
    # 计算整体相似度
    overall_sim = calculate_overall_similarity(
        result.get('pHash_distance', 0),
        result.get('cnn_cosine_distance', 0), 
        result.get('audio_difference_ratio', 0)
    )
    
    overall_similar = overall_sim > (thresholds.get('overall_threshold', 0.6) * 100)
    
    # 至少两个维度相似，或整体相似度高
    similarity_count = sum([phash_similar, cnn_similar, audio_similar])
    
    return similarity_count >= 2 or overall_similar

def create_memory_monitor():
    """创建内存监控器"""
    import psutil
    
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    return get_memory_usage

def safe_remove_file(file_path: str):
    """安全删除文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"⚠️  删除文件失败 {file_path}: {e}")

def ensure_directory(directory: str):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 创建目录: {directory}")

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1, message: str = ""):
        """更新进度"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "未知"
        
        status = f"\r{self.description}: {percentage:.1f}% ({self.current}/{self.total}) - 预计剩余: {eta_str}"
        if message:
            status += f" - {message}"
        
        print(status, end='', flush=True)
        
        if self.current >= self.total:
            print()  # 换行
    
    def finish(self, message: str = "完成"):
        """完成进度跟踪"""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {message} - 总用时: {format_duration(elapsed)}")