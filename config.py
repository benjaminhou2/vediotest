# 配置管理模块
import yaml
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ProcessingConfig:
    """处理参数配置"""
    frame_interval: int = 1
    hash_size: int = 8
    cnn_image_size: Tuple[int, int] = (224, 224)
    max_workers: int = 4
    chunk_size: int = 10

@dataclass
class ThresholdConfig:
    """相似度阈值配置"""
    phash_threshold: float = 15.0
    cnn_threshold: float = 0.3
    audio_threshold: float = 0.4
    overall_threshold: float = 0.6

@dataclass
class FileConfig:
    """文件处理配置"""
    supported_formats: List[str] = None
    temp_dir: str = './temp'
    output_dir: str = './reports'
    max_file_size_mb: int = 500

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

@dataclass
class GUIConfig:
    """GUI配置"""
    window_width: int = 800
    window_height: int = 600
    theme: str = 'modern'

@dataclass
class ReportConfig:
    """报告配置"""
    generate_detailed: bool = True
    include_thumbnails: bool = True
    save_comparison_frames: bool = False

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self.processing = ProcessingConfig()
        self.thresholds = ThresholdConfig()
        self.files = FileConfig()
        self.gui = GUIConfig()
        self.reports = ReportConfig()
        
        self.load_config()
        self._ensure_directories()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 更新配置对象
                if 'processing' in config_data:
                    for key, value in config_data['processing'].items():
                        if hasattr(self.processing, key):
                            setattr(self.processing, key, value)
                
                if 'thresholds' in config_data:
                    for key, value in config_data['thresholds'].items():
                        if hasattr(self.thresholds, key):
                            setattr(self.thresholds, key, value)
                
                if 'files' in config_data:
                    for key, value in config_data['files'].items():
                        if hasattr(self.files, key):
                            setattr(self.files, key, value)
                
                if 'gui' in config_data:
                    for key, value in config_data['gui'].items():
                        if hasattr(self.gui, key):
                            setattr(self.gui, key, value)
                
                if 'reports' in config_data:
                    for key, value in config_data['reports'].items():
                        if hasattr(self.reports, key):
                            setattr(self.reports, key, value)
                            
                print(f"✅ 配置文件 {self.config_file} 加载成功")
                
            except Exception as e:
                print(f"⚠️  配置文件加载失败，使用默认配置: {e}")
        else:
            print(f"⚠️  配置文件 {self.config_file} 不存在，使用默认配置")
    
    def save_config(self):
        """保存配置到文件"""
        config_data = {
            'processing': {
                'frame_interval': self.processing.frame_interval,
                'hash_size': self.processing.hash_size,
                'cnn_image_size': list(self.processing.cnn_image_size),
                'max_workers': self.processing.max_workers,
                'chunk_size': self.processing.chunk_size
            },
            'thresholds': {
                'phash_threshold': self.thresholds.phash_threshold,
                'cnn_threshold': self.thresholds.cnn_threshold,
                'audio_threshold': self.thresholds.audio_threshold,
                'overall_threshold': self.thresholds.overall_threshold
            },
            'files': {
                'supported_formats': self.files.supported_formats,
                'temp_dir': self.files.temp_dir,
                'output_dir': self.files.output_dir,
                'max_file_size_mb': self.files.max_file_size_mb
            },
            'gui': {
                'window_width': self.gui.window_width,
                'window_height': self.gui.window_height,
                'theme': self.gui.theme
            },
            'reports': {
                'generate_detailed': self.reports.generate_detailed,
                'include_thumbnails': self.reports.include_thumbnails,
                'save_comparison_frames': self.reports.save_comparison_frames
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            print(f"✅ 配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [self.files.temp_dir, self.files.output_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 创建目录: {directory}")

# 全局配置实例
config = ConfigManager()