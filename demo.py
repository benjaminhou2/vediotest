#!/usr/bin/env python3
# 视频相似度分析系统演示脚本

import os
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def check_installation():
    """检查安装状态"""
    print("🔍 检查系统安装状态...")
    
    # 检查必要文件
    required_files = [
        'video_analyzer.py',
        'gui.py', 
        'batch_processor.py',
        'config.py',
        'utils.py',
        'config.yaml',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    # 检查目录
    required_dirs = ['temp', 'reports']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 创建目录: {directory}")
    
    # 检查依赖包
    print("📦 检查Python依赖包...")
    
    required_packages = [
        'cv2', 'numpy', 'PIL', 'imagehash',
        'moviepy', 'pydub', 'scipy', 'tensorflow',
        'yaml', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def create_sample_config():
    """创建示例配置"""
    if not os.path.exists('config.yaml'):
        print("📝 创建默认配置文件...")
        from config import ConfigManager
        config_manager = ConfigManager()
        config_manager.save_config()

def show_demo_menu():
    """显示演示菜单"""
    print("\n🎬 视频相似度分析系统 - 演示菜单")
    print("=" * 50)
    print("1. 启动图形界面")
    print("2. 运行命令行示例")
    print("3. 查看配置信息")
    print("4. 检查系统状态")
    print("5. 生成测试视频")
    print("0. 退出")
    print("-" * 50)

def launch_gui():
    """启动图形界面"""
    print("🚀 启动图形界面...")
    try:
        os.system("python gui.py")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def run_cli_example():
    """运行命令行示例"""
    print("📝 命令行使用示例:")
    print("\n1. 分析两个视频:")
    print("   python video_analyzer.py video1.mp4 video2.mp4")
    print("\n2. 批量分析目录:")
    print("   python batch_processor.py directory /path/to/videos")
    print("\n3. 批量分析文件列表:")
    print("   python batch_processor.py files video1.mp4 video2.mp4 video3.mp4")
    print("\n4. 查找重复视频:")
    print("   python batch_processor.py directory /path/to/videos --find-duplicates")

def show_config_info():
    """显示配置信息"""
    try:
        from config import config
        print("\n⚙️  当前配置信息:")
        print(f"📊 处理参数:")
        print(f"  • 帧间隔: {config.processing.frame_interval}秒")
        print(f"  • 哈希大小: {config.processing.hash_size}")
        print(f"  • 并行线程: {config.processing.max_workers}")
        
        print(f"\n🎯 相似度阈值:")
        print(f"  • pHash阈值: {config.thresholds.phash_threshold}")
        print(f"  • CNN阈值: {config.thresholds.cnn_threshold}")
        print(f"  • 音频阈值: {config.thresholds.audio_threshold}")
        print(f"  • 综合阈值: {config.thresholds.overall_threshold}")
        
        print(f"\n📁 文件设置:")
        print(f"  • 最大文件大小: {config.files.max_file_size_mb}MB")
        print(f"  • 临时目录: {config.files.temp_dir}")
        print(f"  • 报告目录: {config.files.output_dir}")
        
    except Exception as e:
        print(f"❌ 配置读取失败: {e}")

def generate_test_videos():
    """生成测试视频说明"""
    print("\n🎥 测试视频生成说明:")
    print("由于生成真实视频需要大量资源，建议:")
    print("\n1. 使用现有视频文件进行测试")
    print("2. 下载一些相似的视频片段")
    print("3. 复制同一视频并进行小幅修改（如压缩、格式转换）")
    print("\n测试建议:")
    print("• 准备2-5个不同的视频文件")
    print("• 包含一些相似内容的视频")
    print("• 文件大小控制在50MB以内")

def run_system_check():
    """运行系统检查"""
    print("\n🔍 系统状态检查:")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查可用内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"可用内存: {memory.available / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB")
    except:
        print("内存信息: 无法获取")
    
    # 检查GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU设备: {len(gpus)}个")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("GPU设备: 未检测到")
    except:
        print("GPU信息: 无法获取")
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        print(f"磁盘空间: {free // (1024**3)}GB 可用 / {total // (1024**3)}GB 总共")
    except:
        print("磁盘信息: 无法获取")

def main():
    """主演示程序"""
    print("🎬 视频相似度分析系统 v2.0")
    print("欢迎使用演示程序！")
    
    # 检查安装
    if not check_installation():
        print("\n❌ 系统检查失败，请先运行 python setup.py 进行安装")
        return
    
    # 创建示例配置
    create_sample_config()
    
    # 主循环
    while True:
        show_demo_menu()
        
        try:
            choice = input("\n请选择操作 (0-5): ").strip()
            
            if choice == '1':
                launch_gui()
            elif choice == '2':
                run_cli_example()
            elif choice == '3':
                show_config_info()
            elif choice == '4':
                run_system_check()
            elif choice == '5':
                generate_test_videos()
            elif choice == '0':
                print("👋 再见！")
                break
            else:
                print("⚠️  无效选择，请输入0-5")
        
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()