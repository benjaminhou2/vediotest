#!/usr/bin/env python3
# 快速水印检测脚本 - 简化版本

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def quick_check(video_path):
    """快速检测视频中的基本水印特征"""
    print(f"🔍 快速检测: {os.path.basename(video_path)}")
    
    # 基本文件检查
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print(f"📁 文件大小: {file_size:.1f} MB")
    
    # 检查文件头
    with open(video_path, 'rb') as f:
        header = f.read(1024)
        
        # 查找平台标识
        tiktok_indicators = [b'tiktok', b'douyin', b'bytedance', b'musical.ly']
        found_indicators = []
        
        for indicator in tiktok_indicators:
            if indicator in header.lower():
                found_indicators.append(indicator.decode())
        
        if found_indicators:
            print(f"🎯 发现平台标识: {', '.join(found_indicators)}")
        else:
            print("❓ 未在文件头发现明显平台标识")
    
    # 简单视频帧分析
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if ret:
            # 检查分辨率
            h, w = frame.shape[:2]
            print(f"📺 分辨率: {w}x{h}")
            
            # 典型TikTok/抖音分辨率
            if (w, h) in [(1080, 1920), (720, 1280), (1080, 1440)]:
                print("📱 检测到竖屏短视频格式")
            
            # 简单边缘检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            if edge_density > 0.1:
                print(f"🔍 高边缘密度 ({edge_density:.3f}) - 可能有水印覆盖")
            
            # 检查颜色通道异常
            b, g, r = cv2.split(frame)
            color_means = [np.mean(b), np.mean(g), np.mean(r)]
            color_balance = max(color_means) / min(color_means) if min(color_means) > 0 else 0
            
            if color_balance > 1.5:
                print(f"🎨 颜色通道不平衡 ({color_balance:.2f}) - 可能有隐形处理")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ 视频分析失败: {e}")
    
    print("✅ 快速检测完成")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python quick_watermark_check.py <视频文件>")
        
        # 自动查找当前目录的视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path('.').glob(f'*{ext}'))
        
        if video_files:
            print(f"\n当前目录的视频文件:")
            for i, video in enumerate(video_files, 1):
                print(f"  {i}. {video.name}")
            
            try:
                choice = int(input(f"\n选择要检测的视频 (1-{len(video_files)}): "))
                if 1 <= choice <= len(video_files):
                    quick_check(str(video_files[choice-1]))
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入数字")
        
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        sys.exit(1)
    
    quick_check(video_path)

if __name__ == '__main__':
    main()