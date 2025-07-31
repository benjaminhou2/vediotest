#!/usr/bin/env python3
# TikTok & 抖音隐形水印检测器 - 演示程序

import os
import sys
import json
from pathlib import Path
from watermark_detector import WatermarkDetector

def demo_analysis():
    """演示水印检测功能"""
    print("🎬 TikTok & 抖音隐形水印检测器演示")
    print("=" * 50)
    
    # 检查示例视频文件
    example_videos = []
    
    # 在当前目录查找视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    for ext in video_extensions:
        videos = list(Path('.').glob(f'*{ext}'))
        example_videos.extend(videos)
    
    if not example_videos:
        print("⚠️  当前目录没有找到视频文件")
        print("\n请将TikTok或抖音视频文件放在当前目录，然后重新运行")
        print("支持的格式: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        return
    
    print(f"📁 找到 {len(example_videos)} 个视频文件:")
    for i, video in enumerate(example_videos, 1):
        print(f"  {i}. {video.name}")
    
    # 让用户选择要分析的视频
    try:
        choice = input(f"\n请选择要分析的视频 (1-{len(example_videos)}) [1]: ").strip()
        if not choice:
            choice = "1"
        
        video_index = int(choice) - 1
        if video_index < 0 or video_index >= len(example_videos):
            print("❌ 无效选择")
            return
        
        selected_video = example_videos[video_index]
        
    except (ValueError, IndexError):
        print("❌ 无效输入")
        return
    
    # 执行分析
    detector = WatermarkDetector()
    
    print(f"\n🔍 开始分析: {selected_video.name}")
    print("-" * 30)
    
    try:
        results = detector.analyze_video(str(selected_video))
        
        # 显示分析结果
        detector.print_summary(results)
        
        # 询问是否保存详细报告
        save_choice = input("\n是否保存详细分析报告? (y/n) [y]: ").strip().lower()
        if save_choice != 'n':
            output_path = detector.save_results(results)
            print(f"\n📄 详细报告已保存: {output_path}")
        
        # 询问是否分析其他视频
        if len(example_videos) > 1:
            continue_choice = input("\n是否分析其他视频? (y/n) [n]: ").strip().lower()
            if continue_choice == 'y':
                demo_analysis()
    
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")

def batch_analysis():
    """批量分析当前目录的所有视频"""
    print("📦 批量水印检测模式")
    print("=" * 30)
    
    # 查找所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        videos = list(Path('.').glob(f'*{ext}'))
        video_files.extend(videos)
    
    if not video_files:
        print("❌ 当前目录没有找到视频文件")
        return
    
    print(f"📁 找到 {len(video_files)} 个视频文件")
    
    # 确认批量处理
    confirm = input(f"确定要分析所有 {len(video_files)} 个视频吗? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消操作")
        return
    
    detector = WatermarkDetector()
    batch_results = []
    
    # 逐个分析
    for i, video_file in enumerate(video_files, 1):
        print(f"\n🔍 [{i}/{len(video_files)}] 分析: {video_file.name}")
        
        try:
            result = detector.analyze_video(str(video_file))
            batch_results.append(result)
            
            # 显示简要结果
            summary = result['summary']
            print(f"  ✅ 平台: {summary['suspected_platform']}, "
                  f"风险: {summary['risk_level']}, "
                  f"置信度: {summary['overall_confidence']:.1%}")
        
        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
            batch_results.append({
                'file_info': {'filename': video_file.name},
                'error': str(e)
            })
    
    # 保存批量结果
    batch_output = {
        'analysis_type': 'batch',
        'total_videos': len(video_files),
        'successful_analysis': len([r for r in batch_results if 'error' not in r]),
        'results': batch_results
    }
    
    timestamp = detector._get_file_info('.')['modified_time'][:10]  # 使用当前日期
    batch_file = f"batch_watermark_analysis_{timestamp}.json"
    
    try:
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_output, f, ensure_ascii=False, indent=2)
        print(f"\n📄 批量分析报告已保存: {batch_file}")
    except Exception as e:
        print(f"❌ 批量报告保存失败: {e}")

def show_detection_info():
    """显示检测原理和技术说明"""
    info_text = """
🔬 TikTok & 抖音隐形水印检测技术说明

📋 检测方法:

1. 元数据分析 (Metadata Analysis)
   • 检查视频文件头部信息
   • 搜索平台特有标识字符串
   • 分析编码器签名

2. DCT频域水印检测 (DCT Watermark Detection)
   • 对视频帧进行离散余弦变换
   • 分析中频系数异常
   • 检测频域隐藏信息

3. LSB隐写术检测 (LSB Steganography Detection)
   • 分析像素最低有效位
   • 检测LSB位面的随机性
   • 识别规律性模式

4. 小波变换分析 (DWT Analysis)
   • 多级小波分解
   • 检查各频率子带能量分布
   • 识别隐藏的水印信号

5. 音频水印检测 (Audio Watermark Detection)
   • 频谱分析检测隐藏频率
   • 静音段信号检测
   • 高频段异常能量分析

6. 时域模式分析 (Temporal Pattern Analysis)
   • 分析帧间亮度变化
   • 检测周期性模式
   • FFT频域分析

7. 像素级异常检测 (Pixel Anomaly Detection)
   • 相邻像素差异统计
   • 颜色分布熵值分析
   • 异常像素比例计算

8. 文件结构分析 (File Structure Analysis)
   • 文件头、中部、尾部熵值分析
   • 重复模式检测
   • 隐藏数据区域识别

⚠️ 免责声明:
本工具仅用于技术研究和学术目的。
请遵守相关法律法规和平台使用条款。
检测结果仅供参考，不构成法律依据。
    """
    
    print(info_text)

def main():
    """主演示程序"""
    while True:
        print("\n🎬 TikTok & 抖音隐形水印检测器")
        print("=" * 40)
        print("1. 单个视频分析")
        print("2. 批量视频分析") 
        print("3. 检测技术说明")
        print("4. 安装依赖包")
        print("0. 退出")
        print("-" * 40)
        
        try:
            choice = input("请选择操作 (0-4): ").strip()
            
            if choice == '1':
                demo_analysis()
            elif choice == '2':
                batch_analysis()
            elif choice == '3':
                show_detection_info()
            elif choice == '4':
                print("📦 安装依赖包:")
                print("pip install -r watermark_requirements.txt")
                print("\n或者手动安装:")
                print("pip install opencv-python numpy pillow moviepy librosa scipy PyWavelets matplotlib ExifRead scikit-image")
            elif choice == '0':
                print("👋 再见!")
                break
            else:
                print("⚠️  无效选择，请输入0-4")
        
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()