# 🎬 智能视频分析系统 (Advanced Video Analysis System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

> 🚀 **专业级视频内容分析与隐形水印检测系统**  
> 集成多维度相似度检测、TikTok/抖音水印识别、批量处理等先进功能

## 📋 目录

- [项目概述](#-项目概述)
- [核心功能](#-核心功能)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [使用指南](#-使用指南)
- [API文档](#-api文档)
- [配置说明](#-配置说明)
- [常见问题](#-常见问题)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🎯 项目概述

本项目是一个**专业级视频内容分析系统**，采用多种先进的计算机视觉、深度学习和数字信号处理技术，为视频内容分析提供全方位解决方案。

### 🌟 主要亮点

- **🔍 多维度相似度检测**: 结合视觉、语义、音频三重检测技术
- **🎭 隐形水印识别**: 专业检测TikTok、抖音等平台的隐形标识
- **⚡ 高性能处理**: 多线程并行优化，支持大规模批量处理
- **🖥️ 现代化界面**: 直观易用的图形用户界面
- **⚙️ 智能配置**: 灵活的参数调优和自适应阈值
- **📊 详细报告**: 多格式输出和可视化分析结果

### 🎯 应用场景

- **📺 内容去重**: 识别和清理重复视频内容
- **🛡️ 版权保护**: 检测视频来源和版权信息
- **🔬 学术研究**: 数字水印和视频分析算法研究
- **📈 媒体分析**: 大规模视频内容分析和管理
- **🕵️ 取证分析**: 视频真实性和完整性验证

## ✨ 核心功能

### 🎬 视频相似度分析系统

#### 1. **感知哈希 (pHash) 检测**
- 基于DCT变换的图像指纹技术
- 抗压缩、缩放、亮度变化
- 毫秒级快速比对
- 支持自定义哈希尺寸 (4-16位)

#### 2. **CNN深度学习分析**
- 使用预训练ResNet50模型
- 提取高维语义特征向量
- 理解视频内容语义相似性
- 支持批量特征提取优化

#### 3. **音频频谱分析**
- FFT频域变换音频指纹
- 检测背景音乐相似性
- 抗噪声和压缩干扰
- 支持多声道音频处理

#### 4. **智能融合算法**
- 多维度特征加权融合
- 自适应阈值判断
- 置信度评估系统
- 详细相似度报告

### 🔍 TikTok & 抖音水印检测系统

#### 检测技术矩阵
| 技术类型 | 检测方法 | 应用领域 | 检测精度 |
|---------|----------|----------|----------|
| **元数据分析** | 文件头签名识别 | 平台来源检测 | 95% |
| **DCT频域水印** | 离散余弦变换分析 | 频域隐藏水印 | 85% |
| **LSB隐写术** | 最低位平面分析 | 像素级隐藏信息 | 80% |
| **小波变换** | 多分辨率频域分析 | 抗压缩水印 | 82% |
| **音频水印** | 频谱异常检测 | 音轨隐藏标识 | 78% |
| **时域模式** | 帧序列分析 | 时间戳水印 | 75% |
| **文件结构** | 二进制模式识别 | 容器级标识 | 90% |
| **像素异常** | 统计分布分析 | 视觉不可见修改 | 72% |

#### 支持检测的水印类型
- ✅ **平台签名水印**: TikTok/抖音元数据标识
- ✅ **视觉不可见水印**: 频域和空域隐藏标识
- ✅ **音频隐藏水印**: 高频段和静音段信号
- ✅ **时间戳水印**: 基于时序的隐藏标识
- ✅ **用户ID水印**: 个人身份隐藏标识
- ✅ **设备指纹**: 录制设备特征码

## 🏗️ 技术架构

### 🛠️ 技术栈详情

#### 核心依赖
- **🐍 Python 3.7+**: 主要开发语言
- **📷 OpenCV 4.5+**: 计算机视觉处理
- **🧠 TensorFlow 2.8+**: 深度学习框架
- **🎵 librosa 0.9+**: 音频信号处理
- **🎬 moviepy 1.0+**: 视频文件处理
- **🖼️ Pillow 8.0+**: 图像处理库
- **📊 NumPy 1.21+**: 数值计算基础

#### 专业库
- **🔤 imagehash**: 感知哈希算法
- **🌊 PyWavelets**: 小波变换
- **📈 scipy**: 科学计算
- **📋 PyYAML**: 配置文件处理
- **💻 psutil**: 系统监控
- **📊 matplotlib**: 数据可视化

## 🚀 快速开始

### 📋 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10 / macOS 10.14 / Ubuntu 18.04 | 最新版本 |
| **Python** | 3.7 | 3.9+ |
| **内存** | 4GB RAM | 8GB+ RAM |
| **存储** | 2GB 可用空间 | 10GB+ SSD |
| **显卡** | 集成显卡 | 独立GPU (CUDA支持) |

### ⚡ 一键安装

#### 方法一：自动安装脚本
```bash
# 克隆项目
git clone https://github.com/benjaminhou2/vediotest.git
cd vediotest

# 运行自动安装
python setup.py
```

#### 方法二：手动安装
```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装水印检测专用依赖
pip install -r watermark_requirements.txt

# 3. 创建必要目录
mkdir -p temp reports examples
```

### 🎯 快速体验

#### 1. 启动演示程序
```bash
python demo.py
```

#### 2. 图形界面体验
```bash
python gui.py
```

#### 3. 命令行快速检测
```bash
# 视频相似度分析
python video_analyzer.py video1.mp4 video2.mp4

# 水印检测
python watermark_detector.py tiktok_video.mp4

# 快速水印检查
python quick_watermark_check.py douyin_video.mp4
```

#### 4. 批量处理体验
```bash
# 批量分析目录
python batch_processor.py directory /path/to/videos

# 查找重复视频
python batch_processor.py directory /path/to/videos --find-duplicates
```

## 📁 项目结构

```
vediotest/
├── 📄 README.md                    # 项目说明文档
├── 📄 LICENSE                      # MIT开源许可证
├── 📄 .gitignore                   # Git忽略文件配置
├── 📄 requirements.txt             # Python依赖包列表
├── 📄 setup.py                     # 自动安装脚本
├── 📄 config.yaml                  # 系统配置文件
│
├── 🎬 核心分析模块/
│   ├── 📄 main.py                  # 原始核心算法实现
│   ├── 📄 video_analyzer.py        # 优化后的视频分析引擎
│   ├── 📄 config.py                # 配置管理系统
│   └── 📄 utils.py                 # 通用工具函数库
│
├── 🔍 水印检测模块/
│   ├── 📄 watermark_detector.py    # 专业水印检测引擎
│   ├── 📄 watermark_demo.py        # 水印检测演示程序
│   ├── 📄 quick_watermark_check.py # 快速水印检查工具
│   ├── 📄 watermark_requirements.txt # 水印检测专用依赖
│   └── 📄 WATERMARK_README.md      # 水印检测详细文档
│
├── 🖥️ 用户界面模块/
│   ├── 📄 gui.py                   # 现代化图形用户界面
│   └── 📄 demo.py                  # 交互式演示程序
│
├── 📦 批量处理模块/
│   └── 📄 batch_processor.py       # 高性能批量处理器
│
├── 📁 工作目录/
│   ├── 📁 temp/                    # 临时文件存储
│   ├── 📁 reports/                 # 分析报告输出
│   └── 📁 examples/                # 示例文件目录
│
└── 📚 文档目录/
    └── 📄 WATERMARK_README.md      # 水印检测技术文档
```

## 📖 使用指南

### 🎬 视频相似度分析

#### 基础用法
```python
from video_analyzer import VideoAnalyzer

# 创建分析器
analyzer = VideoAnalyzer()

# 分析两个视频的相似度
result = analyzer.analyze_video_similarity('video1.mp4', 'video2.mp4')

# 打印结果
print(f"综合相似度: {result['overall_similarity_percent']:.1f}%")
print(f"是否相似: {'是' if result['is_similar'] else '否'}")
```

### 🔍 水印检测使用

#### 单个视频检测
```python
from watermark_detector import WatermarkDetector

# 创建检测器
detector = WatermarkDetector()

# 检测水印
result = detector.analyze_video('tiktok_video.mp4')

# 查看检测结果
detector.print_summary(result)

# 保存详细报告
detector.save_results(result, 'watermark_report.json')
```

## ⚙️ 配置说明

### config.yaml配置文件
```yaml
# 处理参数配置
processing:
  frame_interval: 1          # 帧采样间隔(秒)
  hash_size: 8               # pHash尺寸(4-16)
  cnn_image_size: [224, 224] # CNN输入尺寸
  max_workers: 4             # 并行线程数

# 相似度阈值配置
thresholds:
  phash_threshold: 15        # pHash距离阈值
  cnn_threshold: 0.3         # CNN余弦距离阈值
  audio_threshold: 0.4       # 音频差异阈值
  overall_threshold: 0.6     # 综合相似度阈值

# 文件处理配置
files:
  supported_formats: ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
  temp_dir: './temp'         # 临时文件目录
  output_dir: './reports'    # 报告输出目录
  max_file_size_mb: 500      # 最大文件大小限制
```

## ❓ 常见问题

### 🔧 安装问题

#### Q: 安装TensorFlow时出现版本冲突
**A:** 使用虚拟环境隔离依赖
```bash
python -m venv video_analysis_env
# Windows
video_analysis_env\Scripts\activate
# macOS/Linux  
source video_analysis_env/bin/activate
pip install -r requirements.txt
```

#### Q: 视频分析速度很慢怎么办？
**A:** 多种优化策略
```python
# 1. 减少帧采样频率
config.processing.frame_interval = 2  # 每2秒一帧

# 2. 降低哈希精度
config.processing.hash_size = 4  # 降低到4位

# 3. 增加并行线程
config.processing.max_workers = 8
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

#### 🐛 报告问题
- 使用GitHub Issues报告bugs
- 提供详细的重现步骤
- 包含系统环境信息

#### 💡 功能建议
- 提交Feature Request
- 详细描述需求场景
- 说明预期的实现方式

#### 🔧 代码贡献
- Fork项目到个人仓库
- 创建功能分支 (`git checkout -b feature/AmazingFeature`)
- 提交更改 (`git commit -m 'Add some AmazingFeature'`)
- 推送到分支 (`git push origin feature/AmazingFeature`)
- 创建Pull Request

## 📄 许可证

本项目采用 **MIT License** 开源许可证。

### ⚠️ 免责声明

- **学术用途**: 本软件仅供学术研究和技术学习使用
- **法律合规**: 使用者需遵守当地法律法规和平台使用条款
- **版权尊重**: 请勿用于侵犯他人版权或知识产权的行为
- **结果参考**: 检测结果仅供技术参考，不构成法律依据
- **风险自担**: 使用本软件产生的任何风险由使用者自行承担

### 🙏 致谢

感谢以下开源项目和技术社区的支持：
- **OpenCV**: 计算机视觉基础库
- **TensorFlow**: 深度学习框架
- **librosa**: 音频信号处理库
- **moviepy**: 视频处理工具
- **NumPy & SciPy**: 科学计算基础
- **Python社区**: 丰富的生态系统

---

## 📞 联系方式

- **GitHub**: [benjaminhou2/vediotest](https://github.com/benjaminhou2/vediotest)
- **Issues**: [项目问题反馈](https://github.com/benjaminhou2/vediotest/issues)

---

<div align="center">

**🎬 让视频分析更智能，让内容检测更精准！**

[![Star this repo](https://img.shields.io/github/stars/benjaminhou2/vediotest?style=social)](https://github.com/benjaminhou2/vediotest)
[![Fork this repo](https://img.shields.io/github/forks/benjaminhou2/vediotest?style=social)](https://github.com/benjaminhou2/vediotest/fork)

*如果这个项目对您有帮助，请给我们一个⭐！*

</div>