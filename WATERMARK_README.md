# TikTok & 抖音隐形水印检测器

## 🎯 项目简介
这是一个专门用于检测TikTok和抖音短视频中隐形水印信息的Python工具。该工具使用多种先进的数字信号处理和计算机视觉技术，能够全面分析视频文件中可能存在的各种隐形标识。

## ⚠️ 重要声明
- **本工具仅用于技术研究和学术目的**
- **请遵守相关法律法规和平台使用条款**
- **检测结果仅供参考，不构成法律依据**
- **禁止用于非法用途**

## 🔬 检测技术

### 1. 元数据分析 (Metadata Analysis)
- 检查视频文件头部和元数据信息
- 搜索平台特有的标识字符串
- 分析编码器和容器格式签名
- 检测隐藏在文件结构中的平台标识

### 2. DCT频域水印检测 (DCT Watermark Detection)
- 对视频帧进行8x8块离散余弦变换
- 分析中频系数的异常模式
- 检测频域中隐藏的水印信号
- 识别不可见的频率域修改

### 3. LSB隐写术检测 (LSB Steganography Detection)
- 分析像素最低有效位的分布
- 检测LSB位面的随机性和规律性
- 识别可能的隐写术模式
- 计算像素修改的可疑程度

### 4. 小波变换分析 (DWT Analysis)
- 多级小波分解分析
- 检查各频率子带的能量分布
- 识别隐藏在小波域的水印信号
- 检测高频细节中的异常

### 5. 音频水印检测 (Audio Watermark Detection)
- 音频频谱分析和异常频率检测
- 静音段中的隐藏信号检测
- 高频段异常能量分析
- 不可听见频率范围的信号检测

### 6. 时域模式分析 (Temporal Pattern Analysis)
- 分析视频帧间的亮度变化模式
- 检测周期性的时域特征
- FFT频域分析寻找隐藏的周期信号
- 识别时间序列中的异常模式

### 7. 像素级异常检测 (Pixel Anomaly Detection)
- 相邻像素差异统计分析
- 颜色分布熵值计算
- 异常像素比例统计
- 边缘和纹理异常检测

### 8. 文件结构分析 (File Structure Analysis)
- 文件不同区域的熵值分析
- 重复模式和隐藏数据检测
- 文件头、中部、尾部的异常检测
- 二进制模式分析

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r watermark_requirements.txt
```

或手动安装：
```bash
pip install opencv-python numpy pillow moviepy librosa scipy PyWavelets matplotlib ExifRead scikit-image
```

### 2. 基本使用

#### 命令行模式
```bash
# 分析单个视频文件
python watermark_detector.py your_video.mp4
```

#### 交互式演示
```bash
# 启动演示程序
python watermark_demo.py
```

### 3. 程序化使用
```python
from watermark_detector import WatermarkDetector

# 创建检测器
detector = WatermarkDetector()

# 分析视频
results = detector.analyze_video('video.mp4')

# 打印摘要
detector.print_summary(results)

# 保存详细报告
detector.save_results(results, 'analysis_report.json')
```

## 📊 检测结果解读

### 置信度等级
- **High (高)**: > 70% - 强烈怀疑存在水印
- **Medium (中)**: 40-70% - 可能存在水印
- **Low (低)**: 10-40% - 水印存在可能性较低
- **Minimal (极低)**: < 10% - 基本无水印特征

### 风险级别
- **High Risk**: 检测到多种水印特征，建议避免商业使用
- **Medium Risk**: 检测到部分水印特征，建议谨慎使用
- **Low Risk**: 少量可疑特征，可考虑使用
- **Minimal Risk**: 未检测到明显水印特征

### 检测类型说明
- **metadata_signatures**: 元数据中的平台签名
- **dct_frequency_watermark**: DCT频域水印
- **lsb_steganography**: LSB隐写术
- **audio_watermark**: 音频水印
- **temporal_patterns**: 时域模式水印

## 📁 输出文件

### 分析报告结构
```json
{
  "file_info": {
    "filename": "video.mp4",
    "size_mb": 15.3,
    "duration_seconds": 30.5,
    "resolution": "1080x1920"
  },
  "watermarks_detected": {
    "metadata": {...},
    "dct_watermark": {...},
    "lsb_analysis": {...},
    "audio": {...}
  },
  "confidence_scores": {
    "overall_confidence": 0.75,
    "metadata_confidence": 0.8,
    "dct_confidence": 0.6
  },
  "summary": {
    "suspected_platform": "tiktok",
    "risk_level": "high",
    "detected_watermark_types": [...],
    "recommendations": [...]
  }
}
```

## 🛠️ 高级功能

### 批量处理
```bash
# 分析目录中的所有视频
python watermark_demo.py
# 选择选项 2 进行批量分析
```

### 自定义配置
可以通过修改 `WatermarkDetector` 类中的参数来调整检测敏感度：

```python
detector = WatermarkDetector()
# 调整DCT检测阈值
detector.dct_threshold = 2.5  # 默认2.0
# 调整LSB检测敏感度
detector.lsb_threshold = 0.4  # 默认0.45
```

## 🔧 技术原理

### 水印检测原理
现代短视频平台为了版权保护和内容追踪，会在视频中嵌入多种类型的隐形水印：

1. **可见水印**: 透明Logo或文字覆盖
2. **不可见水印**: 嵌入在像素、频域或音频中
3. **鲁棒水印**: 抗压缩、抗转换的持久性标识
4. **脆弱水印**: 用于完整性验证的易损水印

### 检测挑战
- 水印通常具有不可感知性
- 需要在多个域进行综合分析
- 不同平台使用不同的嵌入算法
- 水印可能经过加密或编码

## 📈 性能优化

### 处理大文件
- 自动采样关键帧进行分析
- 内存使用监控和优化
- 分块处理避免内存溢出

### 检测精度
- 多算法融合提升准确率
- 置信度加权计算
- 误报率控制机制

## 🔍 使用场景

### 学术研究
- 数字水印技术研究
- 视频内容分析
- 隐写术检测算法验证

### 内容审核
- 平台来源识别
- 版权保护分析
- 内容完整性验证

### 技术分析
- 逆向工程研究
- 安全性评估
- 算法性能测试

## ⚡ 注意事项

### 检测限制
- 某些高级水印技术可能无法检测
- 检测结果具有一定的误差率
- 需要结合多种方法综合判断

### 技术要求
- Python 3.7+ 环境
- 充足的内存和计算资源
- 支持的视频格式有限

### 法律风险
- 请确保有权分析目标视频
- 遵守当地法律法规
- 不要用于侵犯他人权益

## 🤝 贡献和支持

### 反馈问题
如果发现检测不准确或程序错误，欢迎提供反馈。

### 技术改进
欢迎贡献新的检测算法或优化现有代码。

### 学术引用
如果在学术研究中使用此工具，请适当引用。

---

**再次提醒**: 本工具仅供技术研究使用，请遵守相关法律法规和道德准则！