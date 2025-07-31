# 图形用户界面模块
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
from datetime import datetime
import webbrowser

from video_analyzer import VideoAnalyzer
from config import config
from utils import format_duration, get_file_info, validate_video_file

class VideoSimilarityGUI:
    """视频相似度分析图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("视频相似度分析系统 v2.0")
        self.root.geometry(f"{config.gui.window_width}x{config.gui.window_height}")
        self.root.resizable(True, True)
        
        # 设置图标和样式
        self.setup_style()
        
        # 初始化变量
        self.video1_path = tk.StringVar()
        self.video2_path = tk.StringVar()
        self.analyzer = None
        self.current_analysis = None
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        
        # 设置主题
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='green', font=('Arial', 11, 'bold'))
        style.configure('Error.TLabel', foreground='red', font=('Arial', 11, 'bold'))
        style.configure('Warning.TLabel', foreground='orange', font=('Arial', 11, 'bold'))
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🎬 视频相似度分析系统", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 视频选择区域
        self.create_video_selection_area(main_frame)
        
        # 分析控制区域
        self.create_analysis_control_area(main_frame)
        
        # 进度显示区域
        self.create_progress_area(main_frame)
        
        # 结果显示区域
        self.create_results_area(main_frame)
        
        # 底部按钮区域
        self.create_bottom_buttons(main_frame)
        
    def create_video_selection_area(self, parent):
        """创建视频选择区域"""
        # 视频选择框架
        video_frame = ttk.LabelFrame(parent, text="📁 选择视频文件", padding="10")
        video_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(1, weight=1)
        
        # 视频1选择
        ttk.Label(video_frame, text="视频1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.video1_entry = ttk.Entry(video_frame, textvariable=self.video1_path, state='readonly')
        self.video1_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(video_frame, text="浏览", 
                  command=lambda: self.select_video(1)).grid(row=0, column=2)
        
        # 视频1信息
        self.video1_info_label = ttk.Label(video_frame, text="", style='Info.TLabel')
        self.video1_info_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # 视频2选择
        ttk.Label(video_frame, text="视频2:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.video2_entry = ttk.Entry(video_frame, textvariable=self.video2_path, state='readonly')
        self.video2_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(video_frame, text="浏览", 
                  command=lambda: self.select_video(2)).grid(row=2, column=2, pady=(10, 0))
        
        # 视频2信息
        self.video2_info_label = ttk.Label(video_frame, text="", style='Info.TLabel')
        self.video2_info_label.grid(row=3, column=1, sticky=tk.W, pady=(5, 0))
        
    def create_analysis_control_area(self, parent):
        """创建分析控制区域"""
        control_frame = ttk.LabelFrame(parent, text="⚙️ 分析控制", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # 分析按钮
        self.analyze_button = ttk.Button(control_frame, text="🚀 开始分析", 
                                       command=self.start_analysis, state='disabled')
        self.analyze_button.grid(row=0, column=0, padx=(0, 20))
        
        # 停止按钮
        self.stop_button = ttk.Button(control_frame, text="⏹️ 停止分析", 
                                    command=self.stop_analysis, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(0, 20))
        
        # 设置按钮
        ttk.Button(control_frame, text="⚙️ 设置", 
                  command=self.open_settings).grid(row=0, column=2)
        
    def create_progress_area(self, parent):
        """创建进度显示区域"""
        progress_frame = ttk.LabelFrame(parent, text="📊 分析进度", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 进度文本
        self.progress_label = ttk.Label(progress_frame, text="等待开始...")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)
        
    def create_results_area(self, parent):
        """创建结果显示区域"""
        results_frame = ttk.LabelFrame(parent, text="📈 分析结果", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # 结果摘要
        self.results_summary_frame = ttk.Frame(results_frame)
        self.results_summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.results_summary_frame.columnconfigure(1, weight=1)
        
        # 综合相似度显示
        ttk.Label(self.results_summary_frame, text="综合相似度:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.similarity_label = ttk.Label(self.results_summary_frame, text="--", style='Header.TLabel')
        self.similarity_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 判断结果显示
        ttk.Label(self.results_summary_frame, text="判断结果:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.judgment_label = ttk.Label(self.results_summary_frame, text="--", style='Header.TLabel')
        self.judgment_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 详细结果文本框
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, wrap=tk.WORD)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_bottom_buttons(self, parent):
        """创建底部按钮区域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 保存报告按钮
        self.save_report_button = ttk.Button(button_frame, text="💾 保存报告", 
                                           command=self.save_report, state='disabled')
        self.save_report_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 打开报告目录按钮
        ttk.Button(button_frame, text="📂 打开报告目录", 
                  command=self.open_reports_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        # 关于按钮
        ttk.Button(button_frame, text="ℹ️ 关于", 
                  command=self.show_about).pack(side=tk.RIGHT)
        
        # 帮助按钮
        ttk.Button(button_frame, text="❓ 帮助", 
                  command=self.show_help).pack(side=tk.RIGHT, padx=(0, 10))
        
    def select_video(self, video_num):
        """选择视频文件"""
        filetypes = [
            ('视频文件', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv'),
            ('所有文件', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title=f"选择视频{video_num}",
            filetypes=filetypes
        )
        
        if filename:
            if video_num == 1:
                self.video1_path.set(filename)
                self.update_video_info(1, filename)
            else:
                self.video2_path.set(filename)
                self.update_video_info(2, filename)
            
            self.update_analyze_button()
    
    def update_video_info(self, video_num, filepath):
        """更新视频信息显示"""
        try:
            valid, error = validate_video_file(filepath, config.files.max_file_size_mb)
            if valid:
                info = get_file_info(filepath)
                info_text = f"✅ {info['size_mb']:.1f}MB | {info['modified']}"
                style = 'Info.TLabel'
            else:
                info_text = f"❌ {error}"
                style = 'Error.TLabel'
            
            if video_num == 1:
                self.video1_info_label.config(text=info_text, style=style)
            else:
                self.video2_info_label.config(text=info_text, style=style)
                
        except Exception as e:
            error_text = f"❌ 信息获取失败: {str(e)}"
            if video_num == 1:
                self.video1_info_label.config(text=error_text, style='Error.TLabel')
            else:
                self.video2_info_label.config(text=error_text, style='Error.TLabel')
    
    def update_analyze_button(self):
        """更新分析按钮状态"""
        if self.video1_path.get() and self.video2_path.get():
            # 检查两个视频文件是否都有效
            valid1, _ = validate_video_file(self.video1_path.get(), config.files.max_file_size_mb)
            valid2, _ = validate_video_file(self.video2_path.get(), config.files.max_file_size_mb)
            
            if valid1 and valid2:
                self.analyze_button.config(state='normal')
            else:
                self.analyze_button.config(state='disabled')
        else:
            self.analyze_button.config(state='disabled')
    
    def start_analysis(self):
        """开始分析"""
        if not self.video1_path.get() or not self.video2_path.get():
            messagebox.showerror("错误", "请先选择两个视频文件")
            return
        
        # 重置界面状态
        self.analyze_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_report_button.config(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text="准备开始分析...")
        self.results_text.delete(1.0, tk.END)
        self.similarity_label.config(text="--")
        self.judgment_label.config(text="--")
        
        # 在新线程中执行分析
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def run_analysis(self):
        """在后台线程中运行分析"""
        try:
            # 创建分析器
            if self.analyzer is None:
                self.update_progress("加载分析模型...", 5)
                self.analyzer = VideoAnalyzer()
            
            # 执行分析
            def progress_callback(message, percentage):
                self.root.after(0, lambda: self.update_progress(message, percentage))
            
            self.current_analysis = self.analyzer.analyze_video_similarity(
                self.video1_path.get(),
                self.video2_path.get(),
                progress_callback
            )
            
            # 在主线程中更新界面
            self.root.after(0, self.analysis_completed)
            
        except Exception as e:
            error_msg = f"分析失败: {str(e)}"
            self.root.after(0, lambda: self.analysis_failed(error_msg))
    
    def update_progress(self, message, percentage):
        """更新进度显示"""
        self.progress_var.set(percentage)
        self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def analysis_completed(self):
        """分析完成后的处理"""
        self.analyze_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.save_report_button.config(state='normal')
        
        if 'error' in self.current_analysis:
            self.analysis_failed(self.current_analysis['error'])
            return
        
        # 显示结果
        self.display_results(self.current_analysis)
        
        # 更新进度
        self.update_progress("分析完成!", 100)
        
        # 播放完成提示音（如果系统支持）
        try:
            self.root.bell()
        except:
            pass
    
    def analysis_failed(self, error_msg):
        """分析失败后的处理"""
        self.analyze_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.progress_label.config(text=f"分析失败: {error_msg}")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ 分析失败:\n{error_msg}")
        
        messagebox.showerror("分析失败", error_msg)
    
    def display_results(self, result):
        """显示分析结果"""
        # 更新摘要显示
        similarity_percent = result.get('overall_similarity_percent', 0)
        self.similarity_label.config(text=f"{similarity_percent:.1f}%")
        
        is_similar = result.get('is_similar', False)
        if is_similar:
            self.judgment_label.config(text="✅ 相似", style='Success.TLabel')
        else:
            self.judgment_label.config(text="❌ 不相似", style='Error.TLabel')
        
        # 显示详细结果
        self.results_text.delete(1.0, tk.END)
        
        # 基本信息
        self.results_text.insert(tk.END, "📊 视频相似度分析报告\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # 视频信息
        self.results_text.insert(tk.END, f"🎬 视频1: {result['video1_info']['name']}\n")
        self.results_text.insert(tk.END, f"   大小: {result['video1_info']['size_mb']:.1f}MB\n")
        self.results_text.insert(tk.END, f"   修改时间: {result['video1_info']['modified']}\n\n")
        
        self.results_text.insert(tk.END, f"🎬 视频2: {result['video2_info']['name']}\n")
        self.results_text.insert(tk.END, f"   大小: {result['video2_info']['size_mb']:.1f}MB\n")
        self.results_text.insert(tk.END, f"   修改时间: {result['video2_info']['modified']}\n\n")
        
        # 相似度指标
        self.results_text.insert(tk.END, "📈 相似度指标:\n")
        self.results_text.insert(tk.END, f"  • 视觉相似度 (pHash): {result['phash_similarity_percent']:.1f}%\n")
        self.results_text.insert(tk.END, f"  • 语义相似度 (CNN):   {result['cnn_similarity_percent']:.1f}%\n")
        self.results_text.insert(tk.END, f"  • 音频相似度:         {result['audio_similarity_percent']:.1f}%\n")
        self.results_text.insert(tk.END, f"  • 综合相似度:         {result['overall_similarity_percent']:.1f}%\n\n")
        
        # 原始距离值
        self.results_text.insert(tk.END, "🔢 原始距离值:\n")
        self.results_text.insert(tk.END, f"  • pHash距离:     {result['pHash_distance']:.4f}\n")
        self.results_text.insert(tk.END, f"  • CNN余弦距离:   {result['cnn_cosine_distance']:.4f}\n")
        self.results_text.insert(tk.END, f"  • 音频差异比例: {result['audio_difference_ratio']:.4f}\n\n")
        
        # 分析信息
        self.results_text.insert(tk.END, "ℹ️  分析信息:\n")
        self.results_text.insert(tk.END, f"  • 分析时间: {result['analysis_time']}\n")
        self.results_text.insert(tk.END, f"  • 处理时长: {format_duration(result['processing_time_seconds'])}\n")
        self.results_text.insert(tk.END, f"  • 帧间隔: {result['config_used']['frame_interval']}秒\n")
        self.results_text.insert(tk.END, f"  • 哈希大小: {result['config_used']['hash_size']}\n")
        
        # 滚动到顶部
        self.results_text.see(1.0)
    
    def stop_analysis(self):
        """停止分析"""
        # 注意：这里的停止功能有限，主要是UI状态重置
        self.analyze_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_label.config(text="分析已停止")
        
        messagebox.showinfo("停止", "分析已停止")
    
    def save_report(self):
        """保存分析报告"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showerror("错误", "没有有效的分析结果可以保存")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"similarity_report_{timestamp}.json"
        
        filename = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".json",
            initialfilename=default_filename,
            filetypes=[
                ('JSON文件', '*.json'),
                ('所有文件', '*.*')
            ]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_analysis, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("保存成功", f"报告已保存到:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("保存失败", f"保存报告时出错:\n{str(e)}")
    
    def open_reports_folder(self):
        """打开报告目录"""
        reports_dir = config.files.output_dir
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(reports_dir)
            elif os.name == 'posix':  # macOS and Linux
                import sys
                os.system(f'open "{reports_dir}"' if sys.platform == 'darwin' else f'xdg-open "{reports_dir}"')
        except Exception as e:
            messagebox.showerror("打开失败", f"无法打开目录:\n{str(e)}")
    
    def open_settings(self):
        """打开设置窗口"""
        settings_window = SettingsWindow(self.root, config)
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
🎬 视频相似度分析系统 - 使用帮助

📋 功能介绍:
• 本系统使用三种技术分析视频相似度:
  - 感知哈希(pHash): 比较视觉外观
  - CNN深度学习: 分析语义内容  
  - 音频频谱: 比较音轨特征

🚀 使用步骤:
1. 点击"浏览"按钮选择两个视频文件
2. 确认视频信息显示正确
3. 点击"开始分析"执行比较
4. 查看分析结果和相似度报告
5. 可选择保存详细报告

⚙️ 支持格式:
MP4, AVI, MOV, MKV, WMV, FLV

📊 结果解读:
• 相似度百分比越高表示越相似
• 综合相似度 > 60% 通常判定为相似
• 可在设置中调整判定阈值

❗ 注意事项:
• 建议视频文件小于500MB
• 分析时间取决于视频长度和质量
• 第一次运行需要下载AI模型
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("使用帮助")
        help_window.geometry("600x500")
        help_window.resizable(False, False)
        
        # 居中显示
        help_window.transient(self.root)
        help_window.grab_set()
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=20, pady=20)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
🎬 视频相似度分析系统 v2.0

🔬 技术特性:
• 多维度相似度检测
• 多线程并行处理
• 智能阈值判断
• 详细分析报告

💻 技术栈:
• Python + TensorFlow
• OpenCV + MoviePy
• ResNet50 深度学习模型

👨‍💻 开发信息:
基于先进的计算机视觉和音频处理技术
适用于视频去重、版权检测等场景

📧 支持与反馈:
如有问题或建议，欢迎联系开发团队
        """
        
        messagebox.showinfo("关于", about_text)
    
    def on_closing(self):
        """程序退出处理"""
        if messagebox.askokcancel("退出", "确定要退出视频相似度分析系统吗？"):
            self.root.destroy()
    
    def run(self):
        """启动GUI"""
        self.root.mainloop()

class SettingsWindow:
    """设置窗口"""
    
    def __init__(self, parent, config_manager):
        self.parent = parent
        self.config = config_manager
        
        self.window = tk.Toplevel(parent)
        self.window.title("设置")
        self.window.geometry("500x400")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建设置界面"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 处理设置页
        processing_frame = ttk.Frame(notebook)
        notebook.add(processing_frame, text="处理设置")
        
        # 阈值设置页
        threshold_frame = ttk.Frame(notebook)
        notebook.add(threshold_frame, text="阈值设置")
        
        # 文件设置页
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="文件设置")
        
        # 创建各个设置页面
        self.create_processing_settings(processing_frame)
        self.create_threshold_settings(threshold_frame)
        self.create_file_settings(file_frame)
        
        # 底部按钮
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="保存", command=self.save_settings).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="取消", command=self.window.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="恢复默认", command=self.reset_defaults).pack(side=tk.LEFT)
    
    def create_processing_settings(self, parent):
        """创建处理设置页面"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 帧间隔设置
        ttk.Label(frame, text="帧抽取间隔 (秒):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.frame_interval_var = tk.IntVar(value=self.config.processing.frame_interval)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.frame_interval_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 哈希大小设置
        ttk.Label(frame, text="pHash尺寸:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hash_size_var = tk.IntVar(value=self.config.processing.hash_size)
        ttk.Spinbox(frame, from_=4, to=16, textvariable=self.hash_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 工作线程数
        ttk.Label(frame, text="并行线程数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_workers_var = tk.IntVar(value=self.config.processing.max_workers)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.max_workers_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
    
    def create_threshold_settings(self, parent):
        """创建阈值设置页面"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # pHash阈值
        ttk.Label(frame, text="pHash阈值:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.phash_threshold_var = tk.DoubleVar(value=self.config.thresholds.phash_threshold)
        ttk.Entry(frame, textvariable=self.phash_threshold_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # CNN阈值
        ttk.Label(frame, text="CNN阈值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.cnn_threshold_var = tk.DoubleVar(value=self.config.thresholds.cnn_threshold)
        ttk.Entry(frame, textvariable=self.cnn_threshold_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 音频阈值
        ttk.Label(frame, text="音频阈值:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.audio_threshold_var = tk.DoubleVar(value=self.config.thresholds.audio_threshold)
        ttk.Entry(frame, textvariable=self.audio_threshold_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # 综合阈值
        ttk.Label(frame, text="综合阈值:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.overall_threshold_var = tk.DoubleVar(value=self.config.thresholds.overall_threshold)
        ttk.Entry(frame, textvariable=self.overall_threshold_var, width=15).grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
    
    def create_file_settings(self, parent):
        """创建文件设置页面"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 最大文件大小
        ttk.Label(frame, text="最大文件大小 (MB):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_file_size_var = tk.IntVar(value=self.config.files.max_file_size_mb)
        ttk.Entry(frame, textvariable=self.max_file_size_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 临时目录
        ttk.Label(frame, text="临时目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temp_dir_var = tk.StringVar(value=self.config.files.temp_dir)
        ttk.Entry(frame, textvariable=self.temp_dir_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 输出目录
        ttk.Label(frame, text="报告目录:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value=self.config.files.output_dir)
        ttk.Entry(frame, textvariable=self.output_dir_var, width=30).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
    
    def save_settings(self):
        """保存设置"""
        try:
            # 更新配置
            self.config.processing.frame_interval = self.frame_interval_var.get()
            self.config.processing.hash_size = self.hash_size_var.get()
            self.config.processing.max_workers = self.max_workers_var.get()
            
            self.config.thresholds.phash_threshold = self.phash_threshold_var.get()
            self.config.thresholds.cnn_threshold = self.cnn_threshold_var.get()
            self.config.thresholds.audio_threshold = self.audio_threshold_var.get()
            self.config.thresholds.overall_threshold = self.overall_threshold_var.get()
            
            self.config.files.max_file_size_mb = self.max_file_size_var.get()
            self.config.files.temp_dir = self.temp_dir_var.get()
            self.config.files.output_dir = self.output_dir_var.get()
            
            # 保存到文件
            self.config.save_config()
            
            messagebox.showinfo("保存成功", "设置已保存")
            self.window.destroy()
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存设置时出错:\n{str(e)}")
    
    def reset_defaults(self):
        """恢复默认设置"""
        if messagebox.askyesno("确认", "确定要恢复所有默认设置吗？"):
            self.frame_interval_var.set(1)
            self.hash_size_var.set(8)
            self.max_workers_var.set(4)
            
            self.phash_threshold_var.set(15.0)
            self.cnn_threshold_var.set(0.3)
            self.audio_threshold_var.set(0.4)
            self.overall_threshold_var.set(0.6)
            
            self.max_file_size_var.set(500)
            self.temp_dir_var.set('./temp')
            self.output_dir_var.set('./reports')

def main():
    """启动GUI应用"""
    try:
        app = VideoSimilarityGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("启动失败", f"程序启动失败:\n{str(e)}")

if __name__ == '__main__':
    main()