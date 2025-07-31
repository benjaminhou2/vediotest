#!/usr/bin/env python3
# 视频相似度分析系统安装脚本

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}完成")
            return True
        else:
            print(f"❌ {description}失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}出错: {e}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("需要Python 3.7或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """安装依赖包"""
    print("\n📦 安装Python依赖包...")
    
    # 升级pip
    if not run_command("python -m pip install --upgrade pip", "升级pip"):
        return False
    
    # 安装依赖
    if not run_command("pip install -r requirements.txt", "安装依赖包"):
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建项目目录...")
    
    directories = ['temp', 'reports', 'examples']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ 创建目录: {directory}")
        except Exception as e:
            print(f"  ❌ 创建目录失败 {directory}: {e}")
            return False
    
    return True

def create_desktop_shortcut():
    """创建桌面快捷方式（仅Windows）"""
    if platform.system() != "Windows":
        return True
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "视频相似度分析.lnk")
        target = os.path.join(os.getcwd(), "gui.py")
        wDir = os.getcwd()
        icon = target
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{target}"'
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        
        print("✅ 桌面快捷方式创建成功")
        return True
        
    except ImportError:
        print("⚠️  跳过桌面快捷方式创建（需要pywin32包）")
        return True
    except Exception as e:
        print(f"⚠️  桌面快捷方式创建失败: {e}")
        return True

def main():
    """主安装流程"""
    print("🎬 视频相似度分析系统 - 安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies():
        print("\n❌ 依赖安装失败")
        sys.exit(1)
    
    # 创建目录
    if not create_directories():
        print("\n❌ 目录创建失败")
        sys.exit(1)
    
    # 创建快捷方式
    create_desktop_shortcut()
    
    print("\n" + "=" * 50)
    print("🎉 安装完成！")
    print("\n使用方法:")
    print("1. 图形界面: python gui.py")
    print("2. 命令行分析: python video_analyzer.py video1.mp4 video2.mp4")
    print("3. 批量处理: python batch_processor.py directory /path/to/videos")
    print("\n查看README.md获取更多信息")

if __name__ == "__main__":
    main()