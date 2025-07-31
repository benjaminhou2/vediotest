# 批量处理器模块
import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import traceback

from video_analyzer import VideoAnalyzer
from config import config
from utils import (
    validate_video_file, get_file_info, format_duration,
    ProgressTracker, calculate_overall_similarity, is_similar,
    clean_temp_files, ensure_directory
)

class BatchProcessor:
    """批量视频处理器"""
    
    def __init__(self):
        self.config = config
        self.analyzer = None
        self.results = []
        
    def initialize_analyzer(self):
        """初始化分析器"""
        if self.analyzer is None:
            print("🔄 初始化视频分析器...")
            self.analyzer = VideoAnalyzer()
            print("✅ 分析器初始化完成")
    
    def find_video_files(self, directory: str, recursive: bool = True) -> list:
        """
        在目录中查找视频文件
        
        Args:
            directory: 搜索目录
            recursive: 是否递归搜索子目录
        
        Returns:
            视频文件路径列表
        """
        video_files = []
        search_path = Path(directory)
        
        if not search_path.exists():
            raise ValueError(f"目录不存在: {directory}")
        
        # 支持的视频格式
        video_extensions = set(self.config.files.supported_formats)
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in search_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                # 验证文件
                valid, _ = validate_video_file(str(file_path), self.config.files.max_file_size_mb)
                if valid:
                    video_files.append(str(file_path))
                else:
                    print(f"⚠️  跳过无效文件: {file_path.name}")
        
        return sorted(video_files)
    
    def compare_file_pair(self, video_pair: tuple) -> dict:
        """
        比较一对视频文件
        
        Args:
            video_pair: (video1_path, video2_path)
        
        Returns:
            比较结果字典
        """
        video1_path, video2_path = video_pair
        
        try:
            result = self.analyzer.analyze_video_similarity(video1_path, video2_path)
            
            # 添加额外信息
            result.update({
                'video1_file': os.path.basename(video1_path),
                'video2_file': os.path.basename(video2_path),
                'video1_full_path': video1_path,
                'video2_full_path': video2_path,
                'comparison_id': f"{os.path.basename(video1_path)}_vs_{os.path.basename(video2_path)}"
            })
            
            return result
            
        except Exception as e:
            error_result = {
                'video1_file': os.path.basename(video1_path),
                'video2_file': os.path.basename(video2_path),
                'video1_full_path': video1_path,
                'video2_full_path': video2_path,
                'comparison_id': f"{os.path.basename(video1_path)}_vs_{os.path.basename(video2_path)}",
                'error': str(e),
                'traceback': traceback.format_exc(),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"❌ 比较失败 {os.path.basename(video1_path)} vs {os.path.basename(video2_path)}: {e}")
            return error_result
    
    def batch_compare_directory(self, directory: str, recursive: bool = True, 
                              max_workers: int = None, output_file: str = None) -> list:
        """
        批量比较目录中的所有视频
        
        Args:
            directory: 视频目录
            recursive: 是否递归搜索
            max_workers: 最大并行工作数
            output_file: 输出文件路径
        
        Returns:
            比较结果列表
        """
        print(f"🔍 搜索目录: {directory}")
        
        # 查找视频文件
        video_files = self.find_video_files(directory, recursive)
        
        if len(video_files) < 2:
            print(f"⚠️  目录中找到的视频文件不足2个 ({len(video_files)}个)")
            return []
        
        print(f"📁 找到 {len(video_files)} 个视频文件")
        
        # 生成所有可能的配对
        video_pairs = list(combinations(video_files, 2))
        total_comparisons = len(video_pairs)
        
        print(f"🔄 需要进行 {total_comparisons} 次比较")
        
        if total_comparisons > 100:
            response = input(f"⚠️  将进行 {total_comparisons} 次比较，可能需要很长时间。是否继续？ (y/n): ")
            if response.lower() != 'y':
                print("❌ 用户取消操作")
                return []
        
        # 初始化分析器
        self.initialize_analyzer()
        
        # 设置并行工作数
        if max_workers is None:
            max_workers = min(self.config.processing.max_workers, total_comparisons)
        
        print(f"⚙️  使用 {max_workers} 个并行工作进程")
        
        # 执行批量比较
        results = []
        start_time = time.time()
        progress = ProgressTracker(total_comparisons, "批量比较进度")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(self.compare_file_pair, pair): pair 
                      for pair in video_pairs}
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新进度
                    pair = futures[future]
                    message = f"完成: {os.path.basename(pair[0])} vs {os.path.basename(pair[1])}"
                    progress.update(1, message)
                    
                except Exception as e:
                    print(f"❌ 任务执行失败: {e}")
                    progress.update(1, "失败")
        
        progress.finish("批量比较完成")
        
        total_time = time.time() - start_time
        print(f"⏱️  总用时: {format_duration(total_time)}")
        print(f"📊 平均每次比较: {format_duration(total_time / total_comparisons)}")
        
        # 保存结果
        if output_file:
            self.save_results(results, output_file)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_output = os.path.join(self.config.files.output_dir, 
                                        f"batch_comparison_{timestamp}.json")
            self.save_results(results, default_output)
        
        # 生成统计报告
        self.generate_summary_report(results, directory)
        
        return results
    
    def batch_compare_file_list(self, file_list: list, output_file: str = None) -> list:
        """
        批量比较指定的文件列表
        
        Args:
            file_list: 视频文件路径列表
            output_file: 输出文件路径
        
        Returns:
            比较结果列表
        """
        # 验证文件
        valid_files = []
        for file_path in file_list:
            valid, error = validate_video_file(file_path, self.config.files.max_file_size_mb)
            if valid:
                valid_files.append(file_path)
            else:
                print(f"⚠️  跳过无效文件 {file_path}: {error}")
        
        if len(valid_files) < 2:
            print(f"⚠️  有效视频文件不足2个 ({len(valid_files)}个)")
            return []
        
        print(f"📁 处理 {len(valid_files)} 个有效视频文件")
        
        # 生成配对并执行比较
        video_pairs = list(combinations(valid_files, 2))
        
        # 初始化分析器
        self.initialize_analyzer()
        
        results = []
        progress = ProgressTracker(len(video_pairs), "批量比较进度")
        
        for pair in video_pairs:
            result = self.compare_file_pair(pair)
            results.append(result)
            
            progress.update(1, f"完成: {os.path.basename(pair[0])} vs {os.path.basename(pair[1])}")
        
        progress.finish("批量比较完成")
        
        # 保存结果
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: list, output_file: str):
        """保存结果到文件"""
        ensure_directory(os.path.dirname(output_file))
        
        file_ext = Path(output_file).suffix.lower()
        
        try:
            if file_ext == '.json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            elif file_ext == '.csv':
                self.save_results_csv(results, output_file)
            
            else:
                # 默认保存为JSON
                output_file = output_file + '.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"💾 结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def save_results_csv(self, results: list, output_file: str):
        """保存结果为CSV格式"""
        if not results:
            return
        
        # 定义CSV列
        csv_columns = [
            'video1_file', 'video2_file', 'comparison_id',
            'overall_similarity_percent', 'is_similar',
            'phash_similarity_percent', 'cnn_similarity_percent', 'audio_similarity_percent',
            'pHash_distance', 'cnn_cosine_distance', 'audio_difference_ratio',
            'processing_time_seconds', 'analysis_time'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            
            for result in results:
                # 只写入有效结果
                if 'error' not in result:
                    row_data = {col: result.get(col, '') for col in csv_columns}
                    writer.writerow(row_data)
    
    def generate_summary_report(self, results: list, source_directory: str = ""):
        """生成汇总报告"""
        if not results:
            return
        
        # 统计信息
        total_comparisons = len(results)
        successful_comparisons = len([r for r in results if 'error' not in r])
        failed_comparisons = total_comparisons - successful_comparisons
        
        # 相似度统计
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            similar_pairs = len([r for r in successful_results if r.get('is_similar', False)])
            
            # 相似度分布
            similarities = [r.get('overall_similarity_percent', 0) for r in successful_results]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            max_similarity = max(similarities) if similarities else 0
            min_similarity = min(similarities) if similarities else 0
            
            # 找出最相似的几对
            top_similar = sorted(successful_results, 
                               key=lambda x: x.get('overall_similarity_percent', 0), 
                               reverse=True)[:5]
        else:
            similar_pairs = 0
            avg_similarity = max_similarity = min_similarity = 0
            top_similar = []
        
        # 生成报告
        print("\n" + "="*60)
        print("📊 批量比较汇总报告")
        print("="*60)
        
        if source_directory:
            print(f"📁 源目录: {source_directory}")
        
        print(f"📈 比较统计:")
        print(f"  • 总比较次数: {total_comparisons}")
        print(f"  • 成功比较: {successful_comparisons}")
        print(f"  • 失败比较: {failed_comparisons}")
        print(f"  • 相似对数: {similar_pairs}")
        print(f"  • 相似比例: {(similar_pairs/successful_comparisons*100):.1f}%" if successful_comparisons > 0 else "  • 相似比例: 0%")
        
        if successful_results:
            print(f"\n📊 相似度分布:")
            print(f"  • 平均相似度: {avg_similarity:.1f}%")
            print(f"  • 最高相似度: {max_similarity:.1f}%")
            print(f"  • 最低相似度: {min_similarity:.1f}%")
            
            if top_similar:
                print(f"\n🏆 最相似的视频对:")
                for i, result in enumerate(top_similar[:3], 1):
                    print(f"  {i}. {result['video1_file']} vs {result['video2_file']}")
                    print(f"     相似度: {result['overall_similarity_percent']:.1f}%")
        
        # 保存汇总报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(self.config.files.output_dir, 
                                  f"batch_summary_{timestamp}.txt")
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("批量视频相似度比较汇总报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if source_directory:
                    f.write(f"源目录: {source_directory}\n")
                f.write(f"\n比较统计:\n")
                f.write(f"总比较次数: {total_comparisons}\n")
                f.write(f"成功比较: {successful_comparisons}\n")
                f.write(f"失败比较: {failed_comparisons}\n")
                f.write(f"相似对数: {similar_pairs}\n")
                
                if successful_results:
                    f.write(f"\n相似度分布:\n")
                    f.write(f"平均相似度: {avg_similarity:.1f}%\n")
                    f.write(f"最高相似度: {max_similarity:.1f}%\n")
                    f.write(f"最低相似度: {min_similarity:.1f}%\n")
                    
                    if top_similar:
                        f.write(f"\n最相似的视频对:\n")
                        for i, result in enumerate(top_similar, 1):
                            f.write(f"{i}. {result['video1_file']} vs {result['video2_file']}\n")
                            f.write(f"   相似度: {result['overall_similarity_percent']:.1f}%\n")
            
            print(f"\n📄 汇总报告已保存: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  汇总报告保存失败: {e}")
    
    def find_duplicate_groups(self, results: list, similarity_threshold: float = None) -> list:
        """
        从比较结果中找出重复视频组
        
        Args:
            results: 比较结果列表
            similarity_threshold: 相似度阈值
        
        Returns:
            重复组列表
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.thresholds.overall_threshold * 100
        
        # 构建图结构
        video_graph = {}
        similar_pairs = []
        
        for result in results:
            if 'error' in result:
                continue
            
            video1 = result['video1_full_path']
            video2 = result['video2_full_path']
            similarity = result.get('overall_similarity_percent', 0)
            
            # 记录相似的配对
            if similarity >= similarity_threshold:
                similar_pairs.append((video1, video2, similarity))
                
                # 建立图连接
                if video1 not in video_graph:
                    video_graph[video1] = set()
                if video2 not in video_graph:
                    video_graph[video2] = set()
                
                video_graph[video1].add(video2)
                video_graph[video2].add(video1)
        
        # 找出连通分量（重复组）
        visited = set()
        duplicate_groups = []
        
        def dfs(video, group):
            if video in visited:
                return
            visited.add(video)
            group.append(video)
            
            for neighbor in video_graph.get(video, []):
                dfs(neighbor, group)
        
        for video in video_graph:
            if video not in visited:
                group = []
                dfs(video, group)
                if len(group) > 1:
                    duplicate_groups.append(sorted(group))
        
        return duplicate_groups
    
    def print_duplicate_groups(self, results: list):
        """打印重复视频组"""
        duplicate_groups = self.find_duplicate_groups(results)
        
        if not duplicate_groups:
            print("✅ 未发现重复视频")
            return
        
        print(f"\n🔍 发现 {len(duplicate_groups)} 个重复视频组:")
        print("-" * 50)
        
        for i, group in enumerate(duplicate_groups, 1):
            print(f"\n重复组 {i} ({len(group)} 个文件):")
            for video in group:
                filename = os.path.basename(video)
                print(f"  • {filename}")
        
        # 保存重复组信息
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        duplicates_file = os.path.join(self.config.files.output_dir, 
                                     f"duplicate_groups_{timestamp}.json")
        
        try:
            duplicate_data = {
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'threshold_used': self.config.thresholds.overall_threshold * 100,
                'total_groups': len(duplicate_groups),
                'groups': [
                    {
                        'group_id': i,
                        'file_count': len(group),
                        'files': [
                            {
                                'filename': os.path.basename(video),
                                'full_path': video
                            } for video in group
                        ]
                    } for i, group in enumerate(duplicate_groups, 1)
                ]
            }
            
            with open(duplicates_file, 'w', encoding='utf-8') as f:
                json.dump(duplicate_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 重复组信息已保存: {duplicates_file}")
            
        except Exception as e:
            print(f"⚠️  重复组信息保存失败: {e}")

def main():
    """命令行主程序"""
    parser = argparse.ArgumentParser(description='批量视频相似度分析工具')
    
    subparsers = parser.add_subparsers(dest='command', help='操作命令')
    
    # 目录比较命令
    dir_parser = subparsers.add_parser('directory', help='比较目录中的所有视频')
    dir_parser.add_argument('directory', help='视频目录路径')
    dir_parser.add_argument('--recursive', '-r', action='store_true', help='递归搜索子目录')
    dir_parser.add_argument('--output', '-o', help='输出文件路径')
    dir_parser.add_argument('--workers', '-w', type=int, help='并行工作数')
    dir_parser.add_argument('--find-duplicates', '-d', action='store_true', help='查找重复视频组')
    
    # 文件列表比较命令
    files_parser = subparsers.add_parser('files', help='比较指定的视频文件列表')
    files_parser.add_argument('files', nargs='+', help='视频文件路径列表')
    files_parser.add_argument('--output', '-o', help='输出文件路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 清理临时文件
    clean_temp_files(config.files.temp_dir)
    
    processor = BatchProcessor()
    
    try:
        if args.command == 'directory':
            print(f"🎬 批量比较目录: {args.directory}")
            
            results = processor.batch_compare_directory(
                args.directory,
                recursive=args.recursive,
                max_workers=args.workers,
                output_file=args.output
            )
            
            if args.find_duplicates and results:
                processor.print_duplicate_groups(results)
        
        elif args.command == 'files':
            print(f"🎬 批量比较 {len(args.files)} 个文件")
            
            results = processor.batch_compare_file_list(
                args.files,
                output_file=args.output
            )
        
        print("\n✅ 批量处理完成")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 批量处理失败: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()