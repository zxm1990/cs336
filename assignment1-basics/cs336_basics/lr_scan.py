# lr_scan.py
"""
学习率超参数扫描脚本
实现两阶段扫描策略：快速筛选 + 精细调优
"""

import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def run_training_experiment(
    train_data: str,
    valid_data: str,
    max_lr: float,
    min_lr: float,
    experiment_name: str,
    max_iterations: int = 1500,
    eval_interval: int = 300,
    eval_batches: int = 30,
    warmup_iters: int = 300,
    **kwargs
) -> Dict:
    """
    运行单个训练实验
    
    Returns:
        dict: 包含实验结果的字典
    """
    cmd = [
        "uv", "run", "python", "-m", "cs336_basics.train_loop",
        "--train-data", train_data,
        "--valid-data", valid_data,
        "--max-lr", str(max_lr),
        "--min-lr", str(min_lr),
        "--max-iterations", str(max_iterations),
        "--warmup-iters", str(warmup_iters),
        "--eval-interval", str(eval_interval),
        "--experiment-name", experiment_name,
    ]
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"max_lr={max_lr:.2e}, min_lr={min_lr:.2e}")
    print(f"max_iterations={max_iterations}, eval_interval={eval_interval}")
    print(f"\n执行命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    output_lines = []
    final_loss = None
    diverged = False
    
    try:
        # 实时输出，确保训练真正启动
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时打印输出并收集
        print("训练开始，实时输出：")
        print("-" * 60)
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip()
            output_lines.append(line)
            print(line)  # 实时显示
            
            # 实时解析关键信息
            if "Valid Loss:" in line:
                try:
                    parts = line.split("Valid Loss:")
                    if len(parts) > 1:
                        loss_str = parts[1].split()[0]
                        final_loss = float(loss_str)
                except (ValueError, IndexError):
                    pass
            if "❌" in line or "diverged" in line.lower() or "Training diverged" in line:
                diverged = True
        
        # 等待进程完成
        returncode = process.wait()
        elapsed_time = time.time() - start_time
        
        output = '\n'.join(output_lines)
        
        # 检查是否真的运行了训练
        if not output or len(output) < 100:
            print(f"\n⚠️  警告：训练输出异常短，可能没有正确启动")
            print(f"返回码: {returncode}")
            print(f"输出前200字符:\n{output[:200] if output else 'None'}")
        
        # 如果之前没找到最终损失，重新查找
        if final_loss is None:
            # 查找最终验证损失（查找最后一个Valid Loss）
            lines = output.split('\n')
            for line in lines:
                if "Valid Loss:" in line:
                    try:
                        parts = line.split("Valid Loss:")
                        if len(parts) > 1:
                            loss_str = parts[1].split()[0]
                            final_loss = float(loss_str)
                    except (ValueError, IndexError):
                        pass
                if "❌" in line or "diverged" in line.lower() or "Training diverged" in line:
                    diverged = True
            
            # 如果没找到最终损失，尝试从训练完成信息中获取
            if final_loss is None:
                # 查找最后的迭代信息
                for line in reversed(lines):
                    if "iterations:" in line and "Loss:" in line:
                        try:
                            parts = line.split("Loss:")
                            if len(parts) > 1:
                                loss_str = parts[1].split()[0]
                                final_loss = float(loss_str)
                                break
                        except (ValueError, IndexError):
                            pass
        
        print("\n" + "-" * 60)
        print(f"实验完成: {experiment_name}")
        print(f"  返回码: {returncode}")
        print(f"  最终验证损失: {final_loss if final_loss else 'N/A'}")
        print(f"  是否发散: {diverged}")
        print(f"  耗时: {elapsed_time:.1f}秒")
        
        return {
            "experiment_name": experiment_name,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "final_valid_loss": final_loss,
            "diverged": diverged,
            "elapsed_time": elapsed_time,
            "success": returncode == 0,
        }
    except Exception as e:
        return {
            "experiment_name": experiment_name,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "final_valid_loss": None,
            "diverged": True,
            "elapsed_time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }


def phase1_coarse_search(
    train_data: str,
    valid_data: str,
    lr_candidates: List[float],
    max_iterations: int = 1500,
    min_lr_ratio: float = 0.1,
    **kwargs
) -> List[Dict]:
    """
    阶段1：快速粗搜索
    
    Args:
        lr_candidates: 学习率候选列表
        max_iterations: 快速筛选的迭代步数（较短）
        min_lr_ratio: min_lr = max_lr * min_lr_ratio
    
    Returns:
        实验结果列表
    """
    results = []
    
    for i, max_lr in enumerate(lr_candidates):
        min_lr = max_lr * min_lr_ratio
        experiment_name = f"lr_scan_phase1_max{max_lr:.2e}_min{min_lr:.2e}"
        
        result = run_training_experiment(
            train_data=train_data,
            valid_data=valid_data,
            max_lr=max_lr,
            min_lr=min_lr,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
            eval_interval=300,
            eval_batches=30,
            warmup_iters=300,
            **kwargs
        )
        
        results.append(result)
        
        print(f"\n实验结果 {i+1}/{len(lr_candidates)}:")
        print(f"  max_lr: {max_lr:.2e}")
        loss_str = f"{result['final_valid_loss']:.4f}" if result['final_valid_loss'] is not None else 'N/A'
        print(f"  最终验证损失: {loss_str}")
        print(f"  是否发散: {result['diverged']}")
        print(f"  耗时: {result['elapsed_time']:.1f}秒")
    
    return results


def phase2_fine_search(
    train_data: str,
    valid_data: str,
    best_lr_range: Tuple[float, float],
    num_samples: int = 5,
    max_iterations: int = 5000,
    min_lr_ratio: float = 0.1,
    **kwargs
) -> List[Dict]:
    """
    阶段2：精细搜索（围绕阶段1的最佳结果）
    
    Args:
        best_lr_range: (min_lr, max_lr) 在阶段1中找到的最佳学习率范围
        num_samples: 精细搜索的采样数量
    """
    min_lr_candidate, max_lr_candidate = best_lr_range
    
    # 在最佳范围附近采样
    step = (max_lr_candidate - min_lr_candidate) / (num_samples - 1) if num_samples > 1 else 0
    lr_candidates = [min_lr_candidate + i * step for i in range(num_samples)]
    
    results = []
    
    for i, max_lr in enumerate(lr_candidates):
        min_lr = max_lr * min_lr_ratio
        experiment_name = f"lr_scan_phase2_max{max_lr:.2e}_min{min_lr:.2e}"
        
        result = run_training_experiment(
            train_data=train_data,
            valid_data=valid_data,
            max_lr=max_lr,
            min_lr=min_lr,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
            eval_interval=1000,
            eval_batches=100,
            warmup_iters=1000,
            **kwargs
        )
        
        results.append(result)
        
        print(f"\n精细搜索结果 {i+1}/{num_samples}:")
        print(f"  max_lr: {max_lr:.2e}")
        loss_str = f"{result['final_valid_loss']:.4f}" if result['final_valid_loss'] is not None else 'N/A'
        print(f"  最终验证损失: {loss_str}")
        print(f"  是否发散: {result['diverged']}")
        print(f"  耗时: {result['elapsed_time']:.1f}秒")
    
    return results


def find_divergence_point(results: List[Dict]) -> float:
    """
    找到发散临界点（导致发散的最小学习率）
    """
    diverged_lrs = [r['max_lr'] for r in results if r['diverged']]
    stable_lrs = [r['max_lr'] for r in results if not r['diverged']]
    
    if diverged_lrs:
        divergence_point = min(diverged_lrs)
        print(f"\n发现发散临界点: {divergence_point:.2e}")
        if stable_lrs:
            max_stable = max(stable_lrs)
            print(f"最大稳定学习率: {max_stable:.2e}")
            print(f"比率: {max_stable/divergence_point:.2f}x")
        return divergence_point
    else:
        print("\n未发现发散的学习率（可能需要扩大搜索范围）")
        return None


def find_best_lr(results: List[Dict], target_loss: float = 1.45) -> Dict:
    """
    找到最佳学习率（验证损失最低且未发散）
    """
    stable_results = [r for r in results if not r['diverged'] and r['final_valid_loss'] is not None]
    
    if not stable_results:
        print("\n没有找到稳定的训练结果")
        return None
    
    # 按验证损失排序
    stable_results.sort(key=lambda x: x['final_valid_loss'])
    best = stable_results[0]
    
    print(f"\n最佳学习率:")
    print(f"  max_lr: {best['max_lr']:.2e}")
    if best['final_valid_loss'] is not None:
        print(f"  最终验证损失: {best['final_valid_loss']:.4f}")
        print(f"  是否达到目标 (≤{target_loss}): {best['final_valid_loss'] <= target_loss}")
    else:
        print(f"  最终验证损失: N/A")
    
    return best


def main():
    parser = argparse.ArgumentParser(description="学习率超参数扫描")
    
    # 数据路径
    parser.add_argument("--train-data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--valid-data", type=str, required=True, help="验证数据路径")
    
    # 扫描策略
    parser.add_argument("--phase", type=str, choices=["1", "2", "both"], default="both",
                       help="扫描阶段: 1=快速筛选, 2=精细调优, both=两阶段")
    
    # 阶段1参数
    parser.add_argument("--phase1-lr-range", nargs="+", type=float,
                       default=[1e-4, 3e-4, 6e-4, 1e-3, 1.5e-3, 2e-3, 3e-3],
                       help="阶段1学习率候选列表")
    parser.add_argument("--phase1-max-iterations", type=int, default=1500,
                       help="阶段1最大迭代步数（快速筛选）")
    
    # 阶段2参数
    parser.add_argument("--phase2-lr-range", nargs=2, type=float, default=None,
                       help="阶段2学习率范围 [min, max]（如果不指定，从阶段1结果推断）")
    parser.add_argument("--phase2-max-iterations", type=int, default=5000,
                       help="阶段2最大迭代步数（完整训练）")
    parser.add_argument("--phase2-num-samples", type=int, default=5,
                       help="阶段2采样数量")
    
    # 其他训练参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="min_lr = max_lr * min_lr_ratio")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="lr_scan_results",
                       help="结果保存目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练参数
    train_kwargs = {
        "batch_size": args.batch_size,
        "context_length": args.context_length,
    }
    
    all_results = []
    
    # 阶段1：快速筛选
    if args.phase in ["1", "both"]:
        print("\n" + "="*60)
        print("阶段1：快速筛选")
        print("="*60)
        
        phase1_results = phase1_coarse_search(
            train_data=args.train_data,
            valid_data=args.valid_data,
            lr_candidates=args.phase1_lr_range,
            max_iterations=args.phase1_max_iterations,
            min_lr_ratio=args.min_lr_ratio,
            **train_kwargs
        )
        
        all_results.extend(phase1_results)
        
        # 保存阶段1结果
        with open(output_dir / "phase1_results.json", "w") as f:
            json.dump(phase1_results, f, indent=2)
        
        # 分析阶段1结果
        divergence_point = find_divergence_point(phase1_results)
        best_phase1 = find_best_lr(phase1_results)
        
        if args.phase == "1":
            # 只执行阶段1
            print("\n阶段1完成，结果已保存")
            return
    
    # 阶段2：精细调优
    if args.phase in ["2", "both"]:
        print("\n" + "="*60)
        print("阶段2：精细调优")
        print("="*60)
        
        # 确定阶段2的学习率范围
        if args.phase2_lr_range:
            best_lr_range = tuple(args.phase2_lr_range)
        else:
            # 从阶段1结果推断
            stable_results = [r for r in phase1_results if not r['diverged']]
            if stable_results:
                stable_lrs = [r['max_lr'] for r in stable_results]
                best_lr_range = (min(stable_lrs) * 0.8, max(stable_lrs) * 1.2)
            else:
                print("警告：阶段1没有稳定结果，使用默认范围")
                best_lr_range = (3e-4, 1e-3)
        
        phase2_results = phase2_fine_search(
            train_data=args.train_data,
            valid_data=args.valid_data,
            best_lr_range=best_lr_range,
            num_samples=args.phase2_num_samples,
            max_iterations=args.phase2_max_iterations,
            min_lr_ratio=args.min_lr_ratio,
            **train_kwargs
        )
        
        all_results.extend(phase2_results)
        
        # 保存阶段2结果
        with open(output_dir / "phase2_results.json", "w") as f:
            json.dump(phase2_results, f, indent=2)
        
        # 分析阶段2结果
        best_phase2 = find_best_lr(phase2_results, target_loss=1.45)
    
    # 保存所有结果
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 生成总结报告
    print("\n" + "="*60)
    print("扫描总结")
    print("="*60)
    
    divergence_point = find_divergence_point(all_results)
    best_overall = find_best_lr(all_results, target_loss=1.45)
    
    # 保存总结
    summary = {
        "divergence_point": divergence_point,
        "best_lr": best_overall,
        "total_experiments": len(all_results),
        "diverged_count": sum(1 for r in all_results if r['diverged']),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

