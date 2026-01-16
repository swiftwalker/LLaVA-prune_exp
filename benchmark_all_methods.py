"""
NFS加载方案完整对比测试

比较三种方案：
1. 传统方案：直接从NFS加载
2. tmpfs方案：多线程复制到tmpfs再加载
3. 终极方案：pread预读到page cache后直接加载
"""

import os
import sys
import time
import torch
from pathlib import Path


def clear_gpu_memory():
    """清理GPU内存"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def test_direct_nfs_loading(model_path: str):
    """测试1: 直接从NFS加载"""
    print("\n" + "=" * 80)
    print("[方案1] 传统方案：直接从NFS加载")
    print("=" * 80)
    
    from llava.model.builder import load_pretrained_model
    
    start = time.time()
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_name="llava-v1.5-13b",
            model_base=None,
            device_map="auto"
        )
        elapsed = time.time() - start
        
        print(f"✓ 加载成功")
        print(f"  耗时: {elapsed:.1f}秒")
        
        # 清理
        del model, tokenizer
        clear_gpu_memory()
        
        return elapsed
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


def test_tmpfs_loading(model_path: str):
    """测试2: tmpfs方案"""
    print("\n" + "=" * 80)
    print("[方案2] tmpfs方案：多线程复制到tmpfs再加载")
    print("=" * 80)
    
    from nfs_fast_loader import load_model_from_nfs_fast
    
    start = time.time()
    try:
        tokenizer, model, image_processor, context_len, cache_path = load_model_from_nfs_fast(
            nfs_model_path=model_path,
            model_name="llava-v1.5-13b",
            num_threads=24,
            use_tmpfs=True,
            cleanup_after=True,  # 测试完清理
            device_map="auto"
        )
        elapsed = time.time() - start
        
        print(f"✓ 加载成功")
        print(f"  总耗时: {elapsed:.1f}秒")
        
        # 清理
        del model, tokenizer
        clear_gpu_memory()
        
        return elapsed
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


def test_ultimate_loading(model_path: str):
    """测试3: 终极方案（pread + page cache）"""
    print("\n" + "=" * 80)
    print("[方案3] 终极方案：pread预读到page cache后直接加载")
    print("=" * 80)
    
    from nfs_ultimate_loader import load_model_from_nfs_ultimate
    
    start = time.time()
    try:
        tokenizer, model, image_processor, context_len = load_model_from_nfs_ultimate(
            nfs_model_path=model_path,
            model_name="llava-v1.5-13b",
            prefetch_workers=32,
            device_map="auto"
        )
        elapsed = time.time() - start
        
        print(f"✓ 加载成功")
        print(f"  总耗时: {elapsed:.1f}秒")
        
        # 清理
        del model, tokenizer
        clear_gpu_memory()
        
        return elapsed
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


def main():
    model_path = "/data_large/liuyu/LM_models/llava-v1.5-13b"
    
    if not Path(model_path).exists():
        print(f"错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("NFS加载方案完整性能测试")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"测试环境:")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  CPU核心数: {os.cpu_count()}")
    
    results = {}
    
    # 测试1: 直接NFS加载
    time.sleep(3)  # 等待系统稳定
    results['direct'] = test_direct_nfs_loading(model_path)
    
    # 尝试清空page cache
    print("\n尝试清空page cache...")
    try:
        os.system("sync")
        time.sleep(1)
        # 注意：这需要root权限
        os.system("sudo sysctl -w vm.drop_caches=3 2>/dev/null")
        print("✓ Page cache已清空")
    except:
        print("⚠ 无法清空page cache（需要root权限）")
    
    time.sleep(3)
    
    # 测试2: tmpfs方案
    results['tmpfs'] = test_tmpfs_loading(model_path)
    
    # 清空page cache
    print("\n尝试清空page cache...")
    try:
        os.system("sync")
        time.sleep(1)
        os.system("sudo sysctl -w vm.drop_caches=3 2>/dev/null")
        print("✓ Page cache已清空")
    except:
        print("⚠ 无法清空page cache")
    
    time.sleep(3)
    
    # 测试3: 终极方案
    results['ultimate'] = test_ultimate_loading(model_path)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    
    baseline = results.get('direct')
    
    print(f"\n方案                    耗时(秒)    加速比    磁盘占用")
    print("-" * 80)
    
    if baseline:
        print(f"传统NFS加载             {baseline:7.1f}     1.00x      无")
    else:
        print(f"传统NFS加载               失败       -        无")
    
    if results['tmpfs']:
        speedup = baseline / results['tmpfs'] if baseline else 0
        print(f"tmpfs方案               {results['tmpfs']:7.1f}     {speedup:4.2f}x     模型2倍")
    else:
        print(f"tmpfs方案                 失败       -        -")
    
    if results['ultimate']:
        speedup = baseline / results['ultimate'] if baseline else 0
        print(f"终极方案(pread)         {results['ultimate']:7.1f}     {speedup:4.2f}x     无")
    else:
        print(f"终极方案(pread)           失败       -        -")
    
    print("\n方案特点对比:")
    print("-" * 80)
    print("传统NFS加载:")
    print("  ✓ 无需额外配置")
    print("  ✗ 速度慢，未充分利用NFS并发能力")
    print("  ✗ 每次启动都很慢")
    
    print("\ntmpfs方案:")
    print("  ✓ 首次加载后，后续启动很快")
    print("  ✓ 实现简单，稳定可靠")
    print("  ✗ 需要额外磁盘空间（模型大小×2）")
    print("  ✗ 首次复制需要时间")
    
    print("\n终极方案(pread + page cache):")
    print("  ✓ 无需额外磁盘空间")
    print("  ✓ 充分利用NFS并发能力")
    print("  ✓ 利用系统page cache，后续加载快")
    print("  ✓ 最灵活，最高效")
    print("  ⚠ 重启后page cache丢失，需重新预读")
    
    print("\n推荐使用方案:")
    if not Path('/dev/shm').exists() or os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize < 30 * 1024**3:
        print("  → 终极方案 (tmpfs空间不足)")
    else:
        print("  → tmpfs方案 (有充足tmpfs空间，需要频繁重启)")
        print("  → 终极方案 (磁盘空间有限，长时间运行)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
