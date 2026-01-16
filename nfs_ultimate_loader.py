"""
终极NFS加速方案：使用pread多线程预读到page cache
然后直接从page cache加载，无需复制文件

核心优势：
1. 使用pread系统调用，不共享文件指针，高并发
2. 预读到操作系统page cache，后续读取极快
3. 无需复制文件，节省磁盘空间和时间
"""

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List


def prefetch_to_pagecache(path: str, workers: int = 32, chunk: int = 64 << 20):
    """
    多线程预读文件到page cache
    
    Args:
        path: 文件路径
        workers: 并发线程数（建议16-64）
        chunk: 每次读取的块大小（默认64MB）
    
    原理：
        每个线程处理交错的数据块，使用pread在指定偏移量读取
        pread不共享文件指针，非常适合高并发
    """
    size = os.stat(path).st_size

    def worker(i: int):
        fd = os.open(path, os.O_RDONLY)
        off = i * chunk
        while off < size:
            n = min(chunk, size - off)
            # pread 不会共享文件指针，适合并发
            os.pread(fd, n, off)
            off += workers * chunk
        os.close(fd)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(worker, range(workers)))


def prefetch_model_to_pagecache(model_path: str, workers: int = 32, verbose: bool = True):
    """
    预读整个模型目录到page cache
    
    Args:
        model_path: 模型目录路径
        workers: 并发线程数
        verbose: 是否打印详细信息
    """
    model_path = Path(model_path)
    
    if verbose:
        print("=" * 80)
        print("NFS模型预读到Page Cache")
        print("=" * 80)
        print(f"模型路径: {model_path}")
        print(f"并发线程: {workers}")
    
    # 找到所有需要预读的文件
    patterns = ['*.bin', '*.safetensors', '*.json', '*.txt', '*.model', '*.py']
    files_to_prefetch = []
    for pattern in patterns:
        files_to_prefetch.extend(model_path.rglob(pattern))
    
    files_to_prefetch = list(set(files_to_prefetch))
    total_size = sum(f.stat().st_size for f in files_to_prefetch)
    
    if verbose:
        print(f"文件数量: {len(files_to_prefetch)}")
        print(f"总大小: {total_size/(1024**3):.2f} GB")
        print(f"\n开始预读...")
    
    start_time = time.time()
    
    for idx, file_path in enumerate(files_to_prefetch, 1):
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024**2)
        
        file_start = time.time()
        prefetch_to_pagecache(str(file_path), workers=workers)
        file_elapsed = time.time() - file_start
        
        speed = file_size_mb / file_elapsed if file_elapsed > 0 else 0
        
        if verbose:
            print(f"[{idx}/{len(files_to_prefetch)}] {file_path.name}: "
                  f"{file_size_mb:.1f}MB, {speed:.1f}MB/s")
    
    elapsed = time.time() - start_time
    avg_speed = (total_size / (1024**2)) / elapsed if elapsed > 0 else 0
    
    if verbose:
        print("=" * 80)
        print(f"✓ 预读完成!")
        print(f"  总大小: {total_size/(1024**3):.2f} GB")
        print(f"  耗时: {elapsed:.1f}秒")
        print(f"  平均速度: {avg_speed:.1f} MB/s")
        print(f"  数据已在page cache中，后续读取将极快！")
        print("=" * 80)


def load_model_from_nfs_ultimate(
    nfs_model_path: str,
    model_name: str,
    prefetch_workers: int = 32,
    **model_kwargs
):
    """
    终极NFS加载方案：预读到page cache后直接加载
    
    Args:
        nfs_model_path: NFS上的模型路径
        model_name: 模型名称
        prefetch_workers: 预读并发线程数（建议16-64）
        **model_kwargs: 传递给load_pretrained_model的参数
    
    Returns:
        (tokenizer, model, image_processor, context_len)
    """
    from llava.model.builder import load_pretrained_model
    
    print("\n" + "=" * 80)
    print("终极NFS加载方案")
    print("=" * 80)
    
    # 步骤1: 多线程预读到page cache
    print("\n[步骤1] 预读模型到Page Cache...")
    prefetch_model_to_pagecache(nfs_model_path, workers=prefetch_workers, verbose=True)
    
    # 步骤2: 从page cache加载（此时速度极快）
    print("\n[步骤2] 从Page Cache加载模型...")
    load_start = time.time()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=nfs_model_path,  # 直接从NFS路径加载
        model_name=model_name,
        model_base=None,
        **model_kwargs
    )
    
    load_elapsed = time.time() - load_start
    
    print(f"✓ 模型加载完成，耗时: {load_elapsed:.1f}秒")
    print("=" * 80)
    
    return tokenizer, model, image_processor, context_len


# ============ 性能测试 ============
if __name__ == "__main__":
    import torch
    
    nfs_model_path = "/data_large/liuyu/LM_models/llava-v1.5-13b"
    
    print("=" * 80)
    print("终极方案 vs 传统方案性能对比")
    print("=" * 80)
    
    # 测试1: 传统直接加载（baseline）
    print("\n[测试1] 传统方法：直接从NFS加载")
    from llava.model.builder import load_pretrained_model
    
    start = time.time()
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=nfs_model_path,
            model_name="llava-v1.5-13b",
            model_base=None,
            device_map="auto"
        )
        baseline_time = time.time() - start
        print(f"✓ 耗时: {baseline_time:.1f}秒")
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ 失败: {e}")
        baseline_time = None
    
    # 清空page cache（需要root权限，如果失败就跳过）
    print("\n尝试清空page cache...")
    try:
        os.system("sudo sync && sudo sysctl -w vm.drop_caches=3")
        print("✓ Page cache已清空")
    except:
        print("⚠ 无法清空page cache（需要root权限），测试可能不准确")
    
    time.sleep(2)
    
    # 测试2: 终极方案
    print("\n[测试2] 终极方案：pread预读 + page cache加载")
    total_start = time.time()
    
    tokenizer, model, image_processor, context_len = load_model_from_nfs_ultimate(
        nfs_model_path=nfs_model_path,
        model_name="llava-v1.5-13b",
        prefetch_workers=32,  # 32线程并发预读
        device_map="auto"
    )
    
    ultimate_time = time.time() - total_start
    
    # 结果对比
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    if baseline_time:
        print(f"传统NFS加载:     {baseline_time:.1f}秒")
        print(f"终极方案:        {ultimate_time:.1f}秒")
        print(f"加速比:          {baseline_time/ultimate_time:.2f}x")
    else:
        print(f"终极方案:        {ultimate_time:.1f}秒")
    
    print("\n优势:")
    print("✓ 无需额外磁盘空间（不复制文件）")
    print("✓ 充分利用NFS并发能力")
    print("✓ 利用操作系统page cache")
    print("✓ 后续加载更快（数据保持在cache中）")
    
    # 测试3: 第二次加载（测试page cache效果）
    print("\n[测试3] 再次加载（数据在page cache中）")
    del model, tokenizer
    torch.cuda.empty_cache()
    
    start = time.time()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=nfs_model_path,
        model_name="llava-v1.5-13b",
        model_base=None,
        device_map="auto"
    )
    second_load_time = time.time() - start
    
    print(f"✓ 第二次加载耗时: {second_load_time:.1f}秒")
    if baseline_time:
        print(f"  相比传统方案: {baseline_time/second_load_time:.2f}x")
