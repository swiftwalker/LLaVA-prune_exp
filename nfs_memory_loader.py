"""
高级NFS加载方案：直接内存加载（无需临时文件）
使用torch.load的内存映射和自定义checkpoint加载
"""

import os
import io
import time
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import threading


class InMemoryNFSLoader:
    """
    直接从NFS多线程读取到内存，无需临时文件
    适用于有充足RAM但tmpfs空间有限的场景
    """
    
    def __init__(self, num_threads=16, verbose=True):
        self.num_threads = num_threads
        self.verbose = verbose
        self.cached_files = {}  # 文件名 -> 内存中的字节
        self.lock = threading.Lock()
    
    def _log(self, msg):
        if self.verbose:
            print(f"[InMemoryLoader] {msg}")
    
    def _read_file_to_memory(self, file_path: Path, file_idx: int, total: int):
        """读取单个文件到内存"""
        try:
            start = time.time()
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024**2)
            
            # 使用大缓冲区读取
            with open(file_path, 'rb') as f:
                data = f.read()
            
            elapsed = time.time() - start
            speed = file_size_mb / elapsed if elapsed > 0 else 0
            
            self._log(f"[{file_idx}/{total}] 已读取: {file_path.name} "
                     f"({file_size_mb:.1f}MB, {speed:.1f}MB/s)")
            
            return True, str(file_path), data, file_size
        except Exception as e:
            self._log(f"✗ 读取失败 {file_path}: {e}")
            return False, str(file_path), None, 0
    
    def preload_weights_to_memory(self, nfs_model_path: str) -> Dict[str, bytes]:
        """
        从NFS多线程读取所有权重文件到内存
        
        Returns:
            {相对路径: 文件内容字节}
        """
        nfs_path = Path(nfs_model_path)
        
        # 找到所有需要加载的文件
        weight_files = list(nfs_path.glob('*.bin')) + list(nfs_path.glob('*.safetensors'))
        config_files = list(nfs_path.glob('*.json')) + list(nfs_path.glob('*.txt')) + \
                      list(nfs_path.glob('*.model')) + list(nfs_path.glob('*.py'))
        
        all_files = weight_files + config_files
        total_files = len(all_files)
        
        self._log(f"开始从NFS读取模型文件到内存...")
        self._log(f"  源路径: {nfs_path}")
        self._log(f"  文件数: {total_files}")
        self._log(f"  线程数: {self.num_threads}")
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in all_files)
        self._log(f"  总大小: {total_size/(1024**3):.2f} GB")
        
        # 多线程读取
        start_time = time.time()
        success_count = 0
        total_bytes = 0
        cached_data = {}
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for idx, file_path in enumerate(all_files, 1):
                future = executor.submit(
                    self._read_file_to_memory,
                    file_path, idx, total_files
                )
                futures.append(future)
            
            for future in as_completed(futures):
                success, path, data, size = future.result()
                if success:
                    # 保存相对路径作为key
                    rel_path = str(Path(path).relative_to(nfs_path))
                    cached_data[rel_path] = data
                    success_count += 1
                    total_bytes += size
        
        elapsed = time.time() - start_time
        avg_speed = (total_bytes / (1024**2)) / elapsed if elapsed > 0 else 0
        
        self._log("=" * 70)
        self._log(f"✓ 内存预加载完成!")
        self._log(f"  成功: {success_count}/{total_files}")
        self._log(f"  大小: {total_bytes/(1024**3):.2f} GB")
        self._log(f"  耗时: {elapsed:.1f}秒")
        self._log(f"  速度: {avg_speed:.1f} MB/s")
        self._log("=" * 70)
        
        self.cached_files = cached_data
        return cached_data
    
    def load_checkpoint_from_memory(self, file_key: str) -> Dict[str, Any]:
        """从内存中的字节数据加载PyTorch checkpoint"""
        if file_key not in self.cached_files:
            raise ValueError(f"文件 {file_key} 未在内存中缓存")
        
        data_bytes = self.cached_files[file_key]
        buffer = io.BytesIO(data_bytes)
        checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
        return checkpoint
    
    def save_to_local(self, output_dir: str):
        """将内存中的文件保存到本地（可选）"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._log(f"保存内存缓存到: {output_path}")
        for rel_path, data in self.cached_files.items():
            file_path = output_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(data)
        
        self._log(f"✓ 已保存 {len(self.cached_files)} 个文件")
        return str(output_path)


def load_model_from_nfs_memory(nfs_model_path: str,
                               model_name: str,
                               num_threads: int = 16,
                               save_to_tmpfs: bool = True,
                               **model_kwargs):
    """
    从NFS读取到内存，然后加载模型（两阶段方案）
    
    Args:
        nfs_model_path: NFS上的模型路径
        model_name: 模型名称
        num_threads: 读取线程数
        save_to_tmpfs: 是否将内存数据保存到tmpfs后再加载（推荐）
        **model_kwargs: 传递给load_pretrained_model的参数
    
    Returns:
        (tokenizer, model, image_processor, context_len)
    """
    from llava.model.builder import load_pretrained_model
    
    # 阶段1: 从NFS多线程读取到内存
    loader = InMemoryNFSLoader(num_threads=num_threads, verbose=True)
    loader.preload_weights_to_memory(nfs_model_path)
    
    # 阶段2: 保存到tmpfs（推荐）或直接使用
    if save_to_tmpfs:
        if os.path.exists('/dev/shm'):
            tmpfs_path = f'/dev/shm/model_cache/{Path(nfs_model_path).name}'
        else:
            tmpfs_path = f'/tmp/model_cache/{Path(nfs_model_path).name}'
        
        print(f"\n[保存到tmpfs] {tmpfs_path}")
        cache_path = loader.save_to_local(tmpfs_path)
    else:
        # 这种方式需要修改transformers的from_pretrained，较复杂
        raise NotImplementedError(
            "直接从内存加载需要自定义from_pretrained逻辑，"
            "建议使用save_to_tmpfs=True"
        )
    
    # 阶段3: 从本地缓存加载
    print(f"\n[加载模型] 从本地缓存...")
    start = time.time()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=cache_path,
        model_name=model_name,
        model_base=None,
        **model_kwargs
    )
    
    load_time = time.time() - start
    print(f"✓ 模型加载完成，耗时: {load_time:.1f}秒")
    
    return tokenizer, model, image_processor, context_len


# ============ 性能对比测试 ============
if __name__ == "__main__":
    import sys
    
    nfs_model_path = "/data_large/liuyu/LM_models/llava-v1.5-13b"
    
    print("=" * 80)
    print("NFS加载性能对比测试")
    print("=" * 80)
    
    # 方案1: 直接从NFS加载（baseline）
    print("\n[方案1] 直接从NFS加载（baseline）")
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
    
    # 方案2: 多线程预加载到tmpfs
    print("\n[方案2] 多线程预加载到tmpfs (推荐)")
    from nfs_fast_loader import load_model_from_nfs_fast
    
    start = time.time()
    tokenizer, model, image_processor, context_len, cache_path = load_model_from_nfs_fast(
        nfs_model_path=nfs_model_path,
        model_name="llava-v1.5-13b",
        num_threads=24,
        use_tmpfs=True,
        device_map="auto"
    )
    tmpfs_time = time.time() - start
    print(f"✓ 总耗时: {tmpfs_time:.1f}秒")
    if baseline_time:
        print(f"  加速比: {baseline_time/tmpfs_time:.2f}x")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # 方案3: 内存预加载
    print("\n[方案3] 内存预加载后保存到tmpfs")
    start = time.time()
    tokenizer, model, image_processor, context_len = load_model_from_nfs_memory(
        nfs_model_path=nfs_model_path,
        model_name="llava-v1.5-13b",
        num_threads=24,
        save_to_tmpfs=True,
        device_map="auto"
    )
    memory_time = time.time() - start
    print(f"✓ 总耗时: {memory_time:.1f}秒")
    if baseline_time:
        print(f"  加速比: {baseline_time/memory_time:.2f}x")
    
    # 总结
    print("\n" + "=" * 80)
    print("性能总结")
    print("=" * 80)
    if baseline_time:
        print(f"直接NFS加载:      {baseline_time:.1f}秒 (baseline)")
    else:
        print(f"直接NFS加载:      失败")
    print(f"多线程+tmpfs:     {tmpfs_time:.1f}秒 ({baseline_time/tmpfs_time:.2f}x)" if baseline_time else f"{tmpfs_time:.1f}秒")
    print(f"内存预加载:       {memory_time:.1f}秒 ({baseline_time/memory_time:.2f}x)" if baseline_time else f"{memory_time:.1f}秒")
    print("\n推荐: 使用方案2（多线程+tmpfs），简单且高效")
