"""
NFS模型快速加载器
解决NFS随机读取慢的问题，通过多线程预加载到本地内存/tmpfs

策略：
1. 使用多线程并行读取NFS上的所有权重文件
2. 缓存到本地tmpfs（/dev/shm）或本地磁盘
3. 从本地缓存加载模型
"""

import os
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import psutil


class NFSModelLoader:
    """NFS模型快速加载器"""
    
    def __init__(self, num_threads=16, use_tmpfs=True, verbose=True):
        """
        Args:
            num_threads: 并行读取线程数（建议8-32）
            use_tmpfs: 是否使用tmpfs(/dev/shm)作为缓存，否则使用/tmp
            verbose: 是否打印详细日志
        """
        self.num_threads = num_threads
        self.use_tmpfs = use_tmpfs
        self.verbose = verbose
        
        # 确定缓存目录
        if use_tmpfs and os.path.exists('/dev/shm'):
            self.cache_base = '/dev/shm/model_cache'
        else:
            self.cache_base = '/tmp/model_cache'
        
        os.makedirs(self.cache_base, exist_ok=True)
    
    def _log(self, msg):
        """打印日志"""
        if self.verbose:
            print(f"[NFSLoader] {msg}")
    
    def _get_model_files(self, model_path: str) -> List[str]:
        """获取模型目录下所有需要复制的文件"""
        model_path = Path(model_path)
        files_to_copy = []
        
        # 需要复制的文件模式
        patterns = [
            '*.bin',           # PyTorch权重
            '*.safetensors',   # SafeTensors权重
            '*.json',          # 配置文件
            '*.txt',           # 词表等文本文件
            '*.model',         # Tokenizer模型
            '*.py',            # Python配置文件
            'config.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'tokenizer.model',
            'preprocessor_config.json',
        ]
        
        for pattern in patterns:
            files_to_copy.extend(model_path.rglob(pattern))
        
        # 去重并转为相对路径
        files_to_copy = list(set(files_to_copy))
        return files_to_copy
    
    def _prefetch_to_pagecache(self, path: str, workers: int = None, chunk: int = 64 << 20):
        """
        使用多线程pread预读文件到page cache
        
        Args:
            path: 文件路径
            workers: 并发线程数（默认使用self.num_threads）
            chunk: 每次读取的块大小（默认64MB）
        """
        if workers is None:
            workers = self.num_threads
            
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
    
    def _copy_file_with_progress(self, src: Path, dst: Path, file_idx: int, total_files: int):
        """复制单个文件并显示进度（使用高效的pread预读取）"""
        try:
            os.makedirs(dst.parent, exist_ok=True)
            
            file_size = src.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            start = time.time()
            
            # 步骤1: 使用多线程pread预读到page cache
            # 这样可以充分利用NFS的并发能力
            self._prefetch_to_pagecache(str(src), workers=self.num_threads)
            
            # 步骤2: 从page cache快速复制到目标位置
            # 此时数据已在内存中，速度很快
            buffer_size = 10 * 1024 * 1024  # 10MB buffer
            with open(src, 'rb') as f_src:
                with open(dst, 'wb') as f_dst:
                    while True:
                        chunk = f_src.read(buffer_size)
                        if not chunk:
                            break
                        f_dst.write(chunk)
            
            elapsed = time.time() - start
            speed_mbps = file_size_mb / elapsed if elapsed > 0 else 0
            
            self._log(f"[{file_idx}/{total_files}] 已复制: {src.name} ({file_size_mb:.1f}MB, {speed_mbps:.1f}MB/s)")
            return True, src, dst, file_size
            
        except Exception as e:
            self._log(f"✗ 复制失败 {src}: {e}")
            return False, src, dst, 0
    
    def preload_model_to_cache(self, nfs_model_path: str, force_reload=False) -> str:
        """
        从NFS多线程预加载模型到本地缓存
        
        Args:
            nfs_model_path: NFS上的模型路径
            force_reload: 是否强制重新加载（忽略已有缓存）
        
        Returns:
            本地缓存路径
        """
        nfs_path = Path(nfs_model_path).resolve()
        model_name = nfs_path.name
        cache_path = Path(self.cache_base) / model_name
        
        # 检查缓存是否已存在
        if cache_path.exists() and not force_reload:
            self._log(f"✓ 发现已有缓存: {cache_path}")
            return str(cache_path)
        
        # 清理旧缓存
        if cache_path.exists():
            self._log(f"清理旧缓存: {cache_path}")
            shutil.rmtree(cache_path)
        
        cache_path.mkdir(parents=True, exist_ok=True)
        
        self._log(f"开始从NFS预加载模型...")
        self._log(f"  源路径: {nfs_path}")
        self._log(f"  缓存路径: {cache_path}")
        self._log(f"  线程数: {self.num_threads}")
        
        # 获取所有需要复制的文件
        files_to_copy = self._get_model_files(nfs_path)
        total_files = len(files_to_copy)
        self._log(f"  文件数量: {total_files}")
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in files_to_copy)
        total_size_gb = total_size / (1024**3)
        self._log(f"  总大小: {total_size_gb:.2f} GB")
        
        # 检查可用空间
        if self.use_tmpfs:
            available = psutil.disk_usage('/dev/shm').free
        else:
            available = psutil.disk_usage('/tmp').free
        available_gb = available / (1024**3)
        
        self._log(f"  可用空间: {available_gb:.2f} GB")
        if available < total_size * 1.1:  # 需要10%的额外空间
            self._log(f"⚠ 警告: 可用空间可能不足！")
        
        # 多线程并行复制
        start_time = time.time()
        success_count = 0
        total_bytes = 0
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for idx, src_file in enumerate(files_to_copy, 1):
                rel_path = src_file.relative_to(nfs_path)
                dst_file = cache_path / rel_path
                future = executor.submit(
                    self._copy_file_with_progress,
                    src_file, dst_file, idx, total_files
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                success, src, dst, size = future.result()
                if success:
                    success_count += 1
                    total_bytes += size
        
        elapsed = time.time() - start_time
        avg_speed = (total_bytes / (1024**2)) / elapsed if elapsed > 0 else 0
        
        self._log("=" * 70)
        self._log(f"✓ 预加载完成!")
        self._log(f"  成功: {success_count}/{total_files}")
        self._log(f"  总大小: {total_bytes/(1024**3):.2f} GB")
        self._log(f"  耗时: {elapsed:.1f}秒")
        self._log(f"  平均速度: {avg_speed:.1f} MB/s")
        self._log(f"  缓存路径: {cache_path}")
        self._log("=" * 70)
        
        return str(cache_path)
    
    def cleanup_cache(self, model_name: str = None):
        """清理缓存"""
        if model_name:
            cache_path = Path(self.cache_base) / model_name
            if cache_path.exists():
                self._log(f"清理缓存: {cache_path}")
                shutil.rmtree(cache_path)
        else:
            if Path(self.cache_base).exists():
                self._log(f"清理所有缓存: {self.cache_base}")
                shutil.rmtree(self.cache_base)
    
    def get_cache_info(self):
        """获取缓存信息"""
        if not Path(self.cache_base).exists():
            return {}
        
        info = {}
        for model_dir in Path(self.cache_base).iterdir():
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                info[model_dir.name] = {
                    'path': str(model_dir),
                    'size_gb': size / (1024**3),
                    'files': len(list(model_dir.rglob('*')))
                }
        return info


def load_model_from_nfs_fast(nfs_model_path: str, 
                             model_name: str,
                             num_threads: int = 16,
                             use_tmpfs: bool = True,
                             cleanup_after: bool = False,
                             **model_kwargs):
    """
    从NFS快速加载模型（一站式函数）
    
    Args:
        nfs_model_path: NFS上的模型路径
        model_name: 模型名称（如'llava-v1.5-13b'）
        num_threads: 预加载线程数
        use_tmpfs: 是否使用tmpfs
        cleanup_after: 加载后是否清理缓存
        **model_kwargs: 传递给load_pretrained_model的其他参数
    
    Returns:
        (tokenizer, model, image_processor, context_len, cache_path)
    """
    from llava.model.builder import load_pretrained_model
    
    # 步骤1: 预加载到本地
    loader = NFSModelLoader(num_threads=num_threads, use_tmpfs=use_tmpfs)
    cache_path = loader.preload_model_to_cache(nfs_model_path)
    
    # 步骤2: 从本地缓存加载模型
    print(f"\n[加载模型] 从本地缓存加载...")
    start = time.time()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=cache_path,
        model_name=model_name,
        model_base=None,
        **model_kwargs
    )
    
    load_time = time.time() - start
    print(f"✓ 模型加载完成，耗时: {load_time:.1f}秒")
    
    # 步骤3: 可选清理
    if cleanup_after:
        print(f"\n[清理缓存] 释放空间...")
        loader.cleanup_cache(Path(nfs_model_path).name)
    
    return tokenizer, model, image_processor, context_len, cache_path


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例1: 使用一站式函数
    print("=" * 80)
    print("示例1: 使用一站式函数从NFS快速加载")
    print("=" * 80)
    
    nfs_model_path = "/data_large/liuyu/LM_models/llava-v1.5-13b"
    
    tokenizer, model, image_processor, context_len, cache_path = load_model_from_nfs_fast(
        nfs_model_path=nfs_model_path,
        model_name="llava-v1.5-13b",
        num_threads=16,  # 16线程并行读取
        use_tmpfs=True,  # 使用tmpfs(/dev/shm)
        cleanup_after=False,  # 保留缓存以便下次使用
        # 以下是传递给load_pretrained_model的参数
        device_map="auto",
        use_safetensors=False
    )
    
    print(f"\n模型已加载，缓存在: {cache_path}")
    print(f"Context Length: {context_len}")
    
    # 示例2: 手动控制流程
    print("\n" + "=" * 80)
    print("示例2: 手动控制预加载和加载流程")
    print("=" * 80)
    
    # 创建加载器
    loader = NFSModelLoader(num_threads=24, use_tmpfs=True, verbose=True)
    
    # 预加载到本地
    cache_path = loader.preload_model_to_cache(nfs_model_path)
    
    # 查看缓存信息
    cache_info = loader.get_cache_info()
    print("\n缓存信息:")
    for name, info in cache_info.items():
        print(f"  {name}: {info['size_gb']:.2f}GB, {info['files']} 文件")
    
    # 从缓存加载（多次加载不需要重新预加载）
    from llava.model.builder import load_pretrained_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=cache_path,
        model_name="llava-v1.5-13b",
        model_base=None,
        device_map="auto"
    )
    
    # 清理缓存（可选）
    # loader.cleanup_cache("llava-v1.5-13b")
