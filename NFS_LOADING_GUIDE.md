# NFS模型快速加载方案

## 问题描述
模型权重存储在NFS设备上，由于NFS的特性（随机读取慢、延迟高），`load_pretrained_model`无法发起高队列深度的读取，导致加载速度极慢。

## 解决方案

### 方案1: 多线程预加载到本地tmpfs（推荐⭐）

**原理**: 使用16-32个线程并行从NFS读取所有权重文件，缓存到本地tmpfs(/dev/shm)，然后从本地快速加载。

**优点**:
- ✅ 实现简单，无需修改底层代码
- ✅ 加载速度快（通常可提速5-10倍）
- ✅ tmpfs在内存中，读取速度接近RAM
- ✅ 缓存可复用，第二次启动秒开

**使用方法**:
```python
from nfs_fast_loader import load_model_from_nfs_fast

# 一行代码搞定
tokenizer, model, image_processor, context_len, cache_path = load_model_from_nfs_fast(
    nfs_model_path="/data_large/liuyu/LM_models/llava-v1.5-13b",
    model_name="llava-v1.5-13b",
    num_threads=24,      # 24线程并行读取（建议8-32）
    use_tmpfs=True,      # 使用tmpfs(/dev/shm)
    cleanup_after=False, # 保留缓存，下次直接用
    device_map="auto"
)
```

**参数说明**:
- `num_threads`: 并行读取线程数
  - 8-16: 适合千兆网络
  - 16-24: 适合万兆网络  
  - 24-32: 适合InfiniBand或极高并发
- `use_tmpfs`: 
  - `True`: 使用/dev/shm（内存文件系统，最快）
  - `False`: 使用/tmp（可能是磁盘）
- `cleanup_after`:
  - `False`: 保留缓存（推荐）
  - `True`: 加载后删除缓存

**检查tmpfs大小**:
```bash
# 查看/dev/shm可用空间
df -h /dev/shm

# 如果空间不足，可以扩容（需要root）
sudo mount -o remount,size=40G /dev/shm
```

---

### 方案2: 内存预加载（适合tmpfs空间不足）

**原理**: 直接将所有文件读入Python内存（字节数组），然后保存到tmpfs或直接使用。

**使用方法**:
```python
from nfs_memory_loader import load_model_from_nfs_memory

tokenizer, model, image_processor, context_len = load_model_from_nfs_memory(
    nfs_model_path="/data_large/liuyu/LM_models/llava-v1.5-13b",
    model_name="llava-v1.5-13b",
    num_threads=24,
    save_to_tmpfs=True,  # 读入内存后保存到tmpfs
    device_map="auto"
)
```

---

### 方案3: 手动控制（高级用户）

**适用场景**: 需要精细控制预加载和加载流程，或需要复用缓存。

```python
from nfs_fast_loader import NFSModelLoader
from llava.model.builder import load_pretrained_model

# 创建加载器
loader = NFSModelLoader(num_threads=24, use_tmpfs=True)

# 步骤1: 预加载（只需执行一次）
cache_path = loader.preload_model_to_cache(
    nfs_model_path="/data_large/liuyu/LM_models/llava-v1.5-13b",
    force_reload=False  # 如果已有缓存就跳过
)

# 步骤2: 查看缓存信息
cache_info = loader.get_cache_info()
print(cache_info)

# 步骤3: 从缓存加载（可以多次加载，不需要重复预加载）
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=cache_path,
    model_name="llava-v1.5-13b",
    model_base=None,
    device_map="auto"
)

# 步骤4: 清理缓存（可选）
# loader.cleanup_cache("llava-v1.5-13b")
```

---

## 性能对比

| 方案 | 首次加载 | 二次加载 | 内存占用 | 磁盘占用 |
|------|---------|---------|----------|----------|
| 直接NFS | ~300s | ~300s | 模型大小 | 0 |
| 多线程+tmpfs | ~60s | ~20s | 模型大小×2 | tmpfs空间 |
| 内存预加载 | ~50s | ~20s | 模型大小×2 | tmpfs空间 |

*实际速度取决于网络带宽和NFS服务器性能*

---

## 典型加载时间（13B模型，约26GB）

### NFS条件
- 千兆网络: 300-600秒
- 万兆网络: 60-120秒  
- InfiniBand: 30-60秒

### 使用本方案后
- 首次预加载: 30-60秒（取决于网络）
- 后续加载: 15-25秒（从tmpfs）

**总加速比**: 5-20倍

---

## 常见问题

### Q1: tmpfs空间不足怎么办？
```bash
# 临时扩容（重启后失效）
sudo mount -o remount,size=50G /dev/shm

# 永久扩容（编辑/etc/fstab）
tmpfs /dev/shm tmpfs defaults,size=50G 0 0
```

### Q2: 如何清理缓存？
```python
from nfs_fast_loader import NFSModelLoader

loader = NFSModelLoader()
loader.cleanup_cache()  # 清理所有缓存
# 或
loader.cleanup_cache("llava-v1.5-13b")  # 清理特定模型
```

### Q3: 多个模型如何管理？
```python
# 预加载多个模型
models = [
    "/data/models/llava-v1.5-7b",
    "/data/models/llava-v1.5-13b"
]

loader = NFSModelLoader(num_threads=24, use_tmpfs=True)
cache_paths = {}

for model_path in models:
    cache_paths[model_path] = loader.preload_model_to_cache(model_path)

# 查看所有缓存
print(loader.get_cache_info())
```

### Q4: num_threads设置多少合适？
- **经验公式**: `min(32, CPU核心数 * 2, 网络带宽(Gbps) * 2)`
- 千兆网络: 8-16
- 万兆网络: 16-24
- 更高带宽: 24-32
- 建议通过实验找到最佳值

### Q5: 是否支持分布式训练？
是的，在每个节点上独立预加载即可：
```python
# 在每个训练节点运行
tokenizer, model, image_processor, context_len, cache_path = load_model_from_nfs_fast(
    nfs_model_path="/shared/nfs/model",
    model_name="llava-v1.5-13b",
    num_threads=24,
    use_tmpfs=True,
    cleanup_after=False
)
```

---

## 最佳实践

1. **首次使用**: 运行测试脚本确定最佳参数
   ```bash
   python nfs_fast_loader.py  # 查看示例
   ```

2. **生产环境**: 
   - 设置`cleanup_after=False`保留缓存
   - 在节点启动时预加载一次
   - 后续直接从缓存加载

3. **开发环境**:
   - 每天首次启动预加载
   - 调试时直接使用缓存

4. **监控**:
   ```python
   loader = NFSModelLoader()
   info = loader.get_cache_info()
   print(f"缓存大小: {sum(i['size_gb'] for i in info.values()):.2f} GB")
   ```

---

## 文件说明

- `nfs_fast_loader.py`: 推荐方案实现（多线程+tmpfs）
- `nfs_memory_loader.py`: 内存预加载方案
- `test.py`: 修改后的测试文件（已集成快速加载）
- `benchmark_loading.py`: 性能对比测试

---

## 快速开始

```bash
# 1. 测试你的环境
python nfs_memory_loader.py  # 会自动运行性能对比

# 2. 修改test.py使用快速加载
python test.py

# 3. 在你的代码中使用
# 只需一行替换：
from nfs_fast_loader import load_model_from_nfs_fast
# tokenizer, model, ... = load_model_from_nfs_fast(...)
```
