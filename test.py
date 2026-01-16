# 任务1：打印模型结构
import os
import shutil
import subprocess
from llava.model.builder import load_pretrained_model
print('import done!')


local_model_path = "/data_ssd/liuyu/LM_models/llava-v1.5-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=local_model_path,
    model_name="llava-v1.5-13b",
    model_base=None,
    use_safetensors=False
)

# 查看模型结构
print(model)
print("\n=== Vision Tower ===")
print(model.get_vision_tower())
print("\n=== MM Projector ===")
print(model.model.mm_projector)

# 任务2：追踪forward流程
import torch

# 准备输入
input_ids = torch.tensor([[1, 2, -200, 3, 4]]) # 包含IMAGE_TOKEN
images = torch.randn(1, 3, 336, 336).half()

with torch.no_grad():
    output = model(input_ids=input_ids, images=images)
    print(output.logits.shape)