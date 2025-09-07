# SteveAI - 快速开始指南

## 概述

SteveAI 是一个基于知识蒸馏的大语言模型训练框架，可以帮助您训练更小、更快的学生模型，同时保持接近教师模型的性能。

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support (推荐 RTX 3080 或更高)
- **内存**: 至少 16GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求
- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- PyTorch 1.12+

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/your-username/SteveAI.git
cd SteveAI
```

### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n steveai python=3.9
conda activate steveai

# 或使用 venv
python -m venv steveai_env
source steveai_env/bin/activate  # Linux/Mac
# steveai_env\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 快速开始

### 步骤 1: 教师模型推理

首先运行教师模型推理，生成用于知识蒸馏的logits：

```bash
python teacher_inference_gpu.py
```

这将：
- 加载 DeepSeek-Coder-6.7B 教师模型
- 在 Alpaca 数据集上运行推理
- 保存教师logits到 `./output/SteveAI_Teacher_Inference/teacher_logits_float16/`

### 步骤 2: 学生模型训练

使用教师logits训练学生模型：

```bash
python student_training.py \
    --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 \
    --output_dir ./output/student_model \
    --num_epochs 3 \
    --batch_size 8
```

### 步骤 3: 模型评估

评估训练好的学生模型：

```bash
python evaluate.py \
    --model_path ./output/student_model/final_model \
    --tokenizer_path ./output/student_model/final_model \
    --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 \
    --output_dir ./output/evaluation_results
```

### 步骤 4: 性能基准测试

运行性能基准测试：

```bash
python benchmark.py \
    --model_path ./output/student_model/final_model \
    --tokenizer_path ./output/student_model/final_model \
    --teacher_logits_path ./output/SteveAI_Teacher_Inference/teacher_logits_float16 \
    --output_dir ./output/benchmark_results
```

## 配置说明

### 基本配置 (config.yaml)

```yaml
# 模型配置
model:
  teacher_model_id: "deepseek-ai/deepseek-coder-6.7b-instruct"
  student_model_id: "deepseek-ai/deepseek-coder-1.3b-instruct"

# 训练配置
training:
  learning_rate: 5e-5
  num_epochs: 3
  batch_size: 8
  temperature: 4.0
  alpha: 0.7  # 蒸馏损失权重
  beta: 0.3   # 硬损失权重

# 数据集配置
dataset:
  dataset_id: "yahma/alpaca-cleaned"
  dataset_subset_size: 500
  max_seq_length: 256
```

### 环境配置

项目会自动检测运行环境：
- **Kaggle**: 自动使用 `/kaggle/working/` 路径
- **本地**: 使用 `./output/` 路径

## 使用示例

### 基本使用

```python
from config_manager import ConfigManager
from data_utils import prepare_student_dataset
from distillation_loss import DistillationLoss
from student_training import load_student_model_and_tokenizer

# 加载配置
config = ConfigManager("config.yaml")

# 加载模型
model, tokenizer = load_student_model_and_tokenizer()

# 准备数据
dataset = prepare_student_dataset(
    teacher_logits_path="./output/teacher_logits",
    tokenizer=tokenizer
)

# 创建损失函数
loss_fn = DistillationLoss()

# 训练模型
# ... 训练代码
```

### 高级使用

```python
from distillation_loss import AdvancedDistillationLoss
from optimize_model import ModelOptimizer
from deploy import ModelServer

# 使用高级蒸馏损失
loss_fn = AdvancedDistillationLoss()

# 模型优化
optimizer = ModelOptimizer(model)
quantized_model = optimizer.quantize_model('dynamic')

# 模型部署
server = ModelServer(model_path="./model", tokenizer_path="./model")
server.load_model()
```

## 常见问题

### Q: 内存不足怎么办？

A: 尝试以下方法：
1. 减小 `batch_size`
2. 减小 `max_seq_length`
3. 使用 `gradient_accumulation_steps`
4. 启用混合精度训练

### Q: 训练速度慢怎么办？

A: 优化建议：
1. 增加 `num_workers` 用于数据加载
2. 使用更大的 `batch_size`
3. 启用 `pin_memory=True`
4. 使用更快的存储设备

### Q: 如何调整蒸馏参数？

A: 关键参数：
- `temperature`: 控制软目标平滑度 (推荐 3-5)
- `alpha`: 蒸馏损失权重 (推荐 0.7)
- `beta`: 硬损失权重 (推荐 0.3)

### Q: 如何监控训练过程？

A: 使用训练监控：

```bash
python monitor_training.py --log_dir ./logs
```

## 性能优化

### 1. 内存优化

```python
# 定期清理GPU缓存
import torch
torch.cuda.empty_cache()

# 使用梯度累积
config.set('training.gradient_accumulation_steps', 4)
```

### 2. 速度优化

```python
# 使用多进程数据加载
config.set('inference.num_workers', 4)

# 启用混合精度
config.set('training.use_amp', True)
```

### 3. 存储优化

```python
# 使用压缩保存
torch.save(model.state_dict(), 'model.pt', _use_new_zipfile_serialization=True)
```

## 结果分析

### 训练结果

训练完成后，您可以在以下位置找到结果：

- **模型检查点**: `./output/checkpoints/`
- **最终模型**: `./output/final_model/`
- **评估结果**: `./output/evaluation_results/`
- **基准测试**: `./output/benchmark_results/`

### 可视化结果

```bash
python visualize_results.py \
    --results_dir ./output \
    --output_dir ./visualizations
```

## 下一步

1. **阅读完整文档**: 查看 [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
2. **运行示例**: 查看 [examples/](examples/) 目录
3. **自定义配置**: 修改 `config.yaml` 适应您的需求
4. **扩展功能**: 基于现有代码添加新功能

## 获取帮助

- **GitHub Issues**: 报告bug或请求功能
- **文档**: 查看完整的API文档
- **示例**: 运行提供的示例代码

## 许可证

本项目使用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

**开始您的知识蒸馏之旅吧！** 🚀
