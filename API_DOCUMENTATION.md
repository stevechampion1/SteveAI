# SteveAI - API Documentation

## 概述

SteveAI 是一个基于知识蒸馏的大语言模型训练框架，提供了完整的模型蒸馏、训练、评估、优化和部署功能。

## 核心模块

### 1. 知识蒸馏 (distillation_loss.py)

#### DistillationLoss
标准的知识蒸馏损失函数，结合软目标和硬目标。

```python
from distillation_loss import DistillationLoss

loss_fn = DistillationLoss(reduction='mean')
dist_loss, hard_loss = loss_fn(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    labels=labels,
    temperature=4.0,
    alpha=0.7,
    beta=0.3
)
```

**参数：**
- `student_logits`: 学生模型logits [batch_size, seq_len, vocab_size]
- `teacher_logits`: 教师模型logits [batch_size, seq_len, vocab_size]
- `labels`: 真实标签 [batch_size, seq_len]
- `temperature`: 温度参数，默认4.0
- `alpha`: 蒸馏损失权重，默认0.7
- `beta`: 硬损失权重，默认0.3

**返回：**
- `dist_loss`: 蒸馏损失
- `hard_loss`: 硬损失

#### AdvancedDistillationLoss
高级蒸馏损失，包含注意力转移和隐藏状态匹配。

```python
from distillation_loss import AdvancedDistillationLoss

loss_fn = AdvancedDistillationLoss()
dist_loss, hard_loss, hidden_loss, attention_loss = loss_fn(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    labels=labels,
    student_hidden_states=student_hidden,
    teacher_hidden_states=teacher_hidden,
    student_attention=student_attn,
    teacher_attention=teacher_attn
)
```

#### FocalDistillationLoss
焦点蒸馏损失，专注于困难样本。

```python
from distillation_loss import FocalDistillationLoss

loss_fn = FocalDistillationLoss(alpha=1.0, gamma=2.0)
dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)
```

### 2. 数据处理 (data_utils.py)

#### TeacherLogitsLoader
教师logits加载器。

```python
from data_utils import TeacherLogitsLoader

loader = TeacherLogitsLoader(logits_dir="/path/to/logits")
all_logits = loader.load_all_logits()
single_logits = loader.load_logits(batch_idx=0)
info = loader.get_logits_info()
```

#### StudentDataset
学生训练数据集。

```python
from data_utils import StudentDataset, create_dataloader

dataset = StudentDataset(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    teacher_logits=teacher_logits
)

dataloader = create_dataloader(
    dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=2
)
```

#### prepare_student_dataset
准备学生训练数据集。

```python
from data_utils import prepare_student_dataset

dataset = prepare_student_dataset(
    teacher_logits_path="/path/to/logits",
    tokenizer=tokenizer,
    max_seq_length=256,
    dataset_id="yahma/alpaca-cleaned",
    dataset_subset_size=500
)
```

### 3. 配置管理 (config_manager.py)

#### ConfigManager
统一的配置管理器。

```python
from config_manager import ConfigManager

# 使用默认配置
config = ConfigManager()

# 从文件加载配置
config = ConfigManager("config.yaml")

# 获取配置值
teacher_model = config.get('model.teacher_model_id')
learning_rate = config.get('training.learning_rate', default=5e-5)

# 设置配置值
config.set('training.learning_rate', 1e-4)

# 验证配置
is_valid = config.validate()

# 获取输出路径
paths = config.get_output_paths()
```

### 4. 模型训练 (student_training.py)

#### StudentTrainingConfig
学生训练配置类。

```python
from student_training import StudentTrainingConfig

config = StudentTrainingConfig()
config.LEARNING_RATE = 1e-4
config.NUM_EPOCHS = 5
```

#### 训练函数

```python
from student_training import (
    load_student_model_and_tokenizer,
    create_optimizer_and_scheduler,
    train_epoch,
    evaluate_model
)

# 加载模型和分词器
model, tokenizer = load_student_model_and_tokenizer(config)

# 创建优化器和调度器
optimizer, scheduler = create_optimizer_and_scheduler(
    model, num_training_steps=1000
)

# 训练一个epoch
train_loss = train_epoch(
    model, dataloader, optimizer, scheduler, 
    distillation_loss_fn, epoch=1, device='cuda'
)

# 评估模型
val_loss = evaluate_model(model, val_dataloader, distillation_loss_fn, device)
```

### 5. 模型评估 (evaluate.py)

#### ModelEvaluator
模型评估器。

```python
from evaluate import ModelEvaluator

evaluator = ModelEvaluator(
    model_path="/path/to/model",
    tokenizer_path="/path/to/tokenizer",
    device="auto"
)

# 评估困惑度
perplexity_metrics = evaluator.evaluate_perplexity(dataloader)

# 评估生成质量
generation_metrics = evaluator.evaluate_generation_quality(
    prompts=["Write a Python function..."],
    max_length=256
)

# 评估内存使用
memory_metrics = evaluator.evaluate_memory_usage()

# 与教师模型比较
comparison_metrics = evaluator.compare_with_teacher(
    teacher_model_path="/path/to/teacher",
    dataloader=dataloader
)

# 综合评估
results = evaluator.comprehensive_evaluation(
    dataloader=dataloader,
    teacher_model_path="/path/to/teacher",
    test_prompts=prompts
)
```

### 6. 性能基准测试 (benchmark.py)

#### PerformanceBenchmark
性能基准测试器。

```python
from benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark(
    model_path="/path/to/model",
    tokenizer_path="/path/to/tokenizer",
    device="auto"
)

# 推理速度测试
speed_metrics = benchmark.benchmark_inference_speed(dataloader)

# 生成速度测试
generation_metrics = benchmark.benchmark_generation_speed(prompts)

# 内存使用测试
memory_metrics = benchmark.benchmark_memory_usage(dataloader)

# 模型大小测试
size_metrics = benchmark.benchmark_model_size()

# 模型比较
comparison = benchmark.compare_models(
    other_model_path="/path/to/other/model",
    other_tokenizer_path="/path/to/other/tokenizer",
    dataloader=dataloader
)

# 综合基准测试
results = benchmark.comprehensive_benchmark(
    dataloader=dataloader,
    test_prompts=prompts,
    other_model_path="/path/to/other/model"
)
```

### 7. 模型优化 (optimize_model.py)

#### ModelOptimizer
模型优化器。

```python
from optimize_model import ModelOptimizer

optimizer = ModelOptimizer(model, device='cuda')

# 量化模型
quantized_model = optimizer.quantize_model(
    quantization_type='dynamic'  # 'dynamic', 'static', 'qat'
)

# 剪枝模型
pruned_model = optimizer.prune_model(
    pruning_ratio=0.2,
    pruning_type='magnitude'  # 'magnitude', 'random'
)

# 推理优化
optimized_model = optimizer.optimize_for_inference()

# 导出为ONNX
optimizer.export_to_onnx(
    input_shape=(1, 256),
    output_path="model.onnx"
)

# 导出为TorchScript
optimizer.export_to_torchscript("model.pt")
```

### 8. 模型部署 (deploy.py)

#### ModelServer
模型服务器。

```python
from deploy import ModelServer, FlaskModelServer, FastAPIModelServer

# 创建模型服务器
model_server = ModelServer(
    model_path="/path/to/model",
    tokenizer_path="/path/to/tokenizer",
    device="auto",
    max_batch_size=8
)

# 加载模型
model_server.load_model()

# 生成文本
generated_texts = model_server.generate_text(
    prompt="Write a Python function...",
    max_length=256,
    temperature=0.7
)

# 批量生成
batch_results = model_server.batch_generate(
    prompts=["Prompt 1", "Prompt 2"],
    max_length=256
)
```

#### Flask服务器

```python
from deploy import FlaskModelServer

flask_server = FlaskModelServer(model_server)
flask_server.run(host='0.0.0.0', port=5000)
```

#### FastAPI服务器

```python
from deploy import FastAPIModelServer

fastapi_server = FastAPIModelServer(model_server)
fastapi_server.run(host='0.0.0.0', port=8000)
```

### 9. 训练监控 (monitor_training.py)

#### TrainingMonitor
训练监控器。

```python
from monitor_training import TrainingMonitor

monitor = TrainingMonitor(
    log_dir="/path/to/logs",
    update_interval=30,
    save_plots=True
)

# 开始监控
monitor.start_monitoring()

# 记录epoch指标
monitor.log_epoch_metrics(
    epoch=1,
    train_loss=0.5,
    val_loss=0.6,
    learning_rate=0.001
)

# 保存指标
monitor.save_metrics()

# 生成报告
monitor.generate_report()

# 停止监控
monitor.stop_monitoring()
```

#### RealTimeMonitor
实时监控器。

```python
from monitor_training import RealTimeMonitor

monitor = RealTimeMonitor("/path/to/logs")

# 添加更新回调
def update_callback(metrics):
    print(f"Current loss: {metrics.get('train_loss', 'N/A')}")

monitor.add_update_callback(update_callback)

# 开始实时监控
monitor.start()
```

### 10. 结果可视化 (visualize_results.py)

#### ResultsVisualizer
结果可视化器。

```python
from visualize_results import ResultsVisualizer

visualizer = ResultsVisualizer(
    results_dir="/path/to/results",
    output_dir="/path/to/visualizations"
)

# 可视化所有结果
visualizer.visualize_all()

# 单独可视化训练曲线
training_metrics = visualizer.load_training_results()
visualizer.plot_training_curves(training_metrics)

# 可视化系统指标
system_metrics = visualizer.load_training_results()
visualizer.plot_system_metrics(system_metrics)

# 创建交互式仪表板
all_results = {
    'training': training_metrics,
    'evaluation': evaluation_metrics,
    'benchmark': benchmark_metrics
}
visualizer.create_interactive_dashboard(all_results)
```

### 11. 工具函数 (utils.py)

#### 日志设置

```python
from utils import setup_logging

setup_logging(
    level="INFO",
    format_string="%(asctime)s - %(levelname)s - %(message)s",
    log_file="training.log"
)
```

#### 内存监控

```python
from utils import print_memory_usage, get_system_info

# 打印内存使用
print_memory_usage("After model loading")

# 获取系统信息
system_info = get_system_info()
print(f"CPU count: {system_info['cpu_count']}")
print(f"Memory: {system_info['memory_total']} GB")
```

#### 文件操作

```python
from utils import save_json, load_json, format_time, format_size

# 保存JSON
data = {"loss": 0.5, "epoch": 1}
save_json(data, "results.json")

# 加载JSON
loaded_data = load_json("results.json")

# 格式化时间
time_str = format_time(3600)  # "1.00h"

# 格式化大小
size_str = format_size(1024**3)  # "1.00 GB"
```

### 12. 模型工具 (model_utils.py)

#### 模型检查点

```python
from model_utils import save_model_checkpoint, load_model_checkpoint

# 保存检查点
save_model_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=1,
    loss=0.5,
    file_path="checkpoint.pt"
)

# 加载检查点
checkpoint_info = load_model_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    file_path="checkpoint.pt"
)
```

#### 模型信息

```python
from model_utils import get_model_summary, count_trainable_parameters

# 获取模型摘要
summary = get_model_summary(model)
print(f"Total parameters: {summary['total_parameters']:,}")
print(f"Model size: {summary['model_size_mb']:.2f} MB")

# 计算可训练参数
trainable_count = count_trainable_parameters(model)
print(f"Trainable parameters: {trainable_count:,}")
```

#### 模型操作

```python
from model_utils import (
    freeze_model_parameters, 
    unfreeze_model_parameters,
    initialize_weights,
    clip_gradients
)

# 冻结参数
freeze_model_parameters(model, ['embedding', 'lm_head'])

# 解冻参数
unfreeze_model_parameters(model, ['embedding'])

# 初始化权重
initialize_weights(model, 'xavier_uniform')

# 梯度裁剪
grad_norm = clip_gradients(model, max_norm=1.0)
```

## 命令行接口

### 教师推理

```bash
python teacher_inference_gpu.py
```

### 学生训练

```bash
python student_training.py \
    --teacher_logits_path /path/to/logits \
    --output_dir /path/to/output \
    --num_epochs 3 \
    --batch_size 8
```

### 模型评估

```bash
python evaluate.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --teacher_logits_path /path/to/logits \
    --output_dir /path/to/results
```

### 性能基准测试

```bash
python benchmark.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --teacher_logits_path /path/to/logits \
    --output_dir /path/to/results
```

### 模型优化

```bash
python optimize_model.py \
    --model_path /path/to/model \
    --output_dir /path/to/optimized \
    --quantization_type dynamic \
    --pruning_ratio 0.2
```

### 模型部署

```bash
python deploy.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --server_type fastapi \
    --port 8000
```

### 训练监控

```bash
python monitor_training.py \
    --log_dir /path/to/logs \
    --update_interval 30
```

### 结果可视化

```bash
python visualize_results.py \
    --results_dir /path/to/results \
    --output_dir /path/to/visualizations
```

## 配置示例

### config.yaml

```yaml
# 模型配置
model:
  teacher_model_id: "deepseek-ai/deepseek-coder-6.7b-instruct"
  student_model_id: "deepseek-ai/deepseek-coder-1.3b-instruct"
  tokenizer_path: "deepseek-ai/deepseek-coder-6.7b-instruct"

# 数据集配置
dataset:
  dataset_id: "yahma/alpaca-cleaned"
  dataset_subset_size: 500
  max_seq_length: 256
  train_split: "train"

# 训练配置
training:
  learning_rate: 5e-5
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  warmup_steps: 100
  temperature: 4.0
  alpha: 0.7
  beta: 0.3

# 推理配置
inference:
  batch_size: 4
  num_workers: 2
  dtype: "float16"

# 输出配置
output:
  base_dir: "./output/SteveAI"
  logits_dir: "teacher_logits_float16"
  checkpoint_dir: "checkpoints"
  final_model_dir: "final_model"
  save_frequency: 20

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  log_memory_every: 20

# 环境配置
environment:
  auto_detect: true
  kaggle_path: "/kaggle/working/SteveAI"
  local_path: "./output/SteveAI"
```

## 错误处理

所有模块都包含完善的错误处理机制：

```python
try:
    # 模型操作
    model = load_model(model_path)
except FileNotFoundError:
    logger.error(f"Model file not found: {model_path}")
    raise
except RuntimeError as e:
    logger.error(f"Model loading failed: {e}")
    raise
```

## 性能优化建议

1. **内存优化**：
   - 使用 `torch.cuda.empty_cache()` 清理GPU缓存
   - 设置合适的batch size
   - 使用梯度累积

2. **速度优化**：
   - 使用多进程数据加载
   - 启用混合精度训练
   - 使用模型量化

3. **存储优化**：
   - 使用压缩格式保存模型
   - 定期清理临时文件
   - 使用增量保存

## 最佳实践

1. **配置管理**：使用 `ConfigManager` 统一管理配置
2. **日志记录**：使用 `setup_logging` 设置日志
3. **内存监控**：定期调用 `print_memory_usage`
4. **错误处理**：使用try-catch包装关键操作
5. **资源清理**：及时删除不需要的变量和对象

## 扩展开发

### 自定义损失函数

```python
from distillation_loss import DistillationLoss

class CustomDistillationLoss(DistillationLoss):
    def forward(self, student_logits, teacher_logits, labels, **kwargs):
        # 自定义实现
        dist_loss, hard_loss = super().forward(
            student_logits, teacher_logits, labels, **kwargs
        )
        # 添加自定义损失项
        custom_loss = self.compute_custom_loss(student_logits, teacher_logits)
        return dist_loss + custom_loss, hard_loss
```

### 自定义数据加载器

```python
from data_utils import StudentDataset

class CustomStudentDataset(StudentDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # 添加自定义数据增强
        item['augmented_data'] = self.augment_data(item)
        return item
```

## 许可证

本项目使用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。
