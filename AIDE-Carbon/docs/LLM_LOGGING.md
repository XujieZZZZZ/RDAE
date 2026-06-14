# LLM调用日志记录功能

## 概述

LLM调用已经升级为使用 `LLMClass` 类，自动记录所有LLM调用的输入、输出和usage信息。

## 功能特性

### 1. 自动日志记录
- **输入记录**: 记录模型名称、消息内容、温度、top_p等参数
- **输出记录**: 记录LLM返回的内容和finish_reason
- **Usage记录**: 记录token使用情况（prompt_tokens, completion_tokens, total_tokens）
- **错误记录**: 记录API错误和异常信息

### 2. 日志存储格式

日志保存在 `logs/llm_calls/` 目录下：

#### 单独的调用日志
每次调用生成一个独立的JSON文件：
```
logs/llm_calls/call_20260526_143022_1.json
```

文件内容示例：
```json
{
  "call_id": 1,
  "timestamp": "20260526_143022",
  "input": {
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 4096
  },
  "output": {
    "content": "Hello! How can I help you today?",
    "finish_reason": "stop"
  },
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

#### 汇总日志
所有调用追加到 `summary.jsonl` 文件中（每行一个JSON对象）：
```
logs/llm_calls/summary.jsonl
```

## 使用方法

### 方法1: 使用原有的 call_llm 函数（推荐）

所有现有代码无需修改，自动启用日志记录：

```python
from llm import call_llm

response = call_llm(
    prompt="Your question here",
    system_message="You are a helpful assistant."
)
```

### 方法2: 直接使用 LLMClass

如果需要更多控制（如禁用日志）：

```python
from llm import LLMClass

# 启用日志（默认）
llm = LLMClass(enable_logging=True)
response = llm.call(prompt="Your question", system_message="System message")

# 禁用日志
llm_no_log = LLMClass(enable_logging=False)
response = llm_no_log.call(prompt="Your question")
```

### 方法3: 使用全局实例

```python
from llm import llm_class

response = llm_class.call(
    prompt="Your question",
    system_message="System message"
)

# 查看调用次数
print(f"Total calls: {llm_class.logger.call_count}")
```

## 代码结构

### LLMLogger 类
负责日志的写入和管理：
- `log_call()`: 记录单次调用
- 自动创建日志目录
- 生成唯一的调用ID和时间戳

### LLMClass 类
封装LLM调用逻辑：
- `call()`: 执行LLM调用并记录日志
- 支持启用/禁用日志
- 完整的错误处理和日志记录

### call_llm 函数
向后兼容的函数接口：
- 内部使用全局 `llm_class` 实例
- 保持原有API不变
- 自动启用日志记录

## 日志分析

### 读取单个调用日志
```python
import json

with open("logs/llm_calls/call_20260526_143022_1.json") as f:
    log = json.load(f)
    print(f"Model: {log['input']['model']}")
    print(f"Tokens used: {log['usage']['total_tokens']}")
```

### 读取汇总日志
```python
import json

with open("logs/llm_calls/summary.jsonl") as f:
    for line in f:
        log = json.loads(line)
        print(f"Call {log['call_id']}: {log['usage']['total_tokens']} tokens")
```

### 统计token使用
```python
import json

total_tokens = 0
with open("logs/llm_calls/summary.jsonl") as f:
    for line in f:
        log = json.loads(line)
        total_tokens += log['usage'].get('total_tokens', 0)

print(f"Total tokens used: {total_tokens}")
```

## 注意事项

1. **日志目录**: 默认为 `logs/llm_calls/`，会自动创建
2. **磁盘空间**: 长时间运行会积累大量日志文件，建议定期清理
3. **敏感信息**: 日志包含完整的输入输出，注意保护敏感数据
4. **性能影响**: 日志写入是同步的，对性能影响很小

## 测试

运行测试脚本验证功能：
```bash
python test_llm_logging.py
```

## 兼容性

- ✅ 所有现有代码无需修改
- ✅ 保持原有API接口
- ✅ 向后完全兼容
- ✅ 可选的日志功能
