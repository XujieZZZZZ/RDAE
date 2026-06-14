#!/usr/bin/env python3
"""
LLM日志分析工具
用于分析和统计LLM调用日志
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def analyze_logs(log_dir: str = "logs/llm_calls"):
    """分析LLM调用日志"""
    log_path = Path(log_dir)
    summary_file = log_path / "summary.jsonl"

    if not summary_file.exists():
        print(f"日志文件不存在: {summary_file}")
        return

    # 统计数据
    total_calls = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    model_usage = defaultdict(int)
    error_count = 0
    errors = []

    print("=" * 60)
    print("LLM调用日志分析")
    print("=" * 60)

    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                total_calls += 1

                # 统计token使用
                usage = log.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

                # 统计模型使用
                model = log.get("input", {}).get("model", "unknown")
                model_usage[model] += 1

                # 统计错误
                if "error" in log.get("output", {}):
                    error_count += 1
                    errors.append({
                        "call_id": log.get("call_id"),
                        "timestamp": log.get("timestamp"),
                        "error": log["output"]["error"][:100]  # 只显示前100字符
                    })

            except json.JSONDecodeError as e:
                print(f"解析日志行失败: {e}")
                continue

    # 输出统计结果
    print(f"\n总调用次数: {total_calls}")
    print(f"成功调用: {total_calls - error_count}")
    print(f"失败调用: {error_count}")

    print(f"\n--- Token使用统计 ---")
    print(f"总Token数: {total_tokens:,}")
    print(f"输入Token数: {total_prompt_tokens:,}")
    print(f"输出Token数: {total_completion_tokens:,}")
    if total_calls > 0:
        print(f"平均每次调用Token数: {total_tokens / total_calls:.2f}")

    print(f"\n--- 模型使用统计 ---")
    for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        print(f"{model}: {count} 次 ({percentage:.1f}%)")

    if errors:
        print(f"\n--- 错误详情 (最近{min(5, len(errors))}条) ---")
        for error in errors[-5:]:
            print(f"Call ID {error['call_id']} @ {error['timestamp']}")
            print(f"  错误: {error['error']}")
            print()

    print("=" * 60)


def show_recent_calls(log_dir: str = "logs/llm_calls", n: int = 5):
    """显示最近的N次调用"""
    log_path = Path(log_dir)
    summary_file = log_path / "summary.jsonl"

    if not summary_file.exists():
        print(f"日志文件不存在: {summary_file}")
        return

    print("=" * 60)
    print(f"最近 {n} 次LLM调用")
    print("=" * 60)

    calls = []
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for call in calls[-n:]:
        print(f"\nCall ID: {call.get('call_id')}")
        print(f"时间: {call.get('timestamp')}")
        print(f"模型: {call.get('input', {}).get('model', 'unknown')}")

        usage = call.get('usage', {})
        print(f"Token使用: {usage.get('total_tokens', 0)} (输入: {usage.get('prompt_tokens', 0)}, 输出: {usage.get('completion_tokens', 0)})")

        # 显示输入的前100字符
        messages = call.get('input', {}).get('messages', [])
        if messages:
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            if user_msg:
                print(f"输入: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")

        # 显示输出的前100字符
        output = call.get('output', {})
        if 'content' in output:
            content = output['content']
            print(f"输出: {content[:100]}{'...' if len(content) > 100 else ''}")
        elif 'error' in output:
            print(f"错误: {output['error'][:100]}")

        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "recent":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            show_recent_calls(n=n)
        elif sys.argv[1] == "analyze":
            analyze_logs()
        else:
            print("用法:")
            print("  python analyze_llm_logs.py analyze  - 分析所有日志")
            print("  python analyze_llm_logs.py recent [N]  - 显示最近N次调用（默认5次）")
    else:
        # 默认显示分析
        analyze_logs()
