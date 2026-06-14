# llm.py
import os
import yaml
import random
import requests
import json
from datetime import datetime
from pathlib import Path


with open("examples/carbon_aware_dag/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

llm_cfg = config["llm"]

API_BASE = llm_cfg["api_base"].rstrip("/")
TEMPERATURE = llm_cfg.get("temperature", 0.7)
TOP_P = llm_cfg.get("top_p", 0.95)
MAX_TOKENS = llm_cfg.get("max_tokens", 4096)
TIMEOUT = llm_cfg.get("timeout", 600)

PRIMARY_MODEL = llm_cfg["primary_model"]
SECONDARY_MODEL = llm_cfg.get("secondary_model")
PRIMARY_WEIGHT = llm_cfg.get("primary_model_weight", 1.0)
SECONDARY_WEIGHT = llm_cfg.get("secondary_model_weight", 0.0)

# API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = "sk-61165e466cdf4187891ea15dd27cf062"
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")


def choose_model():
    if SECONDARY_MODEL and SECONDARY_WEIGHT > 0:
        return random.choices(
            [PRIMARY_MODEL, SECONDARY_MODEL],
            weights=[PRIMARY_WEIGHT, SECONDARY_WEIGHT],
        )[0]
    return PRIMARY_MODEL


class LLMLogger:
    """LLM调用日志记录器"""

    def __init__(self, log_dir: str = "logs/llm_calls"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def log_call(self, input_data: dict, output_data: dict, usage_data: dict):
        """记录单次LLM调用"""
        self.call_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "call_id": self.call_count,
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data,
            "usage": usage_data
        }

        # 保存到单独的JSON文件
        log_file = self.log_dir / f"call_{timestamp}_{self.call_count}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        # 同时追加到汇总日志
        summary_file = self.log_dir / "summary.jsonl"
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry


class LLMClass:
    """LLM调用封装类，支持日志记录"""

    def __init__(self, enable_logging: bool = True):
        self.logger = LLMLogger() if enable_logging else None
        self.api_base = API_BASE
        self.api_key = API_KEY
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.max_tokens = MAX_TOKENS
        self.timeout = TIMEOUT

    def call(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> str:
        """调用LLM并记录日志"""
        model = choose_model()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if isinstance(prompt, dict):
            prompt_str = ""
            for key, value in prompt.items():
                prompt_str += f"{key}: {value}\n"
            prompt = prompt_str
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        # 记录输入
        input_data = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            error_msg = f"LLM API error: {resp.status_code}\n{resp.text}"
            if self.logger:
                self.logger.log_call(
                    input_data=input_data,
                    output_data={"error": error_msg},
                    usage_data={}
                )
            raise RuntimeError(error_msg)

        try:
            response_json = resp.json()
        except Exception as e:
            error_msg = f"Failed to parse LLM response as JSON: {e}\nResponse text: {resp.text}"
            if self.logger:
                self.logger.log_call(
                    input_data=input_data,
                    output_data={"error": error_msg},
                    usage_data={}
                )
            raise RuntimeError(error_msg)

        if "choices" not in response_json:
            error_msg = (
                f"LLM API returned unexpected format. Missing 'choices' key.\n"
                f"Response keys: {list(response_json.keys())}\n"
                f"Full response: {response_json}"
            )
            if self.logger:
                self.logger.log_call(
                    input_data=input_data,
                    output_data={"error": error_msg},
                    usage_data={}
                )
            raise RuntimeError(error_msg)

        if not response_json["choices"]:
            error_msg = f"LLM API returned empty choices array: {response_json}"
            if self.logger:
                self.logger.log_call(
                    input_data=input_data,
                    output_data={"error": error_msg},
                    usage_data={}
                )
            raise RuntimeError(error_msg)

        output_content = response_json["choices"][0]["message"]["content"]
        usage_data = response_json.get("usage", {})

        # 记录输出和usage
        if self.logger:
            self.logger.log_call(
                input_data=input_data,
                output_data={
                    "content": output_content,
                    "finish_reason": response_json["choices"][0].get("finish_reason"),
                },
                usage_data=usage_data
            )

        return output_content


# 全局LLM实例
llm_class = LLMClass(enable_logging=True)


def call_llm(
    prompt: str,
    system_message: str | None = None,
) -> str:
    """
    保持向后兼容的call_llm函数，内部使用LLMClass
    """
    return llm_class.call(prompt=prompt, system_message=system_message)
