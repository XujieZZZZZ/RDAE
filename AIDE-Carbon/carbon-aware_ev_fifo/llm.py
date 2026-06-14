# llm.py
import os
import yaml
import random
import requests


with open("config.yaml", "r", encoding="utf-8") as f:
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



def call_llm(
    prompt: str,
    system_message: str | None = None,
):
    model = choose_model()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    # print("==== MODEL INPUT ====")
    # print("System message:")
    # print(system_message)
    # print("User message / Prompt:")
    # print(prompt)
    # print("Prompt length (tokens approx):", len(prompt.split()))
    # print("====================")


    if resp.status_code != 200:
        raise RuntimeError(f"LLM API error: {resp.status_code}\n{resp.text}")

    return resp.json()["choices"][0]["message"]["content"]
