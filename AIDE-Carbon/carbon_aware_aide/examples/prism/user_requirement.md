As large language models (LLMs) continue to grow in size and deployment scale, efficiently serving multiple models on shared GPU infrastructure becomes increasingly critical. Modern LLM serving systems face the challenge of placing multiple models across a cluster of GPUs while managing limited memory resources and meeting strict performance requirements.

Each model has distinct characteristics: a memory footprint (model size), incoming request rate, and service level objective (SLO) for latency. The GPU cluster consists of K ≥ 1 GPUs, each with a fixed memory capacity (typically 80 GB). Multiple models can be co-located on the same GPU, but the total memory consumption cannot exceed the GPU's capacity.

We index GPUs by k ∈ {0, 1, ..., K-1}. Each model j is characterized by:
- **model_size_j**: Memory footprint in GB
- **req_rate_j**: Incoming request rate (requests per second)
- **slo_j**: Service level objective for latency (milliseconds)

## The KVCache Pressure Challenge

A critical bottleneck in LLM serving is the **KVCache** (key-value cache) memory required during inference. When multiple models share a GPU, they compete for the remaining memory after model weights are loaded. This remaining memory is used for KVCache, which directly impacts inference throughput and latency.

The **KVCache pressure (KVPR)** for a GPU k is defined as:

```
KVPR_k = (Σ_j req_rate_j / slo_j) / (GPU_MEM_SIZE - Σ_j model_size_j)
```

where the summations are over all models j placed on GPU k.

**Interpretation:**
- **Numerator**: The weighted request rate Σ(req_rate_j / slo_j) represents the aggregate demand on the GPU, where models with tighter SLOs contribute more pressure
- **Denominator**: The remaining memory (GPU_MEM_SIZE - Σ model_size_j) represents the available KVCache capacity
- **Higher KVPR** indicates more contention for KVCache, leading to increased latency and potential SLO violations

## Optimization Objective

The goal of an efficient placement algorithm is to minimize the **maximum KVPR** across all GPUs:

```
minimize max_k KVPR_k
```

This objective ensures that no single GPU becomes a bottleneck. A balanced placement distributes both memory consumption and weighted request rates across GPUs, preventing any GPU from being overloaded.

## Success Metrics

A successful placement algorithm should achieve:

1. **Feasibility**: All memory constraints satisfied, no GPU overcommitted
2. **Low KVPR**: Maximum KVPR significantly lower than naive baselines
3. **Robustness**: Consistent performance across diverse workload scenarios
4. **Efficiency**: Fast computation time enabling real-time placement decisions

The ultimate goal is to maximize cluster throughput while meeting SLOs for all models, enabling cost-effective and performant LLM serving at scale.
