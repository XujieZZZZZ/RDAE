GPU_MEM_SIZE = 80  # GB


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    # Phase 1: Initial placement using first-fit decreasing by model size
    sorted_models = sorted(models, key=lambda m: m.model_size, reverse=True)
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    gpu_remaining = [GPU_MEM_SIZE] * gpu_num
    gpu_pressure = [0.0] * gpu_num

    for model in sorted_models:
        placed = False
        best_gpu = None
        best_remaining = -1
        for gpu_id in range(gpu_num):
            if (
                model.model_size <= gpu_remaining[gpu_id]
                and gpu_remaining[gpu_id] > best_remaining
            ):
                best_remaining = gpu_remaining[gpu_id]
                best_gpu = gpu_id
        if best_gpu is None:
            raise ValueError(
                f"Unable to place model {model.model_name} of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {gpu_remaining}"
            )
        placement[best_gpu].append(model)
        gpu_remaining[best_gpu] -= model.model_size
        gpu_pressure[best_gpu] += model.req_rate / model.slo

    # Handle single GPU case
    if gpu_num == 1:
        return placement

    # Helper function to compute KVPR for a GPU
    def compute_gpu_kvpr(gpu_id):
        total_memory = sum(m.model_size for m in placement[gpu_id])
        remaining_memory = GPU_MEM_SIZE - total_memory
        if remaining_memory <= 0:
            return float("inf")
        return gpu_pressure[gpu_id] / remaining_memory

    # Phase 2: Priority-queue based refinement
    # Sort models by pressure density (req_rate/slo / model_size) descending
    models_by_density = sorted(
        models,
        key=lambda m: (
            (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else float("inf")
        ),
        reverse=True,
    )

    # Re-initialize placement with empty GPUs
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    gpu_remaining = [GPU_MEM_SIZE] * gpu_num
    gpu_pressure = [0.0] * gpu_num

    # Use a min-heap priority queue for GPU selection based on current KVPR
    import heapq

    # Initialize heap with (kvpr, gpu_id) for each GPU
    heap = []
    for gpu_id in range(gpu_num):
        # KVPR is 0/GPU_MEM_SIZE = 0 initially
        heapq.heappush(heap, (0.0, gpu_id))

    for model in models_by_density:
        model_size = model.model_size
        model_pressure = model.req_rate / model.slo

        # Try to find a GPU with lowest KVPR that can fit the model
        found = False
        # Collect candidates from heap
        candidates = []
        while heap and not found:
            kvpr, gpu_id = heapq.heappop(heap)
            if model_size <= gpu_remaining[gpu_id]:
                # Assign model to this GPU
                placement[gpu_id].append(model)
                gpu_remaining[gpu_id] -= model_size
                gpu_pressure[gpu_id] += model_pressure
                # Compute new KVPR for this GPU
                remaining_memory = GPU_MEM_SIZE - sum(
                    m.model_size for m in placement[gpu_id]
                )
                if remaining_memory <= 0:
                    new_kvpr = float("inf")
                else:
                    new_kvpr = gpu_pressure[gpu_id] / remaining_memory
                heapq.heappush(heap, (new_kvpr, gpu_id))
                found = True
                break
            else:
                candidates.append((kvpr, gpu_id))

        if not found:
            # If no GPU can fit, try to place on any GPU with remaining capacity
            for gpu_id in range(gpu_num):
                if model_size <= gpu_remaining[gpu_id]:
                    placement[gpu_id].append(model)
                    gpu_remaining[gpu_id] -= model_size
                    gpu_pressure[gpu_id] += model_pressure
                    remaining_memory = GPU_MEM_SIZE - sum(
                        m.model_size for m in placement[gpu_id]
                    )
                    if remaining_memory <= 0:
                        new_kvpr = float("inf")
                    else:
                        new_kvpr = gpu_pressure[gpu_id] / remaining_memory
                    heapq.heappush(heap, (new_kvpr, gpu_id))
                    found = True
                    break

        if not found:
            raise ValueError(
                f"Unable to place model {model.model_name} of size {model_size} GB on any GPU."
            )

        # Push back candidates that were not used
        for kvpr, gpu_id in candidates:
            heapq.heappush(heap, (kvpr, gpu_id))

    # Phase 3: Single-pass swap optimization
    # Compute current KVPR for all GPUs
    gpu_kvpr = [compute_gpu_kvpr(gpu_id) for gpu_id in range(gpu_num)]
    current_max_kvpr = max(gpu_kvpr)

    # Try swaps between all pairs of GPUs
    improved = True
    while improved:
        improved = False
        for i in range(gpu_num):
            for j in range(i + 1, gpu_num):
                # Only consider pairs where one has significantly higher KVPR
                if abs(gpu_kvpr[i] - gpu_kvpr[j]) < 0.001 * current_max_kvpr:
                    continue

                # Try swapping models between GPU i and GPU j
                models_i = placement[i]
                models_j = placement[j]

                best_swap = None
                best_swap_max = current_max_kvpr

                for model_i in models_i:
                    if best_swap:
                        break
                    for model_j in models_j:
                        # Check memory feasibility after swap
                        mem_i_after = (
                            sum(m.model_size for m in models_i)
                            - model_i.model_size
                            + model_j.model_size
                        )
                        mem_j_after = (
                            sum(m.model_size for m in models_j)
                            - model_j.model_size
                            + model_i.model_size
                        )

                        if mem_i_after <= GPU_MEM_SIZE and mem_j_after <= GPU_MEM_SIZE:
                            # Compute new pressures
                            pressure_i_after = (
                                gpu_pressure[i]
                                - (model_i.req_rate / model_i.slo)
                                + (model_j.req_rate / model_j.slo)
                            )
                            pressure_j_after = (
                                gpu_pressure[j]
                                - (model_j.req_rate / model_j.slo)
                                + (model_i.req_rate / model_i.slo)
                            )

                            rem_i = GPU_MEM_SIZE - mem_i_after
                            rem_j = GPU_MEM_SIZE - mem_j_after

                            if rem_i > 0 and rem_j > 0:
                                kvpr_i_after = pressure_i_after / rem_i
                                kvpr_j_after = pressure_j_after / rem_j

                                new_max = max(kvpr_i_after, kvpr_j_after)
                                for g in range(gpu_num):
                                    if g != i and g != j:
                                        new_max = max(new_max, gpu_kvpr[g])

                                if new_max < best_swap_max:
                                    best_swap_max = new_max
                                    best_swap = (model_i, model_j)
                                    break

                if best_swap:
                    model_i, model_j = best_swap
                    # Execute swap
                    placement[i].remove(model_i)
                    placement[j].remove(model_j)
                    placement[i].append(model_j)
                    placement[j].append(model_i)

                    # Update pressures
                    gpu_pressure[i] = (
                        gpu_pressure[i]
                        - (model_i.req_rate / model_i.slo)
                        + (model_j.req_rate / model_j.slo)
                    )
                    gpu_pressure[j] = (
                        gpu_pressure[j]
                        - (model_j.req_rate / model_j.slo)
                        + (model_i.req_rate / model_i.slo)
                    )

                    # Update remaining memory
                    gpu_remaining[i] = GPU_MEM_SIZE - sum(
                        m.model_size for m in placement[i]
                    )
                    gpu_remaining[j] = GPU_MEM_SIZE - sum(
                        m.model_size for m in placement[j]
                    )

                    # Update KVPR values
                    gpu_kvpr[i] = compute_gpu_kvpr(i)
                    gpu_kvpr[j] = compute_gpu_kvpr(j)
                    current_max_kvpr = max(gpu_kvpr)
                    improved = True
                    break  # Restart from beginning after a swap

    return placement