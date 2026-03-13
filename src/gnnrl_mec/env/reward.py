def compute_reward(
    total_delay: float,
    total_energy: float,
    failures: int,
    time_weight: float,
    energy_weight: float,
    failure_penalty: float,
) -> float:
    weight_sum = time_weight + energy_weight
    normalized_cost = (time_weight * total_delay + energy_weight * total_energy) / max(weight_sum, 1e-6)
    return -0.01 * normalized_cost - failure_penalty * failures
