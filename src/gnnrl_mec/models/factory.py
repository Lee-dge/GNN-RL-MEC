from gnnrl_mec.models.mlp_policy import MLPContinuousPPOPolicy, MLPMixContinuousPPOPolicy, MLPPPOPolicy
from gnnrl_mec.models.policy import GraphContinuousPPOPolicy, GraphMixContinuousPPOPolicy, GraphPPOPolicy


def build_policy(cfg: dict, obs, num_devices: int, num_servers: int):
    model_name = cfg["model"].get("name", "gnn").lower()
    action_type = cfg["env"].get("action_type", "discrete").lower()
    hidden_dim = int(cfg["model"]["hidden_dim"])
    num_actions = num_servers + 1
    is_continuous = action_type == "continuous_ratio"
    is_mix_continuous = action_type == "continuous_mix"
    if model_name == "mlp":
        if is_mix_continuous:
            return MLPMixContinuousPPOPolicy(
                num_nodes=obs.num_nodes,
                node_dim=obs.x.size(-1),
                hidden_dim=hidden_dim,
                num_devices=num_devices,
                num_servers=num_servers,
            )
        if is_continuous:
            return MLPContinuousPPOPolicy(
                num_nodes=obs.num_nodes,
                node_dim=obs.x.size(-1),
                hidden_dim=hidden_dim,
                num_devices=num_devices,
            )
        return MLPPPOPolicy(
            num_nodes=obs.num_nodes,
            node_dim=obs.x.size(-1),
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_devices=num_devices,
        )
    if is_mix_continuous:
        return GraphMixContinuousPPOPolicy(
            node_dim=obs.x.size(-1),
            hidden_dim=hidden_dim,
            num_gnn_layers=int(cfg["model"]["gnn_layers"]),
            num_devices=num_devices,
            num_servers=num_servers,
            dropout=float(cfg["model"]["dropout"]),
        )
    if is_continuous:
        return GraphContinuousPPOPolicy(
            node_dim=obs.x.size(-1),
            hidden_dim=hidden_dim,
            num_gnn_layers=int(cfg["model"]["gnn_layers"]),
            num_devices=num_devices,
            dropout=float(cfg["model"]["dropout"]),
        )
    return GraphPPOPolicy(
        node_dim=obs.x.size(-1),
        hidden_dim=hidden_dim,
        num_gnn_layers=int(cfg["model"]["gnn_layers"]),
        num_actions=num_actions,
        num_devices=num_devices,
        dropout=float(cfg["model"]["dropout"]),
    )
