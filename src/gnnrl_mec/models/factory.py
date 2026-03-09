from gnnrl_mec.models.mlp_policy import MLPPPOPolicy
from gnnrl_mec.models.policy import GraphPPOPolicy


def build_policy(cfg: dict, obs, num_devices: int, num_servers: int):
    model_name = cfg["model"].get("name", "gnn").lower()
    hidden_dim = int(cfg["model"]["hidden_dim"])
    num_actions = num_servers + 1
    if model_name == "mlp":
        return MLPPPOPolicy(
            num_nodes=obs.num_nodes,
            node_dim=obs.x.size(-1),
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_devices=num_devices,
        )
    return GraphPPOPolicy(
        node_dim=obs.x.size(-1),
        hidden_dim=hidden_dim,
        num_gnn_layers=int(cfg["model"]["gnn_layers"]),
        num_actions=num_actions,
        num_devices=num_devices,
        dropout=float(cfg["model"]["dropout"]),
    )
