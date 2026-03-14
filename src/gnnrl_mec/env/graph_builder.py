import torch
from torch_geometric.data import Data


def build_bipartite_graph(
    device_features: list[list[float]],
    server_features: list[list[float]],
    edge_scores: list[list[float]] | None = None,
    max_links_per_device: int | None = None,
) -> Data:
    x = torch.tensor(device_features + server_features, dtype=torch.float32)
    num_devices = len(device_features)
    num_servers = len(server_features)
    if edge_scores is None:
        edge_scores = [[1.0 for _ in range(num_servers)] for _ in range(num_devices)]
    link_cap = num_servers if max_links_per_device is None else max(1, min(max_links_per_device, num_servers))
    edge_pairs: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for device_idx in range(num_devices):
        ranked_servers = sorted(range(num_servers), key=lambda s: edge_scores[device_idx][s], reverse=True)
        for server_idx in ranked_servers[:link_cap]:
            src = device_idx
            dst = num_devices + server_idx
            edge_pairs.append([src, dst])
            edge_pairs.append([dst, src])
            score = float(edge_scores[device_idx][server_idx])
            edge_attr.append([score, float(server_idx + 1)])
            edge_attr.append([score, float(server_idx + 1)])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
    data.num_devices = num_devices
    data.num_servers = num_servers
    data.device_node_mask = torch.zeros(x.size(0), dtype=torch.bool)
    data.device_node_mask[:num_devices] = True
    return data
