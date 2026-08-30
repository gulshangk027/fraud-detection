"""
NetworkX Graph Topology and Mule Network Analysis Service for MuleNet AI.
"""

import networkx as nx
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.dataset_service import get_current_dataset
from app.services.ml_service import predict_account, get_model_status, _TRAINED_MODELS, _ACTIVE_MODE

logger = logging.getLogger(__name__)

def generate_mule_network_graph(n_nodes: int = 40, seed: int = 42) -> Dict[str, Any]:
    """Generates account relationship graph with realistic mule topologies (Star, Chain, Fan-out, Fan-in) tied to dataset accounts."""
    random.seed(seed)
    np.random.seed(seed)
    
    df = get_current_dataset()
    if len(df) > 0 and "ACCOUNT_ID" in df.columns:
        dataset_accs = df["ACCOUNT_ID"].head(n_nodes).tolist()
    else:
        dataset_accs = [f"ACC-{100000 + i}" for i in range(n_nodes)]
        
    n_actual = min(len(dataset_accs), n_nodes)
    node_ids = dataset_accs[:n_actual]
    
    # Map predictions for each node
    account_scores = {}
    for aid in node_ids:
        try:
            res = predict_account(aid, mode="full_feature")
            if res.get("found"):
                account_scores[aid] = {
                    "risk_score": res["risk_score"],
                    "risk_level": res.get("risk_level", "LOW"),
                    "classification": res.get("classification", "LEGITIMATE"),
                    "probability": res.get("model_probability", 0.0)
                }
            else:
                account_scores[aid] = {"risk_score": 10.0, "risk_level": "LOW", "classification": "LEGITIMATE", "probability": 0.10}
        except Exception:
            account_scores[aid] = {"risk_score": 10.0, "risk_level": "LOW", "classification": "LEGITIMATE", "probability": 0.10}

    G = nx.DiGraph()
    for nid in node_ids:
        info = account_scores[nid]
        r_score = info["risk_score"]
        G.add_node(
            nid, 
            risk_score=r_score, 
            risk_level=info["risk_level"],
            classification=info["classification"],
            probability=info["probability"],
            is_mule=r_score >= 50.0
        )

    # Topological Hub assignments based on scored risks
    mule_accs = [n for n in node_ids if G.nodes[n]["is_mule"]]
    legit_accs = [n for n in node_ids if not G.nodes[n]["is_mule"]]
    
    if len(mule_accs) >= 4:
        hub_star = mule_accs[0]
        hub_fanout = mule_accs[1]
        chain_nodes = mule_accs[2:6] if len(mule_accs) >= 6 else mule_accs[2:]
    else:
        hub_star = node_ids[0]
        hub_fanout = node_ids[1]
        chain_nodes = node_ids[2:5]

    base_date = datetime.now()

    # 1. Star / Fan-In Topology: multiple feeder accounts -> hub_star
    feeders = [n for n in node_ids if n != hub_star][:7]
    for idx, f in enumerate(feeders):
        amt = round(float(np.random.uniform(12000, 65000)), 2)
        tx_cnt = random.randint(3, 14)
        d_str = (base_date - timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M")
        G.add_edge(f, hub_star, amount=amt, transactions=tx_cnt, pattern="FAN-IN", last_date=d_str)

    # 2. Fan-Out Topology: hub_fanout -> multiple receiver accounts
    receivers = [n for n in node_ids if n != hub_fanout and n not in feeders][:6]
    for idx, r in enumerate(receivers):
        amt = round(float(np.random.uniform(15000, 95000)), 2)
        tx_cnt = random.randint(2, 12)
        d_str = (base_date - timedelta(hours=random.randint(1, 36))).strftime("%Y-%m-%d %H:%M")
        G.add_edge(hub_fanout, r, amount=amt, transactions=tx_cnt, pattern="FAN-OUT", last_date=d_str)

    # 3. Chain Topology: A -> B -> C -> D
    for i in range(len(chain_nodes) - 1):
        amt = round(float(np.random.uniform(35000, 140000)), 2)
        tx_cnt = random.randint(4, 18)
        d_str = (base_date - timedelta(hours=random.randint(1, 24))).strftime("%Y-%m-%d %H:%M")
        G.add_edge(chain_nodes[i], chain_nodes[i+1], amount=amt, transactions=tx_cnt, pattern="CHAIN", last_date=d_str)

    # 4. Standard transactions between remaining nodes
    other_nodes = [n for n in node_ids if n not in chain_nodes]
    for _ in range(min(30, len(other_nodes) * 2)):
        u, v = random.sample(other_nodes, 2)
        if u != v and not G.has_edge(u, v):
            amt = round(float(np.random.uniform(1500, 12000)), 2)
            tx_cnt = random.randint(1, 5)
            d_str = (base_date - timedelta(hours=random.randint(5, 72))).strftime("%Y-%m-%d %H:%M")
            G.add_edge(u, v, amount=amt, transactions=tx_cnt, pattern="STANDARD", last_date=d_str)

    # PageRank computation
    try:
        pagerank = nx.pagerank(G)
    except Exception:
        pagerank = {n: round(1.0/n_actual, 4) for n in G.nodes()}

    # Cluster Identification (Connected Components)
    components = list(nx.weakly_connected_components(G))
    cluster_map = {}
    clusters_data = []
    
    for c_idx, comp in enumerate(components):
        c_id = f"Mule Network #{c_idx+1:02d}"
        c_nodes = list(comp)
        c_subgraph = G.subgraph(c_nodes)
        
        c_susp = sum(1 for n in c_nodes if G.nodes[n]["is_mule"])
        
        # Determine primary cluster topology
        topos = []
        for n in c_nodes:
            if G.in_degree(n) >= 4 and G.out_degree(n) >= 4:
                topos.append("STAR")
            elif G.in_degree(n) >= 3:
                topos.append("FAN-IN")
            elif G.out_degree(n) >= 3:
                topos.append("FAN-OUT")
            elif G.in_degree(n) >= 1 and G.out_degree(n) >= 1:
                topos.append("CHAIN")
        
        main_topo = max(set(topos), key=topos.count) if topos else "STANDARD"
        
        clusters_data.append({
            "cluster_id": c_id,
            "title": f"Mule Cluster #{c_idx+1:02d}",
            "node_count": len(c_nodes),
            "edge_count": c_subgraph.number_of_edges(),
            "suspicious_count": c_susp,
            "primary_topology": main_topo,
            "node_ids": c_nodes
        })
        
        for n in c_nodes:
            cluster_map[n] = c_id

    # Node data assembly
    nodes_data = []
    for n in G.nodes(data=True):
        nid = n[0]
        attr = n[1]
        
        in_deg = G.in_degree(nid)
        out_deg = G.out_degree(nid)
        
        in_amt = sum(G[u][nid].get("amount", 0.0) for u in G.predecessors(nid))
        out_amt = sum(G[nid][v].get("amount", 0.0) for v in G.successors(nid))
        
        # Topology pattern classification
        if in_deg >= 4 and out_deg >= 4:
            primary_topo = "STAR"
            topo_expl = "One central account maintains multiple direct incoming and outgoing transaction relationships with surrounding accounts."
        elif in_deg >= 3 and out_deg <= 1:
            primary_topo = "FAN-IN"
            topo_expl = "Multiple accounts are sending funds into this account, indicating a potential aggregation pattern."
        elif out_deg >= 3 and in_deg <= 1:
            primary_topo = "FAN-OUT"
            topo_expl = "Account demonstrates a fan-out transaction pattern with outgoing transfers to multiple beneficiary accounts."
        elif in_deg >= 1 and out_deg >= 1:
            primary_topo = "CHAIN"
            topo_expl = "Funds move sequentially through this transit account to downstream beneficiary destinations."
        else:
            primary_topo = "REGULAR"
            topo_expl = "No significant mule-network topology detected. Transaction behavior appears normal."

        neighbors = list(set(list(G.predecessors(nid)) + list(G.successors(nid))))
        susp_neighbors = sum(1 for nb in neighbors if G.nodes[nb].get("risk_score", 0) >= 50.0)

        nodes_data.append({
            "id": nid,
            "label": nid,
            "risk_score": attr.get("risk_score", 0.0),
            "risk_level": attr.get("risk_level", "LOW"),
            "classification": attr.get("classification", "LEGITIMATE"),
            "probability": attr.get("probability", 0.0),
            "is_mule": attr.get("is_mule", False),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "pagerank": round(float(pagerank.get(nid, 0.0)), 4),
            "incoming_amount": round(in_amt, 2),
            "outgoing_amount": round(out_amt, 2),
            "total_transactions": in_deg + out_deg,
            "connected_accounts_count": len(neighbors),
            "suspicious_neighbors": susp_neighbors,
            "primary_topology": primary_topo,
            "topology_explanation": topo_expl,
            "cluster_id": cluster_map.get(nid, "Mule Network #01")
        })

    # Edge data assembly
    edges_data = []
    suspicious_edges = 0
    for u, v, attr in G.edges(data=True):
        u_risk = G.nodes[u].get("risk_score", 0.0)
        v_risk = G.nodes[v].get("risk_score", 0.0)
        is_susp = u_risk >= 50.0 or v_risk >= 50.0
        if is_susp:
            suspicious_edges += 1
            
        edges_data.append({
            "id": f"e_{u}_{v}",
            "source": u,
            "target": v,
            "amount": attr.get("amount", 0.0),
            "transactions": attr.get("transactions", 1),
            "last_date": attr.get("last_date", "2026-08-21 12:00"),
            "pattern": attr.get("pattern", "STANDARD"),
            "is_suspicious": is_susp
        })

    susp_nodes_count = sum(1 for n in nodes_data if n["is_mule"])
    legit_nodes_count = len(nodes_data) - susp_nodes_count

    return {
        "summary": {
            "total_accounts": len(nodes_data),
            "suspicious_accounts": susp_nodes_count,
            "legitimate_accounts": legit_nodes_count,
            "total_connections": len(edges_data),
            "suspicious_connections": suspicious_edges,
            "detected_mule_networks": len(clusters_data)
        },
        "clusters": clusters_data,
        "detected_topologies": ["STAR", "CHAIN", "FAN-OUT", "FAN-IN", "REGULAR"],
        "nodes": nodes_data,
        "edges": edges_data
    }
