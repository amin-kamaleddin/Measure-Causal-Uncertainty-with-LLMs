#!/usr/bin/env python3
"""Read existing OpenRouter GraphAnswer records, compare proposed edges offline.

No network or secrets. Input JSONs remain at their original private paths.
All results are descriptive; no causal graph truth, identification or effect is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import networkx as nx
from sharecausal.llm.schemas import GraphAnswer


def _private_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError(f"Expected a JSON record in {path.name}")
    mid = item.get("model_requested")
    if not isinstance(mid, str) or not mid.strip():
        raise ValueError(f"Missing model_requested in {path.name}")
    if item.get("prompt_variant") != "base":
        raise ValueError(f"Non-base prompt variant in {path.name}")
    if not isinstance(item.get("graph"), dict):
        raise ValueError(f"Missing validated graph in {path.name}")
    graph = GraphAnswer.model_validate(item["graph"])
    usage = item.get("usage")
    if not isinstance(usage, dict) or usage.get("cost") is None:
        raise ValueError(f"Missing usage.cost in {path.name}")
    cost = float(usage["cost"])
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"Invalid usage.cost in {path.name}")
    return {
        "record": item,
        "graph": graph,
        "model": mid,
        "cost": cost,
        "source_file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "edge_set": {(e.source, e.target) for e in graph.edges},
    }


def _is_inside_git_repo(path: Path) -> bool:
    return any((p / ".git").exists() for p in (path, *path.parents))


def run(paths: list[Path], results_dir: Path) -> dict:
    if len(paths) != 4:
        raise ValueError("Supply exactly four explicit --record files")
    root = results_dir.resolve()
    if _is_inside_git_repo(root):
        raise ValueError("Refusing to save provider-derived reports inside a Git repository")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory")
    os.makedirs(root, mode=0o700, exist_ok=True)
    root.chmod(0o700)

    records = [_load(p.expanduser()) for p in paths]
    ids = [r["model"] for r in records]
    if len(set(ids)) != 4:
        raise ValueError(f"Expected distinct model IDs; got {ids}")

    prompts = {r["record"].get("input_sha256") for r in records}
    schemas = {r["record"].get("schema_sha256") for r in records}
    prompt_equal = len(prompts) == 1 and None not in prompts
    schema_equal = len(schemas) == 1 and None not in schemas

    model_rows = []
    provenance = []
    for r in records:
        d, g = r["record"], r["graph"]
        model_rows.append({
            "model": r["model"],
            "abstained": g.abstain,
            "n_proposed_edges": len(g.edges),
            "abstention_reason": g.abstention_reason,
            "source_file": r["source_file"],
            "source_sha256": r["source_sha256"],
            "requested_model": d["model_requested"],
            "returned_model": d.get("model"),
            "response_id": d.get("provider_response_id"),
            "prompt_sha256": d.get("input_sha256"),
            "schema_sha256": d.get("schema_sha256"),
            "prompt_variant": d.get("prompt_variant"),
            "seed_sent": d.get("seed_sent"),
            "temperature_sent": d.get("temperature_sent"),
            "reasoning_effort_sent": d.get("reasoning_effort_sent"),
            "max_tokens_sent": d.get("max_tokens_sent"),
            "finish_reason": d.get("finish_reason"),
            "usage_cost_usd": r["cost"],
        })
        for edge in g.edges:
            provenance.append({
                "model": r["model"],
                "arrow": f"{edge.source}->{edge.target}",
                "causal_role": edge.causal_role,
                "temporal_rationale": edge.temporal_rationale,
                "evidence_reference": edge.evidence_reference,
                "uncertainty": edge.uncertainty,
                "model_abstained_on_conclusion": g.abstain,
            })

    proposing = [r for r in records if r["edge_set"]]
    all_edges = sorted(set().union(*(r["edge_set"] for r in records)))
    edge_rows = []
    votes = Counter(edge for r in proposing for edge in r["edge_set"])
    for edge in all_edges:
        row = {
            "arrow": f"{edge[0]}->{edge[1]}",
            "n_proposing_models": votes[edge],
            "n_nonempty_graphs": len(proposing),
            "proposal_fraction_nonempty": round(votes[edge] / len(proposing), 6) if proposing else None,
            "n_all_model_responses": len(records),
        }
        for r in records:
            # Abstention with no proposed graph is missing edge evidence, NOT an edge-absence vote.
            row[r["model"]] = "NOT_PROPOSED_ABSTAIN" if not r["edge_set"] and r["graph"].abstain else int(edge in r["edge_set"])
        edge_rows.append(row)

    overlaps = []
    for i, first in enumerate(records):
        for second in records[i+1:]:
            missing_graph = (not first["edge_set"] and first["graph"].abstain) or (not second["edge_set"] and second["graph"].abstain)
            if missing_graph:
                j = None
                reason = "At least one model abstained without supplying a graph"
            else:
                a, b = first["edge_set"], second["edge_set"]
                j = len(a & b) / len(a | b) if (a | b) else 1.0
                reason = "Descriptive overlap; not causal accuracy"
            overlaps.append({"model_1": first["model"], "model_2": second["model"], "jaccard": j, "note": reason})

    majority = sorted(edge for edge, count in votes.items() if count > len(proposing) / 2)
    majority_acyclic = nx.is_directed_acyclic_graph(nx.DiGraph(majority))
    summary = {
        "analysis_kind": "exploratory_SH​ARE_domain_llm_graph_comparison".replace("\u200b", ""),
        "real_llm_response_records": len(records),
        "n_abstained_on_conclusion": sum(r["graph"].abstain for r in records),
        "n_with_proposed_edges": len(proposing),
        "n_unique_proposed_edges": len(all_edges),
        "reported_pilot_cost_usd": sum(r["cost"] for r in records),
        "identical_prompt_hashes": prompt_equal,
        "identical_schema_hashes": schema_equal,
        "models": model_rows,
        "descriptive_nonempty_proposal_majority": [f"{a}->{b}" for a,b in majority],
        "proposal_majority_is_acyclic": majority_acyclic,
        "warnings": [
            "These outputs are proposal frequencies, not causal truth probabilities or evidence support.",
            "Abstaining models may still propose edges; abstention is not an absence vote.",
            "An abstention with zero edges is not counted as a graph of zero causal links.",
            "One response per model is exploratory: no graph accuracy or relative reliability can be established.",
            "Output-token and reasoning settings may differ across arms; review model_summary.csv.",
            "No SHARE data or outcome statistics were used in this comparison.",
            "No causal identification, adjustment sets, or observational effect estimates are determined here.",
        ],
    }
    if not prompt_equal:
        summary["warnings"].append("PROMPT HASH MISMATCH or missing: verify arms had identical instructions before interpreting disagreement.")
    if not schema_equal:
        summary["warnings"].append("SCHEMA HASH MISMATCH or missing: verify schema comparability.")
    summary["pairwise_jaccard"] = overlaps

    _write_csv(root / "model_summary.csv", model_rows, list(model_rows[0]))
    _write_csv(root / "edge_proposals.csv", edge_rows, list(edge_rows[0]) if edge_rows else ["arrow", "n_proposing_models", "n_nonempty_graphs", "proposal_fraction_nonempty", "n_all_model_responses"] + ids)
    _write_csv(root / "edge_provenance_PRIVATE.csv", provenance, ["model", "arrow", "causal_role", "temporal_rationale", "evidence_reference", "uncertainty", "model_abstained_on_conclusion"])
    _write_csv(root / "pairwise_jaccard.csv", overlaps, ["model_1", "model_2", "jaccard", "note"])
    _private_write(root / "comparison_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True, help="Exact private response JSON; provide 4 times")
    parser.add_argument("--results-dir", type=Path, required=True, help="An empty PRIVATE directory OUTSIDE the Git repo")
    args = parser.parse_args()
    summary = run(args.record, args.results_dir)
    print(json.dumps({
        "status": "SUCCESS_OFFLINE",
        "results_dir": str(args.results_dir.resolve()),
        "n_models": summary["real_llm_response_records"],
        "abstained": summary["n_abstained_on_conclusion"],
        "models_with_edges": summary["n_with_proposed_edges"],
        "distinct_arrows": summary["n_unique_proposed_edges"],
        "reported_cost_usd": summary["reported_pilot_cost_usd"],
        "same_prompt": summary["identical_prompt_hashes"],
        "same_schema": summary["identical_schema_hashes"],
        "majority_acyclic": summary["proposal_majority_is_acyclic"],
        "network_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
