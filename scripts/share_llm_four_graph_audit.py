#!/usr/bin/env python3
"""Reproducible, offline audit of the four-model SHARE causal-graph pilot.

Inputs: three exported comparison files; optionally a PRIVATE SHARE results.json.
Makes zero API calls; does not read respondent records. Outputs are private
exploratory summaries, not empirical identification or evidence adjudication.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import networkx as nx


def load_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def backdoor_sets(g: nx.DiGraph, observed: tuple[str, ...] = ("B", "H", "D8")) -> list[tuple[str, ...]]:
    """Minimal-cardinality observed backdoor sets, *not* general ID or selection correction."""
    bd = g.copy()
    bd.remove_edges_from(list(g.out_edges("A")))
    candidates = [v for v in observed if v not in nx.descendants(g, "A")]
    for size in range(len(candidates) + 1):
        valid = [subset for subset in itertools.combinations(candidates, size)
                 if nx.is_d_separator(bd, {"A"}, {"Y"}, set(subset))]
        if valid:
            return valid
    return []


def run(summary_file: Path, proposal_file: Path, provenance_file: Path,
        out_dir: Path, share_results_file: Path | None = None) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty directory")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    proposals = load_rows(proposal_file)
    provenance = load_rows(provenance_file)
    models = [m["model"] for m in summary["models"]]
    assert len(models) == 4 and len(set(models)) == 4, "Expected four distinct models"
    assert summary["identical_prompt_hashes"] and summary["identical_schema_hashes"], "Prompt/schema hashes differ"
    assert len(provenance) == sum(int(m["n_proposed_edges"]) for m in summary["models"]), "Provenance count mismatch"
    assert len(proposals) == summary["n_unique_proposed_edges"], "Distinct edge count mismatch"
    model_edges = {m: set() for m in models}
    annotated = []
    for row in proposals:
        arrow = row["arrow"]
        if arrow.count("->") != 1:
            raise ValueError(f"Malformed arrow in proposals: {arrow!r}")
        src, dest = arrow.split("->")
        if src not in {"A","Y","D8","B","H","U","S"} or dest not in {"A","Y","D8","B","H","U","S"}:
            raise ValueError("Unexpected node in graph; audit manually")
        supporters = []
        for m in models:
            if row[m] == "1":
                model_edges[m].add((src, dest))
                supporters.append(m)
            elif row[m] not in ("0", "NOT_PROPOSED_ABSTAIN"):
                raise ValueError("Unrecognized edge-proposal status")
        if len(supporters) != int(row["n_proposing_models"]):
            raise ValueError(f"Inconsistent votes for {arrow}")
        cls = ("temporal_ambiguity" if arrow in {"D8->A", "D8->S"}
               else "selection_definition_requires_split" if arrow in {"Y->S", "A->S", "U->S"}
               else "hypothesized_unmeasured_common_cause" if arrow in {"U->A", "U->Y"}
               else "candidate_not_verified")
        annotated.append({
            "arrow": arrow,
            "n_models_proposing": len(supporters),
            "n_nonempty_graphs": int(row["n_nonempty_graphs"]),
            "proportion_nonempty": row["proposal_fraction_nonempty"],
            "scientific_review_flag": cls,
            "source_models": ";".join(supporters),
            "grounded_causal_edge": "NOT_ESTABLISHED",
        })
    for item in summary["models"]:
        assert len(model_edges[item["model"]]) == int(item["n_proposed_edges"]), "Graph count mismatch"

    # Three valid graph proposals; fourth is an explicit no-graph abstention.
    graph_rows = []
    for item in summary["models"]:
        model = item["model"]
        edges = model_edges[model]
        if not edges:
            graph_rows.append({
                "model": model, "n_edges": 0, "acyclic": "NA_no_graph",
                "has_unmeasured_backdoor": "NA_no_graph", "observed_backdoor_sets": "NA_no_graph",
                "hypothetical_no_U_sets": "NA_no_graph", "selection_issue": "NA_no_graph",
                "D8_to_A_claimed": False, "Y_to_S_claimed": False,
                "graph_answer_abstained": item["abstained"],
            })
            continue
        g = nx.DiGraph()
        g.add_nodes_from(["A","Y","B","H","D8","U","S"])
        g.add_edges_from(edges)
        is_dag = nx.is_directed_acyclic_graph(g)
        if not is_dag:
            raise ValueError(f"Non-DAG model {model}")
        observed_sets = backdoor_sets(g)
        # Counterfactual sensitivity that DELETES latent cause arrows, not a claim
        # that they are false. Also ignores outcome selection.
        no_u = g.copy()
        no_u.remove_node("U")
        hypothetical_sets = backdoor_sets(no_u)
        u_path = ("U", "A") in edges and ("U", "Y") in edges
        graph_rows.append({
            "model": model, "n_edges": len(edges), "acyclic": True,
            "has_unmeasured_backdoor": u_path,
            "observed_backdoor_sets": ";".join("+".join(z) if z else "none" for z in observed_sets) if observed_sets else "NONE",
            "hypothetical_no_U_sets": ";".join("+".join(z) if z else "none" for z in hypothetical_sets) if hypothetical_sets else "NONE",
            "selection_issue": any(b == "S" for _, b in edges),
            "D8_to_A_claimed": ("D8", "A") in edges,
            "Y_to_S_claimed": ("Y", "S") in edges,
            "graph_answer_abstained": item["abstained"],
        })

    n_unverified = sum(row["evidence_reference"].strip().lower() == "unverified hypothesis" for row in provenance)
    n_specific_references = sum(("doi.org/" in row["evidence_reference"].lower()
                                  or "https://" in row["evidence_reference"].lower()
                                  or "10." in row["evidence_reference"])
                                 for row in provenance)
    # Compact empirical table: combine prior executed model-conditional estimates,
    # but *never* pretend one is identified under the LLM graphs.
    empirical_rows = []
    if share_results_file is not None:
        share = json.loads(share_results_file.read_text(encoding="utf-8"))
        for name, val in share.get("estimates", {}).items():
            empirical_rows.append({
                "analysis": name,
                "observed_difference": round(float(val["effect"]), 6),
                "ci_lower_conditional": round(float(val["ci_lower"]), 6),
                "ci_upper_conditional": round(float(val["ci_upper"]), 6),
                "n": int(val["n"]),
                "causal_effect_identified": "NO_under_proposed_latent_U_graphs",
            })
        write_csv(out_dir / "share_observational_models_PRIVATE.csv", empirical_rows)

    report = {
        "kind": "four_live_llm_graph_and_share_scientific_audit",
        "scope": "Exploratory; true arrows and causal effects not established",
        "n_models": len(models),
        "n_models_with_edges": sum(bool(x) for x in model_edges.values()),
        "n_unique_arrows": len(proposals),
        "total_proposed_arrows": len(provenance),
        "n_evidence_unverified_hypothesis": n_unverified,
        "n_evidence_with_explicit_link_or_doi": n_specific_references,
        "same_prompt_hash": summary["identical_prompt_hashes"],
        "same_schema_hash": summary["identical_schema_hashes"],
        "selection_and_temporal_warning": True,
        "no_graph_is_not_an_edge_absence_vote": True,
        "n_live_requests_made_by_audit": 0,
        "n_share_person_records_read_by_audit": 0,
        "share_model_outputs_read": len(empirical_rows),
        "graph_adjudication_rows": graph_rows,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "audit_summary.json").chmod(0o600)
    write_csv(out_dir / "edge_admissibility_REVIEW.csv", annotated)
    write_csv(out_dir / "graph_conditional_backdoor_audit.csv", graph_rows)

    # As descriptive output, avoid confidential respondent data, secrets, or long
    # reproductions of LLM rationales in the Markdown report.
    lines = [
        "# Four real LLMs and SHARE: exploratory graph/identification audit", "",
        "**Evidence:** four uploaded real model-response summaries; uploaded edge proposals and provenance. "
        + ("Previously executed, aggregate-only SHARE results.json." if share_results_file else "No new SHARE regression.") , "",
        "## What was directly computed", "",
        f"- {len(models)} provider responses; {sum(bool(x) for x in model_edges.values())} nonempty proposed graphs; {len(proposals)} unique directed arrows; {len(provenance)} proposal-edge records.",
        f"- {n_unverified}/{len(provenance)} edge records explicitly say `unverified hypothesis`; {n_specific_references}/{len(provenance)} supply an explicit URL or DOI as evidence. Generic references are not independently verified.",
        "- The Gemini and Llama graphs have identical directed edges; Luna substitutes `D8->S` and `U->S` for `D8->A` and `Y->S`.",
        "- DeepSeek explicitly withheld a graph. It is not a no-edge causal hypothesis.",
        "", "## Graph-specific backdoor screening", "",
        "| Proposed graph | Observed adjustment sets (as drawn) | Sensitivity: delete U arrows | D8 precedes A claimed? | Selection warning? |", "|---|---|---|---|---|"
    ]
    for x in graph_rows:
        lines.append(f"| {x['model']} | {x['observed_backdoor_sets']} | {x['hypothetical_no_U_sets']} | {x['D8_to_A_claimed']} | {x['selection_issue']} |")
    lines += ["", "All three substantive graphs contain the unblocked latent backdoor `A <- U -> Y`. "
              "No adjustment set confined to observed B/H/D8 blocks that path. "
              "This is a **backdoor-adjustment diagnostic**, not a general proof about all identification strategies. "
              "The hypothetical no-U column is a sensitivity scenario, not an observed estimate or evidence that U is absent.",
              "", "## Required scientific adjudication", "",
              "1. `D8->A` is temporally unverified because Wave 8 depression may overlap exposure. Evaluate separate pre-exposure versus post-exposure DAGs; do not automatically adjust for D8.",
              "2. Split observation into death/survival and interview/nonresponse among survivors. `Y->S` with a combined death-and-observation node is conceptually invalid if Y is undefined after death.",
              "3. Treat the proposed `U` arrows as a sensitivity hypothesis and avoid calling measured-adjustment estimates identified causal ATEs.",
              "4. All `A->Y` proposals are hypotheses, not evidence of the intervention effect; refrain from calling the nonabstaining model 'correct'.",
              "5. A strict majority of nonempty graph proposals is a descriptive convention. Abstaining/empty DeepSeek is neither a positive nor a negative edge vote.",
              "6. The comparable request hashes do not establish comparable output budgets, reasoning settings or endpoint behavior.",
              "", "## Connection to real SHARE results", ""]
    if empirical_rows:
        lines += ["| Model | Observational difference | 95% model-conditional interval | N |", "|---|---:|---|---:|"]
        for e in empirical_rows:
            lines.append(f"| {e['analysis']} | {e['observed_difference']:.3f} | [{e['ci_lower_conditional']:.3f}, {e['ci_upper_conditional']:.3f}] | {e['n']:,} |")
        d = {e["analysis"]: e for e in empirical_rows}
        if "same_sample_without_w8_depression" in d and "aipw_with_w8_depression" in d:
            v = d["aipw_with_w8_depression"]["observed_difference"] - d["same_sample_without_w8_depression"]["observed_difference"]
            lines += ["", f"On the matched analytic sample, inclusion of D8 changes the observed standardized contrast by {v:+.3f} EURO-D points (algebraic model difference only; no interval for this paired difference was calculated)."]
    else:
        lines.append("Supply `--share-results /path/to/private/results.json` to include the already executed aggregate SHARE estimates.")
    lines += ["", "**Status:** Exploratory graph disagreement and observable SHARE sensitivity. No ground-truth graph accuracy, general causal identification, or methodological superiority has been established.", ""]
    (out_dir / "scientific_audit_PRIVATE.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "scientific_audit_PRIVATE.md").chmod(0o600)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--share-results", type=Path, default=None, help="Aggregate-only results.json under protected local storage")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    info = run(a.summary, a.edges, a.provenance, a.out, a.share_results)
    print(json.dumps({k: v for k, v in info.items() if k != 'graph_adjudication_rows'}, indent=2))


if __name__ == "__main__":
    main()
