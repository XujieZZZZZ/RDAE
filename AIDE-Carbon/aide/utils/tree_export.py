"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def strip_code_markers(code: str) -> str:
    """Remove markdown code block markers if present."""
    code = code.strip()
    if code.startswith("```"):
        # Remove opening backticks and optional language identifier
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    # Node sizing: normalize metric values across nodes.
    # We treat larger metric as better (works well for maximize=True metrics).
    raw_metrics = np.array([getattr(n.metric, "value_npsafe", float("nan")) for n in jou])
    finite = raw_metrics[np.isfinite(raw_metrics)]
    if finite.size >= 2:
        mn, mx = float(np.min(finite)), float(np.max(finite))
        if mx == mn:
            metrics = np.array([0.5 for _ in jou])
        else:
            metrics = (raw_metrics - mn) / (mx - mn)
            metrics = np.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        metrics = np.array([0.0 for _ in jou])

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[textwrap.fill(n.plan, width=80) for n in jou.nodes],
        code=[strip_code_markers(n.code) for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=metrics.tolist(),
        is_buggy=[bool(getattr(n, "is_buggy", False)) for n in jou.nodes],
        metrics_payload=[
            getattr(n, "metrics_payload", None) for n in jou.nodes
        ],
        quant_score=[getattr(n, "quant_score", None) for n in jou.nodes],
        llm_quality_score=[getattr(n, "llm_quality_score", None) for n in jou.nodes],
        llm_carbon_awareness=[getattr(n, "llm_carbon_awareness", None) for n in jou.nodes],
        llm_soundness=[getattr(n, "llm_soundness", None) for n in jou.nodes],
        llm_innovation=[getattr(n, "llm_innovation", None) for n in jou.nodes],
        llm_rationale=[getattr(n, "llm_rationale", None) for n in jou.nodes],
    )


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js", encoding="utf-8") as f:
        js = f.read()
        js = js.replace("<placeholder>", tree_graph_str)

    with open(template_dir / "template.html", encoding="utf-8") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
    html = generate_html(tree_graph_str)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
