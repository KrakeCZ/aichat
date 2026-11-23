"""Generic static graph integrity validation: evidence files, line bounds and node references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .model import Graph


@dataclass
class ValidationReport:
    is_valid: bool
    total_nodes: int
    total_edges: int
    isolated_nodes: int
    orphan_edges: List[Tuple[str, str]] = field(default_factory=list)
    invalid_evidence: List[Tuple[str, str, int, str]] = field(default_factory=list)
    regression_results: Dict[str, bool] = field(default_factory=dict)
    regression_details: Dict[str, str] = field(default_factory=dict)


class GraphValidator:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir).resolve()
        self._file_line_cache: Dict[str, int] = {}
        self._file_content_cache: Dict[str, List[str]] = {}

    def _get_file_lines(self, rel_path: str) -> Optional[List[str]]:
        if rel_path in self._file_content_cache:
            return self._file_content_cache[rel_path]
        abs_p = self.root_dir / rel_path
        if not abs_p.is_file():
            return None
        try:
            text = abs_p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            self._file_content_cache[rel_path] = lines
            self._file_line_cache[rel_path] = len(lines)
            return lines
        except Exception:
            return None

    def _get_line_count(self, rel_path: str) -> Optional[int]:
        lines = self._get_file_lines(rel_path)
        return len(lines) if lines is not None else None

    def validate(self, graph: Graph) -> ValidationReport:
        node_ids = set(graph.nodes.keys())
        orphan_edges: List[Tuple[str, str]] = []
        invalid_evidence: List[Tuple[str, str, int, str]] = []

        in_deg: Dict[str, int] = {nid: 0 for nid in node_ids}
        out_deg: Dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in graph.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                orphan_edges.append((edge.source, edge.target))
                continue

            out_deg[edge.source] += 1
            in_deg[edge.target] += 1

            for ev in edge.evidence:
                if ev.line < 1:
                    invalid_evidence.append((edge.source, edge.target, ev.line, "Line number < 1"))
                    continue

                lines = self._get_file_lines(ev.path)
                if lines is None:
                    invalid_evidence.append((edge.source, edge.target, ev.line, f"Evidence file not found: {ev.path}"))
                elif ev.line > len(lines):
                    invalid_evidence.append(
                        (edge.source, edge.target, ev.line, f"Line {ev.line} exceeds file length {len(lines)} in {ev.path}")
                    )

        isolated_count = sum(1 for nid in node_ids if in_deg[nid] == 0 and out_deg[nid] == 0)

        return ValidationReport(
            is_valid=not orphan_edges and not invalid_evidence,
            total_nodes=len(node_ids),
            total_edges=len(graph.edges),
            isolated_nodes=isolated_count,
            orphan_edges=orphan_edges,
            invalid_evidence=invalid_evidence,
        )
