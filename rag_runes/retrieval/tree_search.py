from __future__ import annotations

from dataclasses import dataclass

from rag_runes.index.hybrid_index import HybridTreeIndex, SearchHit
from rag_runes.schema import DocNode
from rag_runes.text_utils import contains_rune_query


@dataclass(slots=True)
class RetrievalHit:
    node_id: str
    level: str
    score: float
    dense: float
    sparse: float
    rune_boost: float
    path: str
    text: str
    page_from: int
    page_to: int
    image_path: str | None


class TreeSearcher:
    def __init__(self, index: HybridTreeIndex) -> None:
        self.index = index

    def _is_low_information_chunk(self, node: DocNode) -> bool:
        if node.level != "chunk":
            return False
        token_count = int(node.metadata.get("token_count", 0))
        alpha_char_count = int(node.metadata.get("alpha_char_count", 0))
        char_count = int(node.metadata.get("char_count", len(node.text or "")))
        return token_count < 3 and alpha_char_count < 12 and char_count < 24

    def _children_of_levels(self, parent_ids: list[str], levels: set[str]) -> list[str]:
        output: list[str] = []
        for parent_id in parent_ids:
            for child_id in self.index.children.get(parent_id, []):
                child = self.index.node_by_id.get(child_id)
                if child and child.level in levels:
                    output.append(child_id)
        return output

    def _node_path(self, node: DocNode) -> str:
        parts = [node.title]
        cursor = node
        while cursor.parent_id is not None:
            parent = self.index.node_by_id.get(cursor.parent_id)
            if parent is None:
                break
            parts.append(parent.title)
            cursor = parent
        parts.reverse()
        return " > ".join(parts)

    def _convert_hit(self, hit: SearchHit) -> RetrievalHit:
        node = self.index.node_by_id[hit.node_id]
        return RetrievalHit(
            node_id=hit.node_id,
            level=node.level,
            score=hit.score,
            dense=hit.dense,
            sparse=hit.sparse,
            rune_boost=hit.rune_boost,
            path=self._node_path(node),
            text=node.text,
            page_from=node.page_from,
            page_to=node.page_to,
            image_path=node.metadata.get("image_path"),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        branch_factor: int = 4,
    ) -> list[RetrievalHit]:
        book_ids = [node.node_id for node in self.index.nodes if node.level == "book"]
        if not book_ids:
            return []

        chapter_candidates = self._children_of_levels(book_ids, {"chapter"})
        ranked_chapters = self.index.hybrid_rank(
            query=query,
            candidate_ids=chapter_candidates,
            top_k=max(1, branch_factor),
        )
        chapter_ids = [hit.node_id for hit in ranked_chapters]

        section_candidates = self._children_of_levels(chapter_ids, {"section"})
        if not section_candidates:
            section_candidates = []
            for book_id in book_ids:
                section_candidates.extend(
                    self.index.descendants(
                        node_id=book_id,
                        target_levels={"section"},
                    )
                )
        ranked_sections = self.index.hybrid_rank(
            query=query,
            candidate_ids=section_candidates,
            top_k=max(top_k * 2, branch_factor),
        )
        section_ids = [hit.node_id for hit in ranked_sections]

        leaf_candidates = self._children_of_levels(section_ids, {"chunk", "image"})
        if not leaf_candidates:
            leaf_candidates = [
                node.node_id
                for node in self.index.nodes
                if node.level in {"chunk", "image"}
            ]

        ranked_leafs = self.index.hybrid_rank(
            query=query,
            candidate_ids=leaf_candidates,
            top_k=max(top_k * 3, top_k),
        )

        selected_ids: set[str] = set()
        selected: list[SearchHit] = []
        seen_sections: set[str] = set()
        rune_query = contains_rune_query(query)

        if rune_query:
            for hit in ranked_leafs:
                node = self.index.node_by_id[hit.node_id]
                if node.level == "image":
                    selected.append(hit)
                    selected_ids.add(hit.node_id)
                    if node.parent_id:
                        seen_sections.add(node.parent_id)
                    break

        for hit in ranked_leafs:
            if hit.node_id in selected_ids:
                continue
            node = self.index.node_by_id[hit.node_id]
            if self._is_low_information_chunk(node):
                continue
            section_id = node.parent_id or ""
            if section_id in seen_sections and len(selected) < (top_k // 2):
                continue
            seen_sections.add(section_id)
            selected.append(hit)
            selected_ids.add(hit.node_id)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for hit in ranked_leafs:
                if hit.node_id in selected_ids:
                    continue
                node = self.index.node_by_id[hit.node_id]
                if self._is_low_information_chunk(node):
                    continue
                selected.append(hit)
                selected_ids.add(hit.node_id)
                if len(selected) >= top_k:
                    break

        if len(selected) < top_k:
            global_leafs = [
                node.node_id
                for node in self.index.nodes
                if node.level in {"chunk", "image"}
            ]
            expanded_ranked = self.index.hybrid_rank(
                query=query,
                candidate_ids=global_leafs,
                top_k=max(top_k * 4, top_k),
            )
            for hit in expanded_ranked:
                if hit.node_id in selected_ids:
                    continue
                node = self.index.node_by_id[hit.node_id]
                if self._is_low_information_chunk(node):
                    continue
                selected.append(hit)
                selected_ids.add(hit.node_id)
                if len(selected) >= top_k:
                    break

        return [self._convert_hit(hit) for hit in selected]
