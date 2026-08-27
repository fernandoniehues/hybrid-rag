"""Quebra de documentos em chunks.

Markdown tem estrutura: títulos delimitam assunto. Cortar por número de
caracteres ignorando isso mistura temas dentro de um mesmo chunk e piora a
recuperação. Aqui a divisão respeita os cabeçalhos primeiro e só então
aplica o corte por size, com sobreposição para não perder contexto que
cai exatamente na fronteira.
"""
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    ordinal: int
    heading: str
    content: str


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """Devolve [(cabeçalho, body)] preservando a ordem do documento."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def _split_by_size(text: str, size: int, overlap: int) -> list[str]:
    """Corta em blocos, preferindo quebrar em end de parágrafo ou de frase."""
    if len(text) <= size:
        return [text]

    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # procura uma fronteira natural nos últimos 30% do bloco
            window = text[start + int(size * 0.7):end]
            for sep in ("\n\n", "\n", ". "):
                pos = window.rfind(sep)
                if pos != -1:
                    end = start + int(size * 0.7) + pos + len(sep)
                    break
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def chunk_markdown(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_by_heading(text):
        for piece in _split_by_size(body, chunk_size, overlap):
            chunks.append(Chunk(ordinal=len(chunks), heading=heading, content=piece))
    return chunks
