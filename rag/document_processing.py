"""Parse knowledge sources into ordered, attributable content blocks."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TitleItem
from pydantic import BaseModel


class DocumentProcessingError(ValueError):
    """The source cannot be safely prepared for knowledge ingestion."""


class DocumentBlock(BaseModel):
    kind: Literal["heading", "paragraph", "list_item", "table", "caption", "code"]
    text: str
    heading_path: list[str]
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(BaseModel):
    source_name: str
    source_format: Literal["md", "docx", "pdf"]
    sha256: str
    parser_version: str
    page_count: int | None
    blocks: list[DocumentBlock]
    warnings: list[str]


_FORMATS = {
    ".md": InputFormat.MD,
    ".docx": InputFormat.DOCX,
    ".pdf": InputFormat.PDF,
}


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=RapidOcrOptions(lang=["ch"]),
    )
    return DocumentConverter(
        allowed_formats=list(_FORMATS.values()),
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def parse_document(path: str | Path) -> ParsedDocument:
    """Read one source file without chunking it or changing the original."""

    source = Path(path)
    if source.suffix.lower() == ".doc":
        raise DocumentProcessingError("Convert legacy .doc files to .docx before processing")
    input_format = _FORMATS.get(source.suffix.lower())
    if input_format is None:
        raise DocumentProcessingError(f"Unsupported document format: {source.suffix}")
    if not source.is_file():
        raise DocumentProcessingError(f"Document not found: {source}")
    if source.stat().st_size == 0:
        raise DocumentProcessingError(f"Document is empty: {source}")

    with source.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    try:
        result = _converter().convert(source, raises_on_error=False)
    except Exception as exc:
        raise DocumentProcessingError(f"Unable to process document: {source.name}") from exc
    if result.status != ConversionStatus.SUCCESS or result.document is None:
        raise DocumentProcessingError(
            f"Document conversion incomplete ({result.status}): {source.name}"
        )

    document = result.document
    headings: dict[int, str] = {}
    blocks: list[DocumentBlock] = []
    warnings: list[str] = []

    for item, _depth in document.iterate_items():
        if isinstance(item, PictureItem):
            if not item.caption_text(document).strip():
                warnings.append("Image without a text caption requires visual review")
            continue
        if isinstance(item, TableItem):
            text = item.export_to_markdown(document).strip()
            kind = "table"
        else:
            text = getattr(item, "text", "").strip()
            kind = "paragraph"
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                kind = "heading"
                level = 0 if isinstance(item, TitleItem) else item.level
                headings = {key: value for key, value in headings.items() if key < level}
                headings[level] = text
            elif item.label.value in ("list_item", "caption", "code"):
                kind = item.label.value
        if not text:
            continue

        pages = sorted({provenance.page_no for provenance in item.prov})
        if input_format == InputFormat.PDF and not pages:
            warnings.append("Text without PDF page provenance requires review")
        blocks.append(
            DocumentBlock(
                kind=kind,
                text=text,
                heading_path=list(headings.values()),
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
            )
        )

    if not blocks:
        raise DocumentProcessingError(f"Document has no extractable text content: {source.name}")
    return ParsedDocument(
        source_name=source.name,
        source_format=source.suffix.lower().lstrip("."),
        sha256=sha256,
        parser_version=version("docling"),
        page_count=len(document.pages) if input_format == InputFormat.PDF else None,
        blocks=blocks,
        warnings=list(dict.fromkeys(warnings)),
    )
