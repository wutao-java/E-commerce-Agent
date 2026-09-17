"""将知识文档解析为保持阅读顺序和来源位置的结构化内容块。"""

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
    """文档无法可靠解析，不能直接进入知识库。"""


class DocumentBlock(BaseModel):
    """一个标题、段落或表格；页码仅记录解析器实际提供的位置。"""

    kind: Literal["heading", "paragraph", "list_item", "table", "caption", "code"]
    text: str
    heading_path: list[str]
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(BaseModel):
    """原文哈希、解析器版本与有序内容块组成后续分块的输入。"""

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
    """复用转换器；PDF 同时启用 OCR 和表格结构识别。"""

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
    """读取一份原文件，不切片、不修改文件内容。"""

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

    # 哈希用于区分同一文档的不同内容版本，与文件名无关。
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

    # 按解析器恢复的阅读顺序遍历，同时追踪当前标题层级。
    for item, _depth in document.iterate_items():
        if isinstance(item, PictureItem):
            # 没有文字说明的图片可能包含未提取的业务规则，交由人工复核。
            if not item.caption_text(document).strip():
                warnings.append("Image without a text caption requires visual review")
            continue
        if isinstance(item, TableItem):
            # 表格整体保留为 Markdown，后续不会按单元格暴力拆分。
            text = item.export_to_markdown(document).strip()
            kind = "table"
        else:
            text = getattr(item, "text", "").strip()
            kind = "paragraph"
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                kind = "heading"
                level = 0 if isinstance(item, TitleItem) else item.level
                # 进入同级或更高层标题时，清除旧的下级章节。
                headings = {key: value for key, value in headings.items() if key < level}
                headings[level] = text
            elif item.label.value in ("list_item", "caption", "code"):
                kind = item.label.value
        if not text:
            continue

        # 只读取真实来源页码，不为 Markdown / Word 推算虚构页码。
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
        # 同类风险只提示一次，避免重复图片或段落产生大量相同告警。
        warnings=list(dict.fromkeys(warnings)),
    )
