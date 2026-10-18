"""PDF ingestion and cleaning.

Turns each 10-K into one LangChain ``Document`` per page with metadata
(company, form, fiscal year, page, 10-K Item). Cleaning steps, in order:

1. Drop exhibit pages after the Exhibit Index (legal boilerplate).
2. Drop running page numbers.
3. Rebuild table rows: PyMuPDF emits one table cell per line, so a numeric
   line is glued back onto the row label above it ("Total revenue | $ 11,139 | $ 17,455").
4. Re-flow wrapped prose lines into paragraphs and normalise whitespace,
   curly quotes and soft hyphens.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from .config import FILINGS, RAW_DIR, Filing

# A table cell: numbers, currency, percentages, (negatives), dashes, years.
_NUMERIC_CELL = re.compile(r"^[\s$€£()\-–—%.,\d]+(?:\s*%)?$")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_ITEM_HEADING = re.compile(
    r"^\s*Item\s+(1A|1B|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)\.\s*(.*)$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.:;!?\")”’]$")

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
    "9": "Changes in Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions",
    "10": "Directors and Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Relationships",
    "14": "Accountant Fees",
    "15": "Exhibits",
    "16": "Form 10-K Summary",
}


def normalise_text(text: str) -> str:
    text = (
        text.replace("­", "")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "—")
        .replace("\xa0", " ")
    )
    return re.sub(r"[ \t]+", " ", text)


def rebuild_tables(lines: list[str]) -> list[str]:
    """Glue numeric cells back onto the label line before them."""
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if out and len(s) <= 24 and _NUMERIC_CELL.match(s):
            out[-1] = f"{out[-1]} | {s}"
        else:
            out.append(s)
    # "Revenue | $ | 17,455" -> "Revenue | $17,455"
    return [re.sub(r"\$ \| ", "$", row) for row in out]


def reflow(lines: list[str]) -> str:
    """Join wrapped prose lines; keep table rows and headings on their own line."""
    paras: list[str] = []
    for line in lines:
        is_row = " | " in line
        if (
            paras
            and not is_row
            and " | " not in paras[-1]
            and not _SENTENCE_END.search(paras[-1])
            and not _ITEM_HEADING.match(line)
            and (line[:1].islower() or line[:1] in "(,$" or len(paras[-1]) > 60)
        ):
            paras[-1] = f"{paras[-1]} {line}"
        else:
            paras.append(line)
    text = "\n".join(paras)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_page(raw: str) -> str:
    lines = normalise_text(raw).splitlines()
    # Trailing / leading page numbers.
    while lines and _PAGE_NUMBER.match(lines[-1]):
        lines.pop()
    while lines and _PAGE_NUMBER.match(lines[0]):
        lines.pop(0)
    return reflow(rebuild_tables(lines))


def detect_item(text: str) -> str | None:
    """Return the last 10-K Item heading that starts a line on this page."""
    found = None
    for line in text.splitlines():
        m = _ITEM_HEADING.match(line)
        if m:
            found = m.group(1).upper()
    return found


def read_pdf_pages(path: Path) -> list[str]:
    import pymupdf  # PyMuPDF keeps word spacing far better than pypdf on these filings

    with pymupdf.open(path) as pdf:
        return [page.get_text() for page in pdf]


def load_filing(filing: Filing, raw_dir: Path = RAW_DIR) -> list[Document]:
    pages = read_pdf_pages(raw_dir / filing.file)[: filing.body_last_page + 1]
    docs: list[Document] = []
    item = "Cover"
    for idx, raw in enumerate(pages):
        text = clean_page(raw)
        if len(text) < 40:
            continue
        # The table of contents (first pages) lists every Item; skip detection there.
        if idx > 3:
            item_on_page = detect_item(text)
        else:
            item_on_page = None
        section = item
        if item_on_page:
            item = item_on_page
            # If the heading sits near the top, the page belongs to the new Item.
            first_lines = "\n".join(text.splitlines()[:4])
            if detect_item(first_lines):
                section = item
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "company": filing.company,
                    "ticker": filing.ticker,
                    "form": filing.form,
                    "fiscal_year": filing.fiscal_year,
                    "source": filing.file,
                    "page": idx + 1,  # 1-based, matches the PDF viewer
                    "item": section,
                    "item_title": ITEM_TITLES.get(section, section),
                },
            )
        )
    return docs


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Document, ...]:
    docs: list[Document] = []
    for filing in FILINGS:
        docs.extend(load_filing(filing))
    return tuple(docs)
