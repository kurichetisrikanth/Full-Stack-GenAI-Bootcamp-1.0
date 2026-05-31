"""Convert PDF files from a folder to Markdown.

No terminal input required.
1. Put PDF(s) in the same folder as this script (or change INPUT_FOLDER).
2. Optionally set PDF_FILE_NAME to target one specific file.
3. Run: python pdf_to_markdown.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import pymupdf


LOGGER = logging.getLogger("pdf_to_markdown")

# ----------------------------
# Configuration (edit these)
# ----------------------------
INPUT_FOLDER = Path(__file__).resolve().parent
PDF_FILE_NAME: str | None = None  # Example: "sample.pdf"
INCLUDE_PAGE_HEADINGS = True
OUTPUT_FOLDER = INPUT_FOLDER  # You can change to: INPUT_FOLDER / "markdown"
ENCODING = "utf-8"


class PdfConversionError(Exception):
    """Base exception for PDF conversion failures."""


@dataclass(frozen=True)
class ConvertOptions:
    include_page_headings: bool = True
    encoding: str = "utf-8"


def _discover_pdfs(folder: Path, pdf_file_name: str | None) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise PdfConversionError(f"Input folder not found: {folder}")

    if pdf_file_name:
        target = folder / pdf_file_name
        if not target.is_file():
            raise PdfConversionError(f"PDF not found: {target}")
        if target.suffix.lower() != ".pdf":
            raise PdfConversionError(f"Not a PDF file: {target}")
        return [target]

    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise PdfConversionError(f"No PDF files found in folder: {folder}")
    return pdfs


def convert_pdf_to_markdown(pdf_path: Path, output_folder: Path, options: ConvertOptions) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{pdf_path.stem}.md"

    sections: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        multipage = doc.page_count > 1
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            if options.include_page_headings and multipage:
                sections.append(f"## Page {page_number}\n\n{text}")
            else:
                sections.append(text)

    markdown_text = "\n\n".join(sections).strip()
    output_path.write_text(f"{markdown_text}\n" if markdown_text else "\n", encoding=options.encoding)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    options = ConvertOptions(
        include_page_headings=INCLUDE_PAGE_HEADINGS,
        encoding=ENCODING,
    )

    try:
        pdf_files = _discover_pdfs(INPUT_FOLDER, PDF_FILE_NAME)
        for pdf_file in pdf_files:
            md_file = convert_pdf_to_markdown(pdf_file, OUTPUT_FOLDER, options)
            LOGGER.info("Converted: %s -> %s", pdf_file.name, md_file.name)
    except PdfConversionError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
