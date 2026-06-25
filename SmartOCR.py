import os
import sys
import csv
import re
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.update({
    "FLAGS_enable_pir_api": "0",
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "0",
    "FLAGS_use_mkldnn": "0",
    "FLAGS_enable_onednn": "0",
    "PADDLE_DISABLE_ONEDNN": "1",
    "GLOG_minloglevel": "3",
    "PADDLE_CPP_LOG_LEVEL": "3",
})
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SORT_BY = "filename"   # "filename" | "date" | "total" | "invoice_number"
OUTPUT_DIR = "ocr_results"

#shared between GPU and CPU modes
PADDLE_BASE_OPTS = {
    "use_textline_orientation": False,
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "lang": "en",
    "enable_mkldnn": False,
}

# ---------------------------------------------------------------------------
# recompilation avoid per call
# ---------------------------------------------------------------------------

_INVOICE_NUMBER_PATTERNS = [
    re.compile(r"invoice\s*#?\s*:?\s*([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"inv\s*#?\s*:?\s*([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"#\s*([A-Z0-9\-]{4,})", re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
    re.compile(r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b"),
]

_LABELED_TOTAL_PATTERN = re.compile(
    r"(?:total|amount\s+due|balance\s+due|grand\s+total)[\s\S]*?\$?\s*([\d,]+\.\d{2})",
    re.IGNORECASE,
)
_BARE_AMOUNT_PATTERN = re.compile(r"\b\d[\d,]*\.\d{2}\b")

# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------

def extract_text(reader, image_path: str) -> str:
    """Run OCR on an image and return all recognized lines joined by newlines."""
    lines = []
    ocr_results = reader.predict(image_path)
    if ocr_results:
        for page in ocr_results:
            for text in page.get("rec_texts", []):
                if text and text.strip():
                    lines.append(text.strip())
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Invoice field extraction
# ---------------------------------------------------------------------------

def find_invoice_number(text: str) -> str:
    """Return the first invoice number found in OCR text, or an empty string."""
    for pattern in _INVOICE_NUMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def find_date(text: str) -> str:
    """Return the first date-like string found in OCR text, or an empty string."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def find_total(text: str) -> float:
   
    labeled_matches = _LABELED_TOTAL_PATTERN.findall(text)
    if labeled_matches:
        try:
            return float(labeled_matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Fallback: largest plausible dollar amount in the document
    amounts = []
    for raw in _BARE_AMOUNT_PATTERN.findall(text):
        try:
            value = float(raw.replace(",", ""))
            if value < 1_000_000:
                amounts.append(value)
        except ValueError:
            pass
    return max(amounts) if amounts else 0.0

# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _make_result_record(image_path: str, output_dir: str) -> dict:
    """Return a skeleton result record with default values for a given image."""
    stem = Path(image_path).stem
    return {
        "filename": os.path.basename(image_path),
        "path": image_path,
        "invoice_number": "",
        "date": "",
        "total": 0.0,
        "text": "",
        "status": "ok",
        "error": "",
        "txt_file": os.path.join(output_dir, f"{stem}_extracted.txt"),
    }


def _run_ocr_on_image(reader, image_path: str, output_dir: str) -> dict:
    
    result = _make_result_record(image_path, output_dir)
    try:
        text = extract_text(reader, image_path)
        result["text"] = text
        result["invoice_number"] = find_invoice_number(text)
        result["date"] = find_date(text)
        result["total"] = find_total(text)
        with open(result["txt_file"], "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def process_image_cpu(image_path: str, output_dir: str) -> dict:
    
    from paddleocr import PaddleOCR
    reader = PaddleOCR(**{**PADDLE_BASE_OPTS, "device": "cpu"})
    return _run_ocr_on_image(reader, image_path, output_dir)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def sort_results(results: list[dict], sort_by: str) -> list[dict]:
    """Sort result records by the specified field."""
    if sort_by == "date":
        return sorted(results, key=lambda r: r["date"] or "\xff")
    if sort_by == "total":
        return sorted(results, key=lambda r: r["total"], reverse=True)
    if sort_by == "invoice_number":
        return sorted(results, key=lambda r: r["invoice_number"] or "\xff")
    return sorted(results, key=lambda r: r["filename"].lower())


def write_outputs(results: list[dict], output_dir: str) -> tuple[str, str]:
    """Write a CSV and a compact JSON summary of all results to output_dir."""
    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary.json")

    csv_fields = ["filename", "invoice_number", "date", "total", "status", "error", "txt_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    
    slim_results = [{k: v for k, v in r.items() if k != "text"} for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def format_progress_line(index: int, total: int, result: dict) -> str:
    """Format a single-line progress entry for console output."""
    icon = "✓" if result["status"] == "ok" else "✗"
    parts = [f"  [{index:>5}/{total}] {icon}  {result['filename']}"]
    if result["invoice_number"]:
        parts.append(f"  →  #{result['invoice_number']}")
    if result["date"]:
        parts.append(f"  |  {result['date']}")
    if result["total"]:
        parts.append(f"  |  ${result['total']:,.2f}")
    return "".join(parts)


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted table of all processed invoices to stdout."""
    col = (5, 35, 15, 14, 10)
    print(
        f"{'#':<{col[0]}} {'File':<{col[1]}} "
        f"{'Invoice #':<{col[2]}} {'Date':<{col[3]}} {'Total':>{col[4]}}  Status"
    )
    print("─" * (sum(col) + 10))
    for i, r in enumerate(results, 1):
        total_str = f"${r['total']:,.2f}" if r["total"] else "—"
        print(
            f"{i:<{col[0]}} {r['filename']:<{col[1]}} "
            f"{(r['invoice_number'] or '—'):<{col[2]}} "
            f"{(r['date'] or '—'):<{col[3]}} "
            f"{total_str:>{col[4]}}  {r['status']}"
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    images = sorted(
        str(p) for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print("No supported image files found.")
        sys.exit(1)

    output_dir = os.path.join(folder, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(images)} invoice image(s) in: {folder}")

    import paddle
    gpu_available = paddle.is_compiled_with_cuda() and paddle.get_device().startswith("gpu")

    results = []
    errors = []
    start_time = time.perf_counter()

    if gpu_available:
        print(" NVIDIA CUDA detected ")
        from paddleocr import PaddleOCR
        reader = PaddleOCR(**{**PADDLE_BASE_OPTS, "device": "gpu"})
        print(f"\nProcessing {len(images)} image(s) sequentially on GPU…\n")
        for i, image_path in enumerate(images, 1):
            result = _run_ocr_on_image(reader, image_path, output_dir)
            results.append(result)
            print(format_progress_line(i, len(images), result))
            if result["status"] == "error":
                errors.append(result)
    else:
        print(" GPU unavailable — spawning parallel CPU worker pool...")
        print(f"\nProcessing {len(images)} image(s) across multiple CPU cores…\n")
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_image_cpu, img, output_dir): img
                for img in images
            }
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(format_progress_line(done, len(images), result))
                if result["status"] == "error":
                    errors.append(result)

    elapsed = time.perf_counter() - start_time
    print(f"\nFinished in {elapsed:.1f}s  ({len(results)} files, {len(errors)} errors)\n")

    results = sort_results(results, SORT_BY)
    write_outputs(results, output_dir)
    print_summary_table(results)

    if errors:
        print(f"\n⚠  {len(errors)} file(s) failed:")
        for err in errors:
            print(f"   {err['filename']}: {err['error']}")


if __name__ == "__main__":
    main()