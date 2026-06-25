import os
import sys
import csv
import re
import json
import time
import logging
import argparse
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

# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file with defaults."""
    default_config = {
        "ocr": {
            "language": "en",
            "use_gpu": True,
            "use_textline_orientation": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "enable_mkldnn": False,
        },
        "processing": {
            "supported_formats": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"],
            "output_directory": "ocr_results",
            "sort_by": "filename",
            "timeout_seconds": 300,
            "max_workers": None,
        },
        "extraction": {
            "invoice_number_patterns": [
                r"invoice\s*#?\s*:?\s*([A-Z0-9\-]+)",
                r"inv\s*#?\s*:?\s*([A-Z0-9\-]+)",
                r"#\s*([A-Z0-9\-]{4,})",
            ],
            "date_patterns": [
                r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
                r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
            ],
            "total_labeled_pattern": r"(?:total|amount\s+due|balance\s+due|grand\s+total)[\s\S]*?\$?\s*([\d,]+\.\d{2})",
            "total_bare_amount_pattern": r"\b\d[\d,]*\.\d{2}\b",
            "total_max_threshold": 1000000.0,
        },
        "logging": {
            "level": "INFO",
            "file": "process.log",
            "console_output": True,
            "suppress_library_warnings": True,
        },
        "validation": {
            "require_date": False,
            "require_total": False,
            "validate_file_size_mb": 50,
        },
    }

    # Try to load config from file
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                # Deep merge file config with defaults
                default_config = {
                    **default_config,
                    **file_config,
                    "ocr": {**default_config["ocr"], **file_config.get("ocr", {})},
                    "processing": {**default_config["processing"], **file_config.get("processing", {})},
                    "extraction": {**default_config["extraction"], **file_config.get("extraction", {})},
                    "logging": {**default_config["logging"], **file_config.get("logging", {})},
                    "validation": {**default_config["validation"], **file_config.get("validation", {})},
                }
        except Exception as e:
            print(f"Warning: Could not load config file '{config_path}': {e}")
            print("Using default configuration.")

    return default_config


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "process.log")
    console_output = log_config.get("console_output", True)
    suppress_warnings = log_config.get("suppress_library_warnings", True)

    if suppress_warnings:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddleocr").setLevel(logging.ERROR)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    if console_output:
        logger.addHandler(ch)

    return logger


def compile_patterns(config: dict) -> tuple[list, list, re.Pattern, re.Pattern]:
    """Compile regex patterns from config."""
    extraction_config = config.get("extraction", {})

    invoice_patterns = [
        re.compile(p, re.IGNORECASE)
        for p in extraction_config.get("invoice_number_patterns", [])
    ]

    date_patterns = [
        re.compile(p)
        for p in extraction_config.get("date_patterns", [])
    ]

    labeled_total_pattern = re.compile(
        extraction_config.get("total_labeled_pattern", ""),
        re.IGNORECASE,
    )

    bare_amount_pattern = re.compile(
        extraction_config.get("total_bare_amount_pattern", "")
    )

    return invoice_patterns, date_patterns, labeled_total_pattern, bare_amount_pattern


# ---------------------------------------------------------------------------
# Global config and patterns (set at startup)
# ---------------------------------------------------------------------------

CONFIG = None
INVOICE_NUMBER_PATTERNS = []
DATE_PATTERNS = []
LABELED_TOTAL_PATTERN = None
BARE_AMOUNT_PATTERN = None
LOGGER = None

# ---------------------------------------------------------------------------
# Text extraction
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
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def find_date(text: str) -> str:
    """Return the first date-like string found in OCR text, or an empty string."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def find_total(text: str) -> float:
    """Extract total amount from text using labeled patterns, then fallback to largest amount."""
    labeled_matches = LABELED_TOTAL_PATTERN.findall(text)
    if labeled_matches:
        try:
            return float(labeled_matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Fallback: largest plausible amount in the document
    max_threshold = CONFIG.get("extraction", {}).get("total_max_threshold", 1000000.0)
    amounts = []
    for raw in BARE_AMOUNT_PATTERN.findall(text):
        try:
            value = float(raw.replace(",", ""))
            if value < max_threshold:
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
    """Run OCR on a single image and extract fields."""
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
        LOGGER.error(f"Error processing {image_path}: {exc}")
    return result


def process_image_cpu(image_path: str, output_dir: str) -> dict:
    """Process image on CPU (for multiprocessing)."""
    from paddleocr import PaddleOCR

    paddle_opts = CONFIG.get("ocr", {})
    reader = PaddleOCR(**{**paddle_opts, "device": "cpu"})
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
    """Write CSV and JSON summaries of results."""
    output_config = CONFIG.get("output", {})
    csv_fields = output_config.get("csv_fields", [
        "filename", "invoice_number", "date", "total", "status", "error", "txt_file"
    ])
    include_raw_text = output_config.get("include_raw_text", False)

    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary.json")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Write JSON
    if include_raw_text:
        json_results = results
    else:
        json_results = [{k: v for k, v in r.items() if k != "text"} for r in results]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    LOGGER.info(f"Results written to {csv_path} and {json_path}")
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
    global CONFIG, INVOICE_NUMBER_PATTERNS, DATE_PATTERNS, LABELED_TOTAL_PATTERN, BARE_AMOUNT_PATTERN, LOGGER

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Intelligent Invoice OCR & Data Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SmartOCR.py /path/to/invoices
  python SmartOCR.py /path/to/invoices --output results --lang en
  python SmartOCR.py /path/to/invoices --config custom_config.json --sort-by total
        """
    )
    parser.add_argument("folder", nargs="?", default=None, help="Path to folder containing invoice images")
    parser.add_argument("--config", default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--output", help="Output directory (overrides config)")
    parser.add_argument("--lang", help="OCR language (overrides config)")
    parser.add_argument("--sort-by", choices=["filename", "date", "total", "invoice_number"], help="Sort results by field")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU processing even if GPU is available")

    args = parser.parse_args()

    # Determine folder
    if args.folder is None:
        folder = os.path.dirname(os.path.abspath(__file__))
    else:
        folder = args.folder

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    # Load configuration
    CONFIG = load_config(args.config)
    LOGGER = setup_logging(CONFIG)

    # Override config with command-line arguments
    if args.output:
        CONFIG["processing"]["output_directory"] = args.output
    if args.lang:
        CONFIG["ocr"]["language"] = args.lang
    if args.sort_by:
        CONFIG["processing"]["sort_by"] = args.sort_by
    if args.no_gpu:
        CONFIG["ocr"]["use_gpu"] = False

    # Compile patterns
    INVOICE_NUMBER_PATTERNS, DATE_PATTERNS, LABELED_TOTAL_PATTERN, BARE_AMOUNT_PATTERN = compile_patterns(CONFIG)

    # Get supported formats and output directory
    supported_formats = set(CONFIG.get("processing", {}).get("supported_formats", [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]))
    output_dir = os.path.join(folder, CONFIG.get("processing", {}).get("output_directory", "ocr_results"))
    sort_by = CONFIG.get("processing", {}).get("sort_by", "filename")

    # Find images
    images = sorted(
        str(p) for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in supported_formats
    )
    if not images:
        print("No supported image files found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info(f"Found {len(images)} invoice image(s) in: {folder}")
    print(f"Found {len(images)} invoice image(s) in: {folder}")

    import paddle
    gpu_available = (
        CONFIG.get("ocr", {}).get("use_gpu", True) and
        paddle.is_compiled_with_cuda() and
        paddle.get_device().startswith("gpu")
    )

    results = []
    errors = []
    start_time = time.perf_counter()

    if gpu_available:
        print("\n ✓ NVIDIA CUDA detected ")
        LOGGER.info("Processing on GPU")
        from paddleocr import PaddleOCR
        paddle_opts = CONFIG.get("ocr", {})
        reader = PaddleOCR(**{**paddle_opts, "device": "gpu"})
        print(f"Processing {len(images)} image(s) sequentially on GPU…\n")
        for i, image_path in enumerate(images, 1):
            result = _run_ocr_on_image(reader, image_path, output_dir)
            results.append(result)
            print(format_progress_line(i, len(images), result))
            if result["status"] == "error":
                errors.append(result)
    else:
        print("\n ✗ GPU unavailable — spawning parallel CPU worker pool...")
        LOGGER.info("Processing on CPU with ProcessPoolExecutor")
        print(f"Processing {len(images)} image(s) across multiple CPU cores…\n")
        max_workers = CONFIG.get("processing", {}).get("max_workers", None)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    LOGGER.info(f"Processing completed in {elapsed:.1f}s with {len(errors)} errors")

    results = sort_results(results, sort_by)
    write_outputs(results, output_dir)
    print_summary_table(results)

    if errors:
        print(f"\n⚠  {len(errors)} file(s) failed:")
        for err in errors:
            print(f"   {err['filename']}: {err['error']}")
            LOGGER.warning(f"Failed: {err['filename']} - {err['error']}")


if __name__ == "__main__":
    main()
