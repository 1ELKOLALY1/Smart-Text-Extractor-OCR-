# Smart Text Extractor - OCR

An intelligent Invoice OCR & Data Extractor tool designed to automate the extraction of text and data from digital receipts, scanned invoices, and billing screenshots.

## 📋 Overview

Dealing with a pile of digital receipts, scanned invoices, or billing screenshots? This tool is built to handle the boring stuff for you. Our Smart Text Extractor uses OCR technology to automatically extract and process text from invoice images and documents.

## ✨ Key Features

- **Dual Execution Modes (GPU/CPU)**: Automatically detects if an NVIDIA GPU (CUDA) is available to process invoices efficiently. Seamlessly falls back to a parallel CPU worker pool across your hardware threads if a GPU isn't present.

- **Context-Aware Data Extraction**: Uses optimized regular expressions to capture standard billing formats.

- **Smart Financial Fallback**: If an explicit total label (e.g., "Grand Total") is missing, the script safely scans the document layout to isolate the largest plausible bare amount.

- **Clean Terminal UI**: Mutes heavy library warnings and logs, replacing them with a live tracking progress bar and a clean tabular summary on completion.

- **Structured Output Generation**: Exports individual raw text extraction results alongside compiled master summaries in both CSV and JSON formats for downstream databases.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PaddlePaddle (or paddlepaddle-gpu for CUDA support)
- PaddleOCR

### Installation

```bash
# Clone the repository
git clone https://github.com/1ELKOLALY1/Smart-Text-Extractor-OCR-.git
cd Smart-Text-Extractor-OCR-

# Install dependencies
pip install paddleocr
pip install paddlepaddle  # or paddlepaddle-gpu for GPU support
```

### Supported Formats

The tool supports a wide variety of image formats:
- `.png`
- `.jpg`
- `.jpeg`
- `.tiff`
- `.tif`
- `.bmp`
- `.webp`

## 📖 Usage

Run the script by passing the target directory containing your invoices as a command-line argument:

```bash
python 23.py /path/to/your/invoice/folder
```

The tool will:
1. Process all supported image formats in the directory
2. Extract text and financial data from each invoice
3. Generate CSV and JSON output files with results

## 📁 Project Structure

```
Smart-Text-Extractor-OCR-/
├── README.md
├── .gitignore
├── requirements.txt
├── 23.py                 # Main extraction script
└── src/
    └── # Additional modules
```

## 🔧 Configuration

Adjust the following parameters in the script:
- GPU/CPU mode selection (automatic by default)
- Financial threshold for fallback amount detection
- Output format preferences (CSV, JSON, or both)

## 📊 Output

The tool generates:
- **CSV files**: Tabular format for spreadsheet analysis
- **JSON files**: Structured format for API integration
- **Progress tracking**: Real-time console updates during processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to report bugs or suggest features.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact & Support

For questions, suggestions, or issues, please open a GitHub issue in the repository.

---

**Last Updated:** 2026-06-25
