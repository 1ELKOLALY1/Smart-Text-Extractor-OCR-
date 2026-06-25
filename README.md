Intelligent Invoice OCR & Data ExtractorA robust, localized script designed to automate the painful process of reading digital receipts, scanned invoices, or billing screenshots. This tool processes an entire directory of images, utilizes PaddleOCR for text recognition, and applies smart pattern matching to instantly extract key metadata like Invoice Numbers, Billing Dates, and Grand Totals without manual data entry.
 Key Features
Dual Execution Modes (GPU/CPU):
Automatically detects if an NVIDIA GPU (CUDA) is available to blast through invoice extraction sequentially.
Seamlessly falls back to spawning a parallel CPU worker pool (ProcessPoolExecutor) across your hardware threads if a GPU isn't present.
Context-Aware Data Extraction: Uses optimized regular expressions to capture standard billing formats. 
Smart Financial Fallback: If an explicit total label (e.g., "Grand Total") is missing, the script safely scans the document layout to isolate the largest plausible bare amount under $1,000,000.  
Clean Terminal UI: Mutes heavy PaddleOCR library warnings and logs, replacing them with a live tracking progress bar and a clean tabular summary on completion. 
Structured Output Generation: Exports individual raw text extraction results side-by-side with compiled master summaries in both CSV and JSON formats for downstream databases.  
 Getting Started
1. Prerequisites
Ensure you have Python 3 installed. You will need to install paddlepaddle (or paddlepaddle-gpu if you have CUDA cores) alongside paddleocr.

pip install paddleocr
# Install the appropriate version of paddlepaddle according to your hardware
pip install paddlepaddle

2. Supported Formats
The tool supports a wide variety of image formats out of the box:
.png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp

3. Usage
You can run the script by passing the target directory containing your invoices as a command-line argument:
python 23.py /path/to/your/invoice/folder
