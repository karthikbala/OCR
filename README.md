# OCR (Optical Character Recognition)

A simple Python-based OCR tool that extracts text from images using Tesseract OCR engine.

## Prerequisites

- Python 3.6 or higher
- Tesseract OCR engine installed on your system

### Installing Tesseract

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr
```

#### Windows
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Installation

1. Clone this repository:
```bash
git clone https://github.com/karthikbala/OCR.git
cd OCR
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python scan.py path/to/your/image.jpg
```

Specify a language (default is English):
```bash
python scan.py path/to/your/image.jpg --lang eng
```

Save the extracted text to a file:
```bash
python scan.py path/to/your/image.jpg --output extracted_text.txt
```

## Supported Languages

Tesseract supports many languages. Some common language codes:
- English: `eng`
- French: `fra`
- German: `deu`
- Spanish: `spa`
- Chinese (Simplified): `chi_sim`

For a complete list, check the [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).

## License

MIT