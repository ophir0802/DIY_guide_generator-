# DIYGuideGen - DoItYourself.com Crawler

A Python web crawler designed to extract "How-to" guide information from DoItYourself.com. The crawler fetches guide data including titles, authors, supplies, steps, and images, then validates and saves the data to individual JSON files.

## Features
- 🔍 **Web Scraping**: Fetches "How-to" articles from DoItYourself.com categories
- 🎯 **Smart Parsing**: Extracts structured data (title, author, supplies, steps)
- 📂 **Individual Output**: Saves each guide as a separate JSON file in `raw_data/`
- 🖼️ **Image Extraction**: Handles images associated with steps and converts to absolute URLs
- ✅ **Data Validation**: Uses Pydantic for strict data validation
- 🛡️ **Error Handling**: Robust error handling for network issues and parsing failures
- 🤖 **Human-like Behavior**: Random delays and user-agent rotation
- 🏷️ **Content Filtering**: Specifically targets articles with "How-to" in the title/link

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The crawler is currently configured to target the Freezer category: `https://www.doityourself.com/scat/freezer`

Run the crawler:

```bash
python crawler.py
```

The extracted data will be saved to the `raw_data` directory, with each guide in its own JSON file named after the title.

### Output Format

Each JSON file has the following structure:

```json
{
    "title": "How to Guide Title",
    "author": "Author Name",
    "supplies": ["Item 1", "Item 2", ...],
    "steps": ["Step 1 description", "Step 2 description", ...],
    "image_urls": ["https://...", "https://...", ...],
    "url": "https://www.doityourself.com/stry/guide-url"
}
```

## Project Structure

```
DIYGuideGen/
├── crawler.py              # Main crawler implementation
├── raw_data/               # Output directory for JSON files
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── tests/
    └── ...                 # Tests
```

## Dependencies

- **requests** - HTTP library
- **beautifulsoup4** - HTML parsing library
- **pydantic** - Data validation
- **lxml** - XML/HTML parser

## Disclaimer

This project is for educational purposes. Please use responsibly and respect the target website's Terms of Service and `robots.txt`.

