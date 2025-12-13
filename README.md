# DIYGuideGen - Instructables Crawler

A Python web crawler designed to extract guide information from Instructables.com. The crawler fetches guide data including titles, authors, supplies, steps, and images, then validates and saves the data to JSON format.

## Features
- 🔍 **Web Scraping**: Fetches HTML content from Instructables guide pages
- 🎯 **Smart Parsing**: Multiple fallback selectors to handle different page layouts
- 🖼️ **Image Extraction**: Handles lazy-loaded images and converts relative URLs to absolute
- ✅ **Data Validation**: Uses Pydantic for strict data validation
- 🛡️ **Error Handling**: Robust error handling for network issues and parsing failures
- 🤖 **Human-like Behavior**: Random delays and user-agent rotation to avoid detection
- 🧪 **Unit Tests**: Comprehensive test suite for core functionality

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

Edit the `target_urls` list in `crawler.py` to include the Instructables URLs you want to crawl:

```python
target_urls = [
    "https://www.instructables.com/Your-Guide-URL/",
    "https://www.instructables.com/Another-Guide-URL/",
]
```

Then run the crawler:

```bash
python crawler.py
```

The extracted data will be saved to `guides.json` in the project root directory.

### Output Format

The crawler generates a JSON file with the following structure:

```json
[
    {
        "title": "Guide Title",
        "author": "Author Name",
        "supplies": ["Item 1", "Item 2", ...],
        "steps": ["Step 1 description", "Step 2 description", ...],
        "image_urls": ["https://...", "https://...", ...]
    }
]
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fetch_page.py -v

# Run with coverage (if pytest-cov is installed)
pytest tests/ --cov=crawler --cov-report=html
```

### Test Coverage

The test suite includes comprehensive tests for:
- Successful page fetching
- HTTP error handling (4xx, 5xx)
- Connection errors
- Timeout handling
- Header configuration
- Delay mechanisms
- Empty response handling
- Generic exception handling

## Project Structure

```
DIYGuideGen/
├── crawler.py              # Main crawler implementation
├── guides.json             # Output file for extracted data
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── tests/
    ├── __init__.py
    └── test_fetch_page.py  # Unit tests for fetch_page function
```

## Dependencies

- **requests** - HTTP library for making web requests
- **beautifulsoup4** - HTML parsing library
- **pydantic** - Data validation using Python type annotations
- **lxml** - Fast XML/HTML parser (used by BeautifulSoup)
- **pytest** - Testing framework
- **pytest-mock** - Mocking utilities for pytest

## Features in Detail

### Smart HTML Parsing

The crawler uses multiple selector strategies to extract data:
- **Title**: Tries 5 different selectors including meta tags
- **Author**: Tries 5 different selectors
- **Supplies**: Searches multiple container types
- **Steps**: Handles various step container structures
- **Images**: Supports lazy-loaded images with multiple attribute checks

### URL Handling

- Automatically converts relative image URLs to absolute URLs
- Handles various image loading attributes (`src`, `data-src`, `data-lazy-src`, etc.)
- Validates that image URLs are properly formatted

### Rate Limiting

- Random delays between requests (2-5 seconds)
- User-agent rotation to mimic different browsers
- Configurable timeout settings

## Important Notes

⚠️ **Legal and Ethical Considerations**:
- Always respect Instructables' Terms of Service
- Check `robots.txt` before crawling
- Use reasonable delays between requests
- Don't overload their servers
- Consider using official APIs if available

⚠️ **HTML Structure Changes**:
- Web scraping is fragile - if Instructables changes their HTML structure, the selectors may need to be updated
- The crawler includes multiple fallback selectors to handle some variations
- Regular testing is recommended to ensure continued functionality

## Troubleshooting

### No data extracted

- Check if the HTML structure of Instructables has changed
- Verify the URLs are accessible
- Check the logs for parsing warnings

### Connection errors

- Verify your internet connection
- Check if Instructables is accessible
- Consider increasing timeout values

### Import errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8+

## Contributing

Feel free to submit issues or pull requests to improve the crawler!

## License

This project is provided as-is for educational purposes. Please use responsibly and in accordance with Instructables' Terms of Service.

