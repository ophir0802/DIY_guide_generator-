# DIY Guide Generator

A Python-based system that crawls DIY guides from DoItYourself.com and transforms them into standardized, structured format using Google Gemini AI.

## Overview

This project consists of three main components:
1. **Web Crawler** (`crawler.py`) - Scrapes DIY guides from DoItYourself.com
2. **AI Agent** (`ai_agent.py`) - Transforms raw guides into standardized format using Google Gemini
3. **Tool Locator** (`tool_locator.py`) - Uses computer vision to locate tools in images and provide geometric coordinates

## Features

### Web Crawler
- 🔍 **Web Scraping**: Fetches "How-to" articles from DoItYourself.com categories
- 🎯 **Smart Parsing**: Extracts structured data (title, author, supplies, steps, images)
- 📂 **Individual Output**: Saves each guide as a separate JSON file in `raw_data/`
- 🖼️ **Image Extraction**: Handles images and converts to absolute URLs
- ✅ **Data Validation**: Uses Pydantic for strict data validation
- 🛡️ **Error Handling**: Robust error handling for network issues
- 🤖 **Human-like Behavior**: Random delays and user-agent rotation
- 🏷️ **Content Filtering**: Specifically targets articles with "How-to" in title

### AI-Powered Guide Synthesis
- 🤖 **LLM Processing**: Uses Google Gemini 1.5 Flash for intelligent text processing
- 📝 **Action Extraction**: Breaks down steps into discrete, actionable strings
- 🔧 **Tool Identification**: Identifies and normalizes tool names from supplies
- 📍 **Geometric Location**: Infers tool locations in images using computer vision
- ✋ **Hand Requirements**: Determines number of hands needed for each step
- 🎯 **Part Recognition**: Identifies specific parts/objects being worked on
- 📊 **Structured Output**: Generates standardized JSON format

## Installation

### Prerequisites

- Python 3.9 or higher (required by google-genai)
- pip (Python package manager)
- Google API Key (for Gemini AI) - Get one at https://ai.google.dev/

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google API Key:

**Windows PowerShell (Current Session)**:
```powershell
$env:GOOGLE_API_KEY="your-api-key-here"
```

**Windows PowerShell (Permanent)**:
```powershell
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'your-api-key-here', 'User')
```

**Linux/Mac**:
```bash
export GOOGLE_API_KEY="your-api-key-here"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

4. Restart your terminal/IDE after setting environment variables

## Usage

### Step 1: Crawl DIY Guides

The crawler targets the Freezer category by default: `https://www.doityourself.com/scat/freezer`

Run the crawler:
```bash
python crawler.py
```

This will save raw guides to `raw_data/` directory.

### Step 2: Process with AI Agent

Process a single guide:
```bash
python ai_agent.py raw_data/how_to_clean_chest_freezer_condenser_coils.json
```

Process all guides in batch:
```bash
python ai_agent.py
```

Standardized guides will be saved to `standardized_data/` directory.

## Output Formats

### Raw Guide Format (from crawler)

```json
{
    "title": "How to Clean Chest Freezer Condenser Coils",
    "author": "Author Name",
    "additional_text_boxes": ["Introduction text..."],
    "supplies": ["Screwdriver", "Vacuum Cleaner", "Brush"],
    "steps": [
        ["Step 1 - Unplug the Freezer", "Safety first..."],
        ["Step 2 - Remove Cover", "Use screwdriver to remove..."]
    ],
    "image_urls": ["https://example.com/image1.jpg"],
    "url": "https://www.doityourself.com/..."
}
```

### Standardized Guide Format (from AI agent)

```json
{
    "header": {
        "title": "How to Clean Chest Freezer Condenser Coils",
        "description": "Learn how to maintain your freezer by cleaning condenser coils.",
        "toolbox": ["Screwdriver", "Vacuum Cleaner", "Brush"]
    },
    "steps": [
        {
            "step_id": 1,
            "description": "Unplug the freezer for safety",
            "actions": [
                "Locate the power cord",
                "Unplug from wall outlet",
                "Verify power is off"
            ],
            "tool": "None",
            "part": "Power Cord",
            "geometric_location": {
                "x": 0.1,
                "y": 0.2,
                "w": 0.15,
                "h": 0.1
            },
            "hands": 1
        }
    ]
}
```

## Project Structure

```
DIY_guide_generator/
├── crawler.py                  # Web crawler for DoItYourself.com
├── ai_agent.py                 # AI-powered guide standardization
├── tool_locator.py             # Computer vision for tool location
├── raw_data/                   # Raw crawled guides (JSON)
├── standardized_data/          # Processed standardized guides (JSON)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── MIGRATION_NOTES.md          # Package migration details
├── ISSUES_AND_FIXES.md         # Complete fix documentation
└── tests/
    ├── test_ai_agent.py        # Tests for AI agent
    └── test_fetch_page.py      # Tests for crawler
```

## Dependencies

### Core Libraries
- **requests** - HTTP library for web requests
- **beautifulsoup4** - HTML parsing library
- **pydantic** - Data validation and serialization
- **lxml** - XML/HTML parser
- **Pillow** - Image processing

### AI & ML
- **google-genai** - Google Gemini API client (v1.0+)
  - Replaces deprecated `google-generativeai` package
  - Supports Gemini 1.5 Flash and Pro models
  - Multimodal input (text + images)

## Recent Updates (Jan 2026)

### ✅ Package Migration
- Migrated from deprecated `google-generativeai` to `google-genai`
- Fixed 404 model errors
- Fixed schema validation errors
- Updated all API calls to new format

**See `MIGRATION_NOTES.md` for detailed migration guide**
**See `ISSUES_AND_FIXES.md` for complete fix documentation**

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_ai_agent.py -v
```

## Troubleshooting

### API Key Issues
```bash
# Verify API key is set
echo $env:GOOGLE_API_KEY  # PowerShell
echo $GOOGLE_API_KEY      # Bash
```

### Package Issues
```bash
# Verify correct package is installed
pip show google-genai

# Should show version 1.0.0 or higher
# If you see "google-generativeai", uninstall it:
pip uninstall google-generativeai
pip install google-genai
```

### Common Errors

**404 Model Error**: Use `gemini-1.5-flash` not `gemini-1.5-flash-latest`

**Schema Error**: Make sure you're using the updated code with recursive schema cleaning

**Import Error**: Restart your terminal/IDE after installing new package

## Performance Notes

- **Rate Limiting**: 1 second delay between API calls
- **Retry Logic**: 3 retries with exponential backoff
- **Timeout**: 10-15 seconds for network requests
- **Cost**: Using Gemini 1.5 Flash (free tier available)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

This project is for educational purposes.

## Disclaimer

- This project is for educational purposes
- Please respect the target website's Terms of Service and `robots.txt`
- API usage subject to Google's terms and rate limits
- Free tier has quota limits - monitor your usage

## Support

For issues related to:
- **API Migration**: See `MIGRATION_NOTES.md`
- **Bug Fixes**: See `ISSUES_AND_FIXES.md`
- **Google API**: https://ai.google.dev/
- **Package Docs**: https://github.com/googleapis/python-genai

