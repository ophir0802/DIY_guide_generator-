import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys

# Add project root to sys.path so we can import crawler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crawler import fetch_category_links, parse_html, save_single_guide, main, make_absolute_url

# Mock HTML Content for Tests
SAMPLE_CATEGORY_HTML = """
<html>
<body>
    <div class="category-list">
        <a href="/stry/how-to-fix-a-freezer">How to Fix a Freezer</a>
        <a href="https://www.doityourself.com/stry/how-to-defrost">How To Defrost</a>
        <a href="/stry/maintenance-tips">Maintenance Tips</a> <!-- Should be ignored -->
    </div>
</body>
</html>
"""

SAMPLE_ARTICLE_HTML = """
<html>
<body>
    <h1 class="how-to__article-title">How to Fix a Freezer</h1>
    <span class="author-name">John Doe</span>
    
    <div class="article-body">
        <p>Introduction text here.</p>
        
        <h3>Things You'll Need</h3>
        <ul>
            <li>Screwdriver</li>
            <li>Wrench</li>
        </ul>
        
        <h3>Step 1: Unplug</h3>
        <p>Unplug the device.</p>
        <img src="/images/unplug.jpg" />
        
        <h3>Step 2: Open Panel</h3>
        <p>Use screwdriver to open.</p>
    </div>
</body>
</html>
"""

@patch('crawler.requests.get')
def test_fetch_category_links(mock_get):
    """Test extracting links from a category page."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = SAMPLE_CATEGORY_HTML
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    base_url = "https://www.doityourself.com/scat/freezer"
    links = fetch_category_links(base_url)

    # Expectations
    assert len(links) == 2
    assert "https://www.doityourself.com/stry/how-to-fix-a-freezer" in links
    assert "https://www.doityourself.com/stry/how-to-defrost" in links
    # "Maintenance Tips" should be excluded because it doesn't contain "How to"

def test_parse_html_success():
    """Test parsing a valid article HTML."""
    url = "https://www.doityourself.com/stry/how-to-fix-a-freezer"
    data = parse_html(SAMPLE_ARTICLE_HTML, url)

    assert data is not None
    assert data['title'] == "How to Fix a Freezer"
    assert data['author'] == "John Doe"
    assert "Screwdriver" in data['supplies']
    assert len(data['steps']) == 2
    assert data['steps'][0][0] == "Step 1: Unplug"
    assert "https://www.doityourself.com/images/unplug.jpg" in data['image_urls']

def test_parse_html_invalid_title():
    """Test that articles without 'How to' in title are skipped."""
    invalid_html = "<html><h1>Fix a Freezer</h1></html>"
    data = parse_html(invalid_html, "http://example.com")
    assert data is None

@patch('builtins.open', new_callable=mock_open)
@patch('crawler.json.dump')
@patch('crawler.os.makedirs')
def test_save_single_guide(mock_makedirs, mock_json_dump, mock_file):
    """Test saving a guide to JSON."""
    guide_data = {
        "title": "Test Guide",
        "author": "Me",
        "additional_text_boxes": [],
        "supplies": [],
        "steps": [["Step 1", "Do it"]],
        "image_urls": [],
        "url": "http://example.com"
    }

    result = save_single_guide(guide_data, output_dir="test_data")
    
    assert result is True
    mock_makedirs.assert_called_once()
    mock_file.assert_called() # Should open a file
    mock_json_dump.assert_called() # Should dump json

@patch('crawler.fetch_category_links')
@patch('crawler.fetch_page')
@patch('crawler.parse_html')
@patch('crawler.save_single_guide')
@patch('builtins.open', new_callable=mock_open, read_data='{"urls": ["http://test.com/cat1"]}')
@patch('crawler.os.path.exists')
def test_main_loop(mock_exists, mock_file, mock_save, mock_parse, mock_fetch_page, mock_fetch_links):
    """Test the main loop reads config and processes URLs."""
    # Setup mocks
    mock_exists.return_value = True # urls.json exists
    mock_fetch_links.return_value = ["http://test.com/guide1"]
    mock_fetch_page.return_value = "<html></html>"
    mock_parse.return_value = {"title": "Guide 1"}
    mock_save.return_value = True

    # Run main
    main()

    # Assertions
    mock_file.assert_called_with("urls.json", 'r')
    mock_fetch_links.assert_called_with("http://test.com/cat1")
    mock_save.assert_called_once()
