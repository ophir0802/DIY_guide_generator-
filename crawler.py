import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from urllib.parse import urljoin, urlparse
import time
import random
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#1. Data Model Definition

class InstructableGuide(BaseModel):
    """
    Pydantic model to define the strict structure of an Instructable guide.
    """
    title: str
    author: str
    supplies: List[str]
    steps: List[str]
    image_urls: List[str]

# List of rotating User-Agents to mimic human browser traffic
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
]

def get_random_headers():
    """Returns a header dict with a random User-Agent."""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
    }

# --- 3. Core Functions ---

def fetch_page(url: str) -> Optional[str]:
    """
    Fetches the HTML content of a page with random delays and user-agent rotation.
    """
    try:
        headers = get_random_headers()
        
        # Random delay to behave like a human (2 to 5 seconds)
        delay = random.uniform(2, 5)
        logging.info(f"Waiting {delay:.2f}s before requesting {url}...")
        time.sleep(delay)
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise error for bad status codes (4xx, 5xx)
        
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def make_absolute_url(url: str, base_url: str) -> str:
    """
    Converts a relative URL to an absolute URL using the base URL.
    """
    if not url:
        return ""
    # If already absolute, return as is
    if urlparse(url).netloc:
        return url
    # Convert relative to absolute
    return urljoin(base_url, url)


def validate_and_save(raw_data_list: List[dict], output_filename: str = "guides.json"):
    """
    Validates raw data using Pydantic and saves valid records to JSON.
    """
    valid_guides = []
    
    for entry in raw_data_list:
        try:
            # Pydantic Validation happens here
            guide = InstructableGuide(**entry)
            valid_guides.append(guide.model_dump())
            logging.info(f"Successfully validated: {guide.title}")
        except ValidationError as e:
            logging.error(f"Validation failed for entry '{entry.get('title', 'Unknown')}': {e}")

    # Save to file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(valid_guides, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved {len(valid_guides)} valid guides to {output_filename}")
    except IOError as e:
        logging.error(f"File write error: {e}")

# --- 4. Execution Logic (Main Loop) ---

def main():
    # 1. Define list of URLs to crawl
    # In a real scenario, this might come from crawling a category page first.
    target_urls = [
        "https://www.instructables.com/3d-Printed-Knee-Hockey-Target/", 
        "https://www.instructables.com/Building-a-Coffee-Table-With-Endless-Depth-Infinit/",
        # Add real URLs here to test
    ]

    raw_results = []

    print("--- Starting Crawler ---")

    # 2. Main Loop
    for url in target_urls:
        logging.info(f"Processing: {url}")
        
        # Step A: Fetch
        html = fetch_page(url)
        
        if html:
            # Step B: Parse (pass base URL for absolute URL conversion)
            data = parse_html(html, url)
            
            if data:
                # Collect raw data for batch validation/saving
                raw_results.append(data)
                logging.info(f"Scraped guide: {data['title']}")
            else:
                logging.warning(f"Failed to parse content from {url}")
        else:
            logging.warning(f"Skipping {url} due to fetch error.")

    # Step C: Validate and Save
    print("--- Validating and Saving Data ---")
    validate_and_save(raw_results)
    print("--- Crawler Finished ---")
def parse_html(html_content: str, base_url: str) -> Optional[dict]:
    """
    Parses raw HTML to extract guide data.
    Returns a dictionary matching the Pydantic schema or None if parsing fails.
    
    Args:
        html_content: The HTML content to parse
        base_url: The base URL used to convert relative image URLs to absolute
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # --- Extract Title ---
        # Try multiple selectors for title
        title = None
        title_selectors = [
            ('h1', {'class': 'header-title'}),
            ('h1', {'class': 'title'}),
            ('h1', {}),
            ('meta', {'property': 'og:title'}),
            ('title', {})
        ]
        
        for tag_name, attrs in title_selectors:
            title_tag = soup.find(tag_name, attrs)
            if title_tag:
                if tag_name == 'meta':
                    title = title_tag.get('content', '').strip()
                else:
                    title = title_tag.get_text(strip=True)
                if title and title != "Unknown Title":
                    break
        
        if not title or title == "Unknown Title":
            title = "Unknown Title"
            logging.warning(f"Could not extract title from {base_url}")

        # --- Extract Author ---
        author = None
        author_selectors = [
            ('a', {'rel': 'author'}),
            ('span', {'class': 'author-name'}),
            ('a', {'class': 'author'}),
            ('div', {'class': 'author'}),
            ('meta', {'property': 'article:author'})
        ]
        
        for tag_name, attrs in author_selectors:
            author_tag = soup.find(tag_name, attrs)
            if author_tag:
                if tag_name == 'meta':
                    author = author_tag.get('content', '').strip()
                else:
                    author = author_tag.get_text(strip=True)
                if author:
                    break
        
        if not author:
            author = "Unknown Author"
            logging.warning(f"Could not extract author from {base_url}")

        # --- Extract Supplies ---
        supplies = []
        supplies_selectors = [
            ('section', {'id': 'supplies'}),
            ('div', {'class': 'supplies-list'}),
            ('div', {'class': 'supplies'}),
            ('ul', {'class': 'supplies'}),
            ('div', {'id': 'supplies-list'})
        ]
        
        supplies_section = None
        for tag_name, attrs in supplies_selectors:
            supplies_section = soup.find(tag_name, attrs)
            if supplies_section:
                break
        
        if supplies_section:
            for item in supplies_section.find_all('li'):
                text = item.get_text(strip=True)
                if text:
                    supplies.append(text)
        else:
            logging.debug(f"Could not find supplies section for {base_url}")

        # --- Extract Steps and Images ---
        steps_text = []
        image_urls = []
        
        # Try multiple selectors for steps - Instructables specific patterns
        step_containers = []
        
        # First try CSS selectors for common patterns
        css_selectors = [
            'div.step',
            'article.step',
            'section.step',
            'div.step-body',
            'div[data-step]',
            'li.step',
            '[class*="step"]',
            '[id*="step"]',
        ]
        
        for selector in css_selectors:
            step_containers = soup.select(selector)
            if step_containers:
                logging.info(f"Found {len(step_containers)} steps using CSS selector: {selector}")
                break
        
        # If CSS selectors didn't work, try attribute-based searches
        if not step_containers:
            step_selectors = [
                ('div', {'class': 'step'}),
                ('article', {'class': 'step'}),
                ('section', {'class': 'step'}),
                ('div', {'class': 'step-body'}),
                ('div', {'data-step': True}),
                ('li', {'class': 'step'}),
            ]
            
            for tag_name, attrs in step_selectors:
                step_containers = soup.find_all(tag_name, attrs)
                if step_containers:
                    logging.info(f"Found {len(step_containers)} steps using selector: {tag_name} with {attrs}")
                    break
            
            # Try finding elements with 'step' in class or id
            if not step_containers:
                all_divs = soup.find_all(['div', 'article', 'section'])
                for elem in all_divs:
                    class_attr = elem.get('class', [])
                    id_attr = elem.get('id', '')
                    if (isinstance(class_attr, list) and any('step' in str(c).lower() for c in class_attr)) or \
                       (isinstance(id_attr, str) and 'step' in id_attr.lower()):
                        step_containers.append(elem)
                if step_containers:
                    logging.info(f"Found {len(step_containers)} steps using flexible class/id matching")
        
        if step_containers:
            for step in step_containers:
                # Get Text - try multiple selectors, prioritizing step-specific content
                text = None
                text_selectors = [
                    ('div', {'class': 'step-body'}),
                    ('div', {'class': 'step-text'}),
                    ('div', {'class': 'step-body-text'}),
                    ('div', {'class': 'caption'}),
                    ('p', {'class': 'step-body'}),
                    ('div', {'class': 'content'}),
                    # Get all paragraphs within step, but filter carefully
                    ('p', {}),
                ]
                
                for tag_name, attrs in text_selectors:
                    if tag_name == 'p' and attrs == {}:
                        # Special handling for paragraphs - get all but filter
                        paragraphs = step.find_all('p')
                        if paragraphs:
                            # Combine paragraphs but filter out very short ones
                            combined_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 10])
                            if combined_text:
                                text = combined_text
                                break
                    else:
                        text_div = step.find(tag_name, attrs)
                        if text_div:
                            text = text_div.get_text(strip=True)
                            if text and len(text.strip()) > 10:
                                break
                
                # If no specific text div found, get text but filter carefully
                if not text:
                    # Get all text but exclude common non-step elements
                    all_text = step.get_text(separator=' ', strip=True)
                    # Filter out very short text or text that looks like navigation/metadata
                    if all_text and len(all_text) > 20:
                        # Exclude text that's likely not step content
                        exclude_patterns = ['share', 'like', 'follow', 'comment', 'next', 'previous', 'step 1 of']
                        if not any(pattern in all_text.lower()[:50] for pattern in exclude_patterns):
                            text = all_text
                
                # Clean and validate step text
                if text:
                    text = text.strip()
                    # Filter out steps that are too short or look like metadata
                    if len(text) > 15 and not text.lower().startswith(('step', 'next', 'previous', 'share')):
                        steps_text.append(text)
                
                # Get Images - check multiple attributes for lazy loading
                imgs = step.find_all('img')
                for img in imgs:
                    # Skip decorative images (icons, logos, etc.)
                    img_class = img.get('class', [])
                    img_alt = img.get('alt', '').lower()
                    if any(skip in str(img_class).lower() or skip in img_alt for skip in ['icon', 'logo', 'avatar', 'button']):
                        continue
                    
                    # Try multiple attributes for image sources (handles lazy loading)
                    src = (img.get('src') or 
                          img.get('data-src') or 
                          img.get('data-lazy-src') or
                          img.get('data-original') or
                          img.get('data-url'))
                    
                    if src:
                        # Convert to absolute URL
                        absolute_url = make_absolute_url(src, base_url)
                        if absolute_url and absolute_url not in image_urls:
                            # Filter out small images (likely icons)
                            width = img.get('width') or img.get('data-width')
                            height = img.get('height') or img.get('data-height')
                            if width and height:
                                try:
                                    if int(width) < 50 or int(height) < 50:
                                        continue
                                except (ValueError, TypeError):
                                    pass
                            image_urls.append(absolute_url)
        else:
            # Fallback: try to find main content area but be more selective
            logging.warning(f"Could not find step containers, trying fallback strategy for {base_url}")
            body_selectors = [
                ('div', {'class': 'main-content'}),
                ('div', {'class': 'article-body'}),
                ('article', {'class': 'article'}),
                ('div', {'id': 'content'}),
                ('div', {'class': 'steps'}),
            ]
            
            body_content = None
            for tag_name, attrs in body_selectors:
                body_content = soup.find(tag_name, attrs)
                if body_content:
                    break
            
            if body_content:
                # More selective: look for step-like structures
                # Try to find numbered steps or step headers first
                step_headers = body_content.find_all(['h2', 'h3', 'h4'], string=lambda text: text and ('step' in text.lower() or text.strip().isdigit()))
                
                if step_headers:
                    # Extract content after each step header
                    for header in step_headers:
                        step_content = []
                        # Get next sibling elements until next header
                        for sibling in header.next_siblings:
                            if isinstance(sibling, str):
                                if sibling.strip():
                                    step_content.append(sibling.strip())
                            elif hasattr(sibling, 'name'):
                                if sibling.name in ['h2', 'h3', 'h4']:
                                    break
                                if sibling.name == 'p':
                                    text = sibling.get_text(strip=True)
                                    if text and len(text) > 15:
                                        step_content.append(text)
                                elif sibling.name in ['div', 'section']:
                                    text = sibling.get_text(strip=True)
                                    if text and len(text) > 15:
                                        step_content.append(text)
                        
                        if step_content:
                            combined = ' '.join(step_content)
                            if len(combined) > 20:
                                steps_text.append(combined)
                
                # If no step headers found, extract paragraphs but filter more carefully
                if not steps_text:
                    paragraphs = body_content.find_all('p')
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        # More aggressive filtering for fallback
                        if text and len(text) > 30:  # Longer minimum for fallback
                            # Exclude common non-step content
                            exclude_keywords = ['intro', 'introduction', 'overview', 'summary', 'conclusion', 
                                              'thanks', 'share', 'like', 'follow', 'subscribe', 'comment']
                            text_lower = text.lower()
                            if not any(keyword in text_lower[:100] for keyword in exclude_keywords):
                                # Check if it looks like step content (has action words or instructions)
                                action_words = ['cut', 'glue', 'attach', 'place', 'install', 'apply', 
                                              'measure', 'mark', 'connect', 'mount', 'prepare', 'build']
                                if any(word in text_lower for word in action_words) or len(text) > 100:
                                    steps_text.append(text)
                
                # Extract images from body
                for img in body_content.find_all('img'):
                    # Skip decorative images
                    img_class = img.get('class', [])
                    img_alt = img.get('alt', '').lower()
                    if any(skip in str(img_class).lower() or skip in img_alt for skip in ['icon', 'logo', 'avatar', 'button', 'ad']):
                        continue
                    
                    src = (img.get('src') or 
                          img.get('data-src') or 
                          img.get('data-lazy-src') or
                          img.get('data-original'))
                    if src:
                        absolute_url = make_absolute_url(src, base_url)
                        if absolute_url and absolute_url not in image_urls:
                            image_urls.append(absolute_url)

        # Validation: Check if we extracted meaningful data
        if title == "Unknown Title":
            logging.warning(f"Title extraction failed for {base_url}")
        
        if not steps_text and not image_urls:
            logging.warning(f"No steps or images found for {base_url}. HTML structure may have changed.")
            return None
        
        if not steps_text:
            logging.warning(f"No step text found for {base_url}, but images were found.")
        
        # Clean and validate extracted data
        steps_text = [step for step in steps_text if step and len(step.strip()) > 0]
        image_urls = [url for url in image_urls if url and url.startswith(('http://', 'https://'))]
        supplies = [s for s in supplies if s and len(s.strip()) > 0]

        raw_data = {
            "title": title,
            "author": author,
            "supplies": supplies,
            "steps": steps_text,
            "image_urls": image_urls
        }
        
        logging.info(f"Extracted: title='{title}', author='{author}', {len(steps_text)} steps, {len(image_urls)} images, {len(supplies)} supplies")
        return raw_data

    except Exception as e:
        logging.error(f"Error parsing HTML from {base_url}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    main()