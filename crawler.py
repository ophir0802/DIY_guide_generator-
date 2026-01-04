import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from urllib.parse import urljoin, urlparse
import time
import random
import json
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Data Model Definition

class Guide(BaseModel):
    """
    Pydantic model to define the strict structure of a Guide.
    """
    title: str
    author: str
    additional_text_boxes: List[str]
    supplies: List[str]
    steps: List[List[str]] # List of [headline, content] tuples
    image_urls: List[str]
    url: str

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
        
        response = requests.get(url, headers=headers, timeout=15)
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

def save_single_guide(guide_data: dict, output_dir: str = "raw_data"):
    """
    Validates and saves a single guide to a JSON file.
    """
    try:
        # Validate
        guide = Guide(**guide_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename from title
        safe_title = re.sub(r'[\\/*?:"<>|]', "", guide.title) # Remove invalid chars
        safe_title = safe_title.replace(" ", "_").lower()[:50] # Slugify roughly
        filename = f"{safe_title}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Check if file exists to avoid overwriting (optional, but good practice)
        # For now, we overwrite or add number? Let's overwrite as it might be an update.
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(guide.model_dump(), f, indent=4, ensure_ascii=False)
            
        logging.info(f"Saved guide to {file_path}")
        return True
        
    except ValidationError as e:
        logging.error(f"Validation failed for guide '{guide_data.get('title', 'Unknown')}': {e}")
        return False
    except IOError as e:
        logging.error(f"File write error: {e}")
        return False

def fetch_category_links(category_url: str) -> List[str]:
    """
    Fetches article links from a category page, filtering for 'How to' articles.
    """
    html = fetch_page(category_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    article_links = []
    
    # Strategy: Find all links, filter by "how-to" in text or href
    # Note: DIY.com often has "How to..." as link text
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        
        # Filter logic:
        # 1. Must be an article (heuristic: not a category page, often has /stry/ or just specific pattern)
        # 2. Text or Title must start with "How to" (case insensitive)
        
        if "how to" in text.lower() or "how-to" in href.lower():
            full_url = make_absolute_url(href, category_url)
            if full_url not in article_links and full_url != category_url:
                article_links.append(full_url)
    
    logging.info(f"Found {len(article_links)} potential 'How-to' articles in category.")
    return article_links

def parse_html(html_content: str, base_url: str) -> Optional[dict]:
    """
    Parses raw HTML to extract guide data specific to doityourself.com
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # --- Extract Title ---
        # User specified: class="how-to__article-title"
        title_tag = soup.find('h1', class_='how-to__article-title')
        if not title_tag:
             # Fallback
             title_tag = soup.find('h1')
        
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
        
        # Double check "How to" in title if strict filtering is needed
        if "how to" not in title.lower():
            logging.warning(f"Skipping {base_url}: Title does not contain 'How to'")
            return None # Strict filter as requested? "we are going to need only containers with headlines starts with 'how-to'"
            
        if not title.lower().startswith('how to'):
             logging.warning(f"Skipping {base_url}: Title does not start with 'How to'")
             return None

        # --- Extract Author ---
        # Common pattern: <span class="author-name"> or similar
        author = "Unknown Author"
        author_tag = soup.find(class_='author-name') or soup.find(rel='author')
        if author_tag:
            author = author_tag.get_text(strip=True)

        # --- Extract Additional Text Boxes ---
        # Strategy: capture all significant text paragraphs that are NOT part of supplies or steps
        additional_text_boxes = []
        
        article_body = soup.find(class_='article-body') or soup.find(id='article-body')
        
        if article_body:
            # We iterate through direct children or all elements?
            # Safer to iterate all elements and skip those we know are steps/supplies?
            # Or just grab all 'p' and filter? 
            # Given structure, let's grab all text immediately in 'p' tags inside article body
            
            # Helper to check if text belongs to a step or supply
            # This is hard because steps parsing is dynamic.
            
            # Alternative Strategy:
            # We already have logical sections for Supplies and Steps.
            # We can traverse the siblings again and classify them.
            
            # Let's try a scan of the article body.
            
            for elem in article_body.descendants:
                if elem.name == 'p':
                    text = elem.get_text(strip=True)
                    if len(text) > 20: 
                        # Heuristic: Check if this text is already in supplies or steps?
                        # This might be slow if steps are huge.
                        
                        # Better: The previous parsers (supplies/steps) extracted specific parts.
                        # Maybe we can just extract everything not "Step X" or "Things Needed"
                        
                        # Let's assume additional text matches 'p' content that doesn't start with "Step"
                        # and isn't in a list <ul>/<ol>
                        
                        is_step = re.match(r"^Step\s+\d+", text, re.I)
                        if not is_step:
                             # Check if it's a list item (parent is li)
                             if elem.parent.name != 'li':
                                  additional_text_boxes.append(text)
        
        # Deduplication might be needed if traversal is recursive
        
        # Refined Strategy (simpler):
        # Intro text: between Title and first header
        # Outro text: after last step?
        
        # Let's stick to the "Intro" logic but expand it to any non-header non-list text blocks?
        # Re-using the "Content" logic but making it a list
        
        content_candidates = []
        start_elem = author_tag if author_tag else title_tag
        if start_elem:
            curr = start_elem.next_sibling
            while curr:
                if hasattr(curr, 'name'):
                    if curr.name in ['h2', 'h3', 'h4']:
                         # If it's a Step header or Supplies header, we skip it and its content?
                         # Wait, we want to capture text "before and after".
                         pass
                    
                    if curr.name == 'p':
                         text = curr.get_text(strip=True)
                         if len(text) > 30: # Filter small
                             # Exclude if it looks like a step starter
                             if not re.match(r"^Step\s+\d+", text, re.I):
                                 content_candidates.append(text)
                                 
                curr = curr.next_sibling
                
        if content_candidates:
            additional_text_boxes = content_candidates

        # --- Extract Supplies ---
        supplies = []
        supplies_elem_attr = None # Initialize to avoid UnboundLocalError
        
        # Strategy 1: Specific Class Structure (tool-and-material__content)
        # Look for div with class 'tool-and-material__content' and inside 'col-12'
        tools_container = soup.find('div', class_='tool-and-material__content')
        if tools_container:
            # Find all col-12 divs inside
            items = tools_container.find_all('div', class_='col-12')
            for item in items:
                text = item.get_text(strip=True)
                if text and text not in supplies:
                    supplies.append(text)

        if not supplies:
            # Strategy 2: Look for specific attribute 'click0label'
            # User mentioned: click0label="what You'll Need"
            supplies_elem_attr = soup.find(attrs={"click0label": re.compile(r"what you'll need|supplies|things you'll need", re.I)})
            
            if supplies_elem_attr:
                # This element might be the header or the container
                # If it's a header (h2, div), look for siblings
                # If it contains ul/ol, extract directly
                
                if supplies_elem_attr.name in ['ul', 'ol']:
                     supplies = [li.get_text(strip=True) for li in supplies_elem_attr.find_all('li') if li.get_text(strip=True)]
                else:
                     # Check children first
                     ul = supplies_elem_attr.find('ul')
                     if ul:
                         supplies = [li.get_text(strip=True) for li in ul.find_all('li') if li.get_text(strip=True)]
                     
                     if not supplies:
                         # Check next siblings like before
                         curr = supplies_elem_attr.find_next_sibling()
                         found_list = False
                         while curr and not found_list:
                            if curr.name in ['ul', 'ol']:
                                supplies = [li.get_text(strip=True) for li in curr.find_all('li') if li.get_text(strip=True)]
                                found_list = True
                            elif curr.name in ['div', 'section']:
                                 ul = curr.find('ul')
                                 if ul:
                                     supplies = [li.get_text(strip=True) for li in ul.find_all('li') if li.get_text(strip=True)]
                                     found_list = True
                            
                            if hasattr(curr, 'name') and curr.name in ['h2', 'h3', 'h4']:
                                break
                            curr = curr.next_sibling

        if not supplies:
            # Strategy 3: Fallback to text header search
            # Find header - include "What You Will Need" and "Supplies"
            supplies_header = soup.find(['h2', 'h3', 'h4'], string=re.compile(r"Things You'll Need|Supplies|What You Will Need", re.I))
            
            if supplies_header:
                # Usually followed by a list <ul> or <div>
                # Get next sibling that is a list or container
                curr = supplies_header.find_next_sibling()
                # aggressive search for list
                found_list = False
                while curr and not found_list:
                    if curr.name in ['ul', 'ol']:
                        supplies = [li.get_text(strip=True) for li in curr.find_all('li') if li.get_text(strip=True)]
                        found_list = True
                    elif curr.name in ['div', 'section']:
                         # Check if list inside
                         ul = curr.find('ul')
                         if ul:
                             supplies = [li.get_text(strip=True) for li in ul.find_all('li') if li.get_text(strip=True)]
                             found_list = True
                    
                    if hasattr(curr, 'name') and curr.name in ['h2', 'h3', 'h4']:
                        # Hit next section
                        break
                        
                    curr = curr.next_sibling
        
        # --- Extract Steps and Images ---
        steps_data = [] # List of [headline, text]
        image_urls = []
        
        # DoItYourself often uses specific headers for steps like "Step 1: ..."
        
        # Find all step headers
        step_headers = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r"^Step\s+\d+", re.I))
        
        if step_headers:
            for header in step_headers:
                # Content is between this header and the next header
                step_content = []
                current_element = header.next_sibling
                
                header_text = header.get_text(strip=True)
                
                while current_element:
                    if hasattr(current_element, 'name') and current_element.name in ['h2', 'h3', 'h4']:
                        # Check if it's another step or something else
                        if re.match(r"^Step\s+\d+", current_element.get_text(strip=True) or "", re.I):
                             break # Next step
                        # Else it might be a subsection, keep going?
                        # Usually "Step" headers are at same level.
                        if current_element.name == header.name: 
                            break 
                            
                    if isinstance(current_element, str):
                        text = current_element.strip()
                        if text:
                            step_content.append(text)
                    elif hasattr(current_element, 'name'):
                         # Extract text
                        if current_element.name in ['p', 'div', 'li']: 
                            text = current_element.get_text(strip=True)
                            if text:
                                step_content.append(text)
                        
                        # Extract Images in this step
                        if current_element.name == 'img':
                            imgs = [current_element]
                        else:
                            imgs = current_element.find_all('img')
                            
                        for img in imgs:
                             src = img.get('src')
                             if src:
                                 abs_url = make_absolute_url(src, base_url)
                                 if abs_url not in image_urls:
                                     image_urls.append(abs_url)

                    current_element = current_element.next_sibling
                
                full_step_text = " ".join(step_content).strip()
                if full_step_text:
                    steps_data.append([header_text, full_step_text])
        else:
            # Fallback for guides without explicit "Step X" headers
            # Maybe just content?
            # Look for article body
            article_body = soup.find(class_='article-body') or soup.find(id='article-body')
            if article_body:
                # Just get paragraphs
                current_step_text = []
                for p in article_body.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 20:
                        current_step_text.append(text)
                
                if current_step_text:
                     steps_data.append(["Instruction", " ".join(current_step_text)]) # Single step fallback

                # Images
                for img in article_body.find_all('img'):
                     src = img.get('src')
                     if src:
                         abs_url = make_absolute_url(src, base_url)
                         if abs_url not in image_urls:
                             image_urls.append(abs_url)

        if not steps_data:
             logging.warning(f"No steps extracted for {base_url}")
             return None

        # Populate data
        data = {
            "title": title,
            "author": author,
            "additional_text_boxes": additional_text_boxes,
            "supplies": supplies,
            "steps": steps_data,
            "image_urls": image_urls,
            "url": base_url
        }
        
        return data

    except Exception as e:
        logging.error(f"Error parsing HTML from {base_url}: {e}", exc_info=True)
        return None

# --- 4. Execution Logic (Main Loop) ---

def main():
    category_url = "https://www.doityourself.com/scat/freezer"
    
    print(f"--- Starting Crawler using category: {category_url} ---")
    
    # 1. Get Links
    target_urls = fetch_category_links(category_url)
    
    if not target_urls:
        print("No articles found in category.")
        return

    print(f"Found {len(target_urls)} articles to process.")
    
    # 2. Main Loop
    success_count = 0
    for url in target_urls:
        logging.info(f"Processing: {url}")
        
        # Step A: Fetch
        html = fetch_page(url)
        
        if html:
            # Step B: Parse
            data = parse_html(html, url)
            
            if data:
                # Step C: Save immediately
                if save_single_guide(data):
                    success_count += 1
                    logging.info(f"Successfully scraped: {data['title']}")
                    
                    if success_count >= 5:
                        print("Reached limit of 5 guides. Stopping.")
                        break
            else:
                logging.warning(f"Failed to parse content from {url}")
        else:
            logging.warning(f"Skipping {url} due to fetch error.")

    print(f"--- Crawler Finished. Processed {success_count} guides. ---")

if __name__ == "__main__":
    main()