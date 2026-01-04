import requests
from bs4 import BeautifulSoup
import logging
import sys

# Configure IO to handle utf-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(message)s')

def fetch_and_analyze():
    base_url = "https://www.doityourself.com"
    cat_url = f"{base_url}/scat/freezer"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Fetching category: {cat_url}")
    try:
        resp = requests.get(cat_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find article links
        # Heuristic: look for links with "How to" in text
        article_links = []
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            href = a['href']
            # Check for explicitly "How to" or just look for article-like urls
            if "how to" in text.lower():
                full_url = href if href.startswith('http') else base_url + href
                article_links.append((text, full_url))
        
        print(f"Found {len(article_links)} 'How to' articles.")
        for title, url in article_links[:3]:
            print(f" - {title}: {url}")
            
        if not article_links:
            print("No 'How to' links found. Dumping first 5 links to check structure:")
            for a in soup.find_all('a', href=True)[:5]:
                 print(f" - {a.get_text(strip=True)}: {a['href']}")
            return

        # Analyze the first one
        target_url = article_links[0][1]
        print(f"\nAnalyzing Article: {target_url}")
        
        resp = requests.get(target_url, headers=headers, timeout=10)
        resp.raise_for_status()
        article_soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Check specific selectors
        print("\n--- Title candidates ---")
        h1 = article_soup.find('h1')
        if h1:
            print(f"H1: {h1.get_text(strip=True)}")
            print(f"H1 classes: {h1.get('class')}")
            print(f"H1 id: {h1.get('id')}")

        print("\n--- Potential Supplies Section ---")
        # Look for "Things You'll Need"
        for header in article_soup.find_all(['h2', 'h3', 'h4']):
            if "thing" in header.get_text(strip=True).lower() or "suppl" in header.get_text(strip=True).lower():
                print(f"Found header: {header.get_text(strip=True)}")
                # Check siblings
                sibling = header.find_next_sibling()
                if sibling:
                    print(f"Next sibling tag: {sibling.name} class: {sibling.get('class')}")
        
        print("\n--- Potential Steps ---")
        # Look for headers starting with "Step"
        step_headers = article_soup.find_all(['h2', 'h3', 'h4'], string=lambda t: t and "step" in t.lower())
        print(f"Found {len(step_headers)} step headers.")
        for sh in step_headers[:2]:
            print(f"Header: {sh.get_text(strip=True)}")
            print(f"Parent class: {sh.parent.get('class')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_analyze()
