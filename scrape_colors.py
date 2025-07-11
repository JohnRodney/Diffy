#!/usr/bin/env python3
"""
Simple script to scrape Wikipedia colors table and save to JSON
"""

import requests
from bs4 import BeautifulSoup
import json

def scrape_colors():
    """Scrape colors from Wikipedia table"""
    
    url = "https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("Fetching Wikipedia page...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the colors table (first table on the page)
    table = soup.find('table')
    if not table:
        raise ValueError("No table found on the page")
    
    # Get column headers from TH elements
    header_row = table.find('tr')
    if not header_row:
        raise ValueError("No header row found in table")
        
    th_elements = header_row.find_all(['th', 'td'])
    columns = [th.get_text(strip=True) for th in th_elements]
    
    print(f"Found columns: {columns}")
    
    # Extract data from table body rows
    colors = []
    rows = table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        td_elements = row.find_all(['td', 'th'])
        if len(td_elements) >= len(columns):
            # Create color dict using column names as keys
            color = {}
            for i, col_name in enumerate(columns):
                if i < len(td_elements):
                    color[col_name] = td_elements[i].get_text(strip=True)
            
            # Only add if it has a name and hex value
            if color.get('Name') and color.get('Hex(RGB)'):
                colors.append(color)
    
    return colors

def save_to_json(colors, filename='colors_a_f.json'):
    """Save colors to JSON file"""
    print(f"Saving {len(colors)} colors to {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(colors, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {filename}")

def main():
    try:
        colors = scrape_colors()
        save_to_json(colors)
        
        print(f"\nCompleted! Found {len(colors)} colors.")
        
        # Show first few examples
        if colors:
            print("\nFirst few colors:")
            for i, color in enumerate(colors[:3]):
                print(f"{i+1}. {color['Name']}: {color['Hex(RGB)']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 