#!/usr/bin/env python3
"""
Color data loader for reading Wikipedia colors JSON file
"""

import json
import os
import re

def clean_color_name(name):
    """
    Clean color name by removing spaces, parentheses, and other problematic characters
    """
    # Remove parentheses and their contents
    cleaned = re.sub(r'\([^)]*\)', '', name)
    
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Remove other special characters but keep letters, numbers, and underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', cleaned)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned

def load_colors_from_json(filename='colors_a_f.json'):
    """
    Load color names from the JSON file created by scrape_colors.py
    
    Args:
        filename: Path to the JSON file containing color data
        
    Returns:
        List of color names as strings
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Color data file '{filename}' not found. Please run scrape_colors.py first.")
    
    print(f"Loading colors from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        colors_data = json.load(f)
    
    # Extract and clean the color names
    raw_color_names = [color['Name'] for color in colors_data if 'Name' in color and color['Name']]
    color_names = [clean_color_name(name) for name in raw_color_names]
    
    # Remove any empty names that might result from cleaning
    color_names = [name for name in color_names if name]
    
    print(f"Loaded and cleaned {len(color_names)} colors")
    
    return color_names

def get_color_names():
    """
    Convenience function to get color names list
    """
    return load_colors_from_json() 