#!/usr/bin/env python3
"""
PDF to CSV converter for Korean vocabulary extraction.
Handles multi-column table layouts and provides fallback text extraction.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pdfplumber
import PyPDF2


def detect_korean(text: str) -> bool:
    """Detect if text contains Korean characters (Hangul)."""
    korean_range = re.compile(r'[\uac00-\ud7af]')
    return bool(korean_range.search(text))


def clean_text(text: str) -> str:
    """Clean and normalize text by removing extra whitespace and special characters."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep Korean, English, numbers, and basic punctuation
    text = re.sub(r'[^\w\s가-힣.,!?()-]', '', text)
    
    return text.strip()


def extract_vocabulary_from_table(page) -> List[Tuple[str, str]]:
    """Extract vocabulary from table structure on a PDF page."""
    vocabulary_pairs = []
    
    try:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if not row or len(row) < 2:
                    continue
                
                # Handle multi-column layout: No. | 한글 | English | No. | 한글 | English
                processed_cells = []
                for cell in row:
                    if cell:
                        cleaned = clean_text(str(cell))
                        if cleaned and not cleaned.isdigit():  # Skip numbers (row indices)
                            processed_cells.append(cleaned)
                
                # Extract pairs of Korean-English from processed cells
                korean_words = []
                english_words = []
                
                for cell in processed_cells:
                    if detect_korean(cell):
                        korean_words.append(cell)
                    elif cell and not cell.isdigit():
                        english_words.append(cell)
                
                # Create pairs from Korean and English words
                min_pairs = min(len(korean_words), len(english_words))
                for i in range(min_pairs):
                    if korean_words[i] and english_words[i]:
                        vocabulary_pairs.append((korean_words[i], english_words[i]))
                        
    except Exception as e:
        print(f"Warning: Table extraction failed for page: {e}")
    
    return vocabulary_pairs


def extract_vocabulary_from_text(page) -> List[Tuple[str, str]]:
    """Fallback method to extract vocabulary from plain text."""
    vocabulary_pairs = []
    
    try:
        text = page.extract_text()
        if not text:
            return vocabulary_pairs
        
        lines = text.split('\n')
        
        for line in lines:
            line = clean_text(line)
            if not line or line.isdigit():
                continue
            
            # Split line into words and try to find Korean-English pairs
            words = line.split()
            korean_words = [w for w in words if detect_korean(w)]
            english_words = [w for w in words if not detect_korean(w) and not w.isdigit()]
            
            # Simple heuristic: if we have both Korean and English words, pair them
            if korean_words and english_words:
                # Take the first Korean and first English word as a pair
                vocabulary_pairs.append((korean_words[0], english_words[0]))
                
    except Exception as e:
        print(f"Warning: Text extraction failed for page: {e}")
    
    return vocabulary_pairs


def extract_vocabulary_from_pdf(pdf_path: str, use_text_fallback: bool = False) -> List[Tuple[str, str]]:
    """Extract Korean-English vocabulary pairs from PDF."""
    vocabulary_pairs = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Processing page {page_num}...")
                
                if use_text_fallback:
                    pairs = extract_vocabulary_from_text(page)
                else:
                    pairs = extract_vocabulary_from_table(page)
                    
                    # If table extraction yields no results, try text fallback
                    if not pairs:
                        print(f"No table data found on page {page_num}, trying text extraction...")
                        pairs = extract_vocabulary_from_text(page)
                
                vocabulary_pairs.extend(pairs)
                print(f"Found {len(pairs)} vocabulary pairs on page {page_num}")
                
    except Exception as e:
        print(f"Error processing PDF with pdfplumber: {e}")
        print("Trying PyPDF2 as fallback...")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    print(f"Processing page {page_num} with PyPDF2...")
                    text = page.extract_text()
                    
                    if text:
                        lines = text.split('\n')
                        for line in lines:
                            line = clean_text(line)
                            if not line or line.isdigit():
                                continue
                            
                            words = line.split()
                            korean_words = [w for w in words if detect_korean(w)]
                            english_words = [w for w in words if not detect_korean(w) and not w.isdigit()]
                            
                            if korean_words and english_words:
                                vocabulary_pairs.append((korean_words[0], english_words[0]))
                                
        except Exception as e2:
            print(f"Error with PyPDF2 fallback: {e2}")
            raise
    
    return vocabulary_pairs


def save_vocabulary_to_csv(vocabulary_pairs: List[Tuple[str, str]], output_path: str):
    """Save vocabulary pairs to CSV file."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['korean', 'english'])  # Header
            
            # Remove duplicates while preserving order
            seen = set()
            unique_pairs = []
            for korean, english in vocabulary_pairs:
                pair_key = (korean.lower(), english.lower())
                if pair_key not in seen:
                    seen.add(pair_key)
                    unique_pairs.append((korean, english))
            
            writer.writerows(unique_pairs)
            
        print(f"Successfully saved {len(unique_pairs)} unique vocabulary pairs to {output_path}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Extract Korean vocabulary from PDF and convert to CSV')
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--text-fallback', action='store_true', 
                       help='Use text extraction instead of table detection')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_pdf).exists():
        print(f"Error: Input file '{args.input_pdf}' does not exist.")
        sys.exit(1)
    
    if not args.input_pdf.lower().endswith('.pdf'):
        print(f"Error: Input file must be a PDF.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Extracting vocabulary from '{args.input_pdf}'...")
        vocabulary_pairs = extract_vocabulary_from_pdf(args.input_pdf, args.text_fallback)
        
        if not vocabulary_pairs:
            print("Warning: No vocabulary pairs found in the PDF.")
            print("Try using --text-fallback option if table detection is not working.")
            sys.exit(1)
        
        print(f"Found {len(vocabulary_pairs)} total vocabulary pairs")
        
        save_vocabulary_to_csv(vocabulary_pairs, args.output_csv)
        print(f"Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()