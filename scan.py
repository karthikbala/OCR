#!/usr/bin/env python3
"""
scan.py - A simple OCR (Optical Character Recognition) script using Tesseract.

This script allows users to extract text from images using the Tesseract OCR engine.
"""

import argparse
import os
from pathlib import Path
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Required packages not found. Please install them using:")
    print("pip install pytesseract pillow")
    exit(1)

def scan_image(image_path, lang='eng', output_file=None):
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        lang (str): Language for OCR (default: 'eng')
        output_file (str, optional): Path to save the extracted text
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Extract text from the image
        text = pytesseract.image_to_string(img, lang=lang)
        
        # Save to output file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to {output_file}")
        
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    """Main function to parse arguments and run the OCR process."""
    parser = argparse.ArgumentParser(description='Extract text from images using OCR')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--lang', default='eng', help='Language for OCR (default: eng)')
    parser.add_argument('--output', '-o', help='Output file to save the extracted text')
    
    args = parser.parse_args()
    
    # Check if the image file exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Process the image
    text = scan_image(args.image, args.lang, args.output)
    
    # Print the extracted text if not saving to a file
    if text and not args.output:
        print("\nExtracted Text:")
        print("-" * 40)
        print(text)
        print("-" * 40)

if __name__ == "__main__":
    main()