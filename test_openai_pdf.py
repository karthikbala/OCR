#!/usr/bin/env python3
"""
Test script for the updated scan_with_openai function with PDF support.
"""

import os
import sys
import keyring
from scan import scan_with_openai, get_stored_api_key

# Constants
APP_NAME = "OCR_Tool"
OPENAI_KEY_NAME = "openai_api_key"

def main():
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_openai_pdf.py <path_to_pdf_or_image>")
        return 1
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return 1
    
    # Get the OpenAI API key
    api_key = get_stored_api_key(OPENAI_KEY_NAME)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key not found. Please set it using:")
        print("python -c \"import keyring; keyring.set_password('OCR_Tool', 'openai_api_key', 'your-api-key')\"")
        print("Or set the OPENAI_API_KEY environment variable.")
        return 1
    
    print(f"Processing file: {file_path}")
    
    # Process the file with OpenAI
    result = scan_with_openai(file_path, api_key)
    
    if result:
        print("\n----- EXTRACTED TEXT -----")
        print(result)
        print("--------------------------\n")
        
        # Save the result to a text file
        output_file = f"{os.path.splitext(file_path)[0]}_openai_output.txt"
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"Results saved to: {output_file}")
        return 0
    else:
        print("Error: Failed to extract text from the file.")
        return 1

if __name__ == "__main__":
    sys.exit(main())