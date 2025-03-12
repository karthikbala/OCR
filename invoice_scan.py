#!/usr/bin/env python3
"""
invoice_scan.py - A specialized OCR tool for invoice processing.

This script extracts text from invoice images or PDFs using Mistral OCR,
then processes the text with OpenAI to generate structured JSON output.
"""

import os
import json
import argparse
import base64
import keyring
import re
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

# Try to import required packages
try:
    from PIL import Image
    import requests
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    # Import Mistral client
    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
    except ImportError as e:
        print(f"Error importing Mistral client: {e}")
        MISTRAL_AVAILABLE = False
    
    # Import OpenAI client
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except ImportError as e:
        print(f"Error importing OpenAI client: {e}")
        OPENAI_AVAILABLE = False
        
except ImportError:
    print("Required packages not found. Please install them using:")
    print("pip install pillow requests rich mistralai openai")
    exit(1)

# Check if required clients are available
if not MISTRAL_AVAILABLE or not OPENAI_AVAILABLE:
    print("Missing required API clients. Please install them using:")
    if not MISTRAL_AVAILABLE:
        print("pip install mistralai")
    if not OPENAI_AVAILABLE:
        print("pip install openai")
    exit(1)

# Constants for API key storage
APP_NAME = "OCR_Tool"
OPENAI_KEY_NAME = "openai_api_key"
MISTRAL_KEY_NAME = "mistral_api_key"

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_stored_api_key(service_name: str) -> Optional[str]:
    """
    Get stored API key from the system keyring.
    
    Args:
        service_name (str): Name of the service (e.g., 'openai_api_key', 'mistral_api_key')
        
    Returns:
        str: API key if found, None otherwise
    """
    try:
        return keyring.get_password(APP_NAME, service_name)
    except Exception:
        return None

def store_api_key(service_name: str, api_key: str) -> bool:
    """
    Store API key in the system keyring.
    
    Args:
        service_name (str): Name of the service (e.g., 'openai_api_key', 'mistral_api_key')
        api_key (str): API key to store
    """
    try:
        keyring.set_password(APP_NAME, service_name, api_key)
        print(f"API key for {service_name} has been securely stored.")
        return True
    except Exception as e:
        print(f"Error storing API key: {e}")
        return False

def extract_text_with_mistral(file_path: str, api_key: str) -> Optional[str]:
    """
    Extract text from an image or PDF using Mistral API with OCR capabilities.
    For PDFs, this function uses pdf2image to convert pages to images and then processes each image.
    
    Args:
        file_path (str): Path to the image or PDF file
        api_key (str): Mistral API key
        
    Returns:
        str: Extracted text from the image or PDF
    """
    # Check if the file is a PDF
    is_pdf = file_path.lower().endswith('.pdf')
    
    try:
        # Initialize the Mistral client
        client = Mistral(api_key=api_key)
        
        try:
            # Different processing for PDF vs image
            if is_pdf:
                # For PDFs, use pdf2image to convert pages to images
                try:
                    # Try to import pdf2image for PDF processing
                    from pdf2image import convert_from_path
                    
                    print(f"Processing PDF file: {file_path}")
                    
                    # Convert PDF to images
                    pages = convert_from_path(file_path)
                    print(f"PDF has {len(pages)} pages")
                    
                    # Process each page and combine the results
                    full_text = ""
                    for i, page in enumerate(pages):
                        print(f"Processing page {i+1}/{len(pages)}...")
                        
                        # Save the page as a temporary image file
                        temp_image_path = f"/tmp/pdf_page_{i+1}.jpg"
                        page.save(temp_image_path, "JPEG")
                        
                        # Encode the image to base64
                        base64_image = encode_image_to_base64(temp_image_path)
                        
                        # Process the image with Mistral OCR
                        ocr_response = client.ocr.process(
                            model="mistral-ocr-latest",
                            document={
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        )
                        
                        # Extract text from the response
                        page_text = ""
                        if ocr_response and hasattr(ocr_response, 'pages') and len(ocr_response.pages) > 0:
                            for ocr_page in ocr_response.pages:
                                if hasattr(ocr_page, 'text') and ocr_page.text:
                                    page_text += ocr_page.text
                                elif hasattr(ocr_page, 'markdown') and ocr_page.markdown:
                                    page_text += ocr_page.markdown
                        
                        full_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                        
                        # Remove temporary file
                        os.remove(temp_image_path)
                    
                    return full_text.strip()
                    
                except ImportError:
                    print("pdf2image library not found. Please install it to process PDFs:")
                    print("pip install pdf2image poppler-utils")
                    return "Error: pdf2image library required for PDF processing."
            else:
                # Process a regular image file
                print(f"Processing image file: {file_path}")
                
                # Encode the image to base64
                base64_image = encode_image_to_base64(file_path)
                
                # Process the image with Mistral OCR
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                )
                
                # Extract the text from the OCR response
                if ocr_response and hasattr(ocr_response, 'pages') and len(ocr_response.pages) > 0:
                    # Extract text from the response
                    extracted_text = ""
                    for page in ocr_response.pages:
                        if hasattr(page, 'text') and page.text:
                            extracted_text += page.text + "\n\n"
                        elif hasattr(page, 'markdown') and page.markdown:
                            extracted_text += page.markdown + "\n\n"
                    
                    return extracted_text.strip()
                else:
                    print("Error: No text found in the OCR response.")
                    return None
                
        except Exception as e:
            print(f"Error processing with Mistral OCR: {e}")
            return None
            
    except Exception as e:
        print(f"Error initializing Mistral client: {e}")
        return None

def clean_extracted_text(text: str) -> str:
    """
    Clean the extracted text to remove problematic characters and formatting.
    
    Args:
        text (str): The raw extracted text from OCR
        
    Returns:
        str: Cleaned text ready for parsing
    """
    if not text:
        return ""
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove special characters that might interfere with JSON parsing
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Trim lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text

def parse_invoice_with_openai(text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Parse the extracted text from an invoice using OpenAI's GPT model.
    
    Args:
        text (str): The extracted text from the invoice
        api_key (str): OpenAI API key
        
    Returns:
        dict: Structured invoice data in JSON format
    """
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Clean the text
        cleaned_text = clean_extracted_text(text)
        
        # Create a system prompt for invoice parsing
        system_prompt = """
        You are an expert invoice parser. Extract the following information from the invoice text:
        
        1. Invoice Number
        2. Invoice Date
        3. Due Date
        4. Vendor Name
        5. Vendor Address
        6. Vendor Contact (phone, email)
        7. Customer/Client Name
        8. Customer/Client Address
        9. Line Items (with description, quantity, unit price, and total for each)
        10. Subtotal
        11. Tax Amount
        12. Total Amount
        13. Payment Terms
        14. Payment Method (if specified)
        15. Currency
        
        Format your response as a valid JSON object with the following structure:
        {
            "invoice_number": "string or null if not found",
            "invoice_date": "string in YYYY-MM-DD format or null if not found",
            "due_date": "string in YYYY-MM-DD format or null if not found",
            "vendor": {
                "name": "string or null if not found",
                "address": "string or null if not found",
                "contact": "string or null if not found"
            },
            "customer": {
                "name": "string or null if not found",
                "address": "string or null if not found"
            },
            "line_items": [
                {
                    "description": "string",
                    "quantity": number or null if not found,
                    "unit_price": number or null if not found,
                    "total": number or null if not found
                }
            ],
            "summary": {
                "subtotal": number or null if not found,
                "tax": number or null if not found,
                "total": number or null if not found,
                "currency": "string or null if not found"
            },
            "payment": {
                "terms": "string or null if not found",
                "method": "string or null if not found"
            }
        }
        
        If you cannot find a specific piece of information, use null for that field. Do not include any explanations or notes in your response, just the JSON object.
        """
        
        # Call the OpenAI API to parse the invoice
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse the following invoice text:\n\n{cleaned_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Extract the JSON response
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            json_str = response.choices[0].message.content
            
            try:
                # Parse the JSON response
                parsed_data = json.loads(json_str)
                return parsed_data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {json_str}")
                return None
        else:
            print("Error: No valid response from OpenAI.")
            return None
            
    except Exception as e:
        print(f"Error parsing invoice with OpenAI: {e}")
        return None

def extract_text_with_openai(file_path: str, api_key: str) -> Optional[str]:
    """
    Extract text from an image or PDF using OpenAI's Vision API.
    
    Args:
        file_path (str): Path to the image or PDF file
        api_key (str): OpenAI API key
        
    Returns:
        str: Extracted text from the image or PDF
    """
    # Check if the file is a PDF
    is_pdf = file_path.lower().endswith('.pdf')
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Different processing for PDF vs image
        if is_pdf:
            try:
                # Try to import pdf2image for PDF processing
                from pdf2image import convert_from_path
                
                print(f"Processing PDF file: {file_path}")
                
                # Convert PDF to images
                pages = convert_from_path(file_path)
                print(f"PDF has {len(pages)} pages")
                
                # Process each page and combine the results
                full_text = ""
                for i, page in enumerate(pages):
                    print(f"Processing page {i+1}/{len(pages)}...")
                    
                    # Save the page as a temporary image file
                    temp_image_path = f"/tmp/pdf_page_{i+1}.jpg"
                    page.save(temp_image_path, "JPEG")
                    
                    # Encode the image to base64
                    base64_image = encode_image_to_base64(temp_image_path)
                    
                    # Process the image with OpenAI Vision
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract all text from this image, preserving the layout as much as possible. Include all details such as dates, numbers, and any tabular data."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=4096
                    )
                    
                    # Extract text from the response
                    page_text = ""
                    if response and hasattr(response, 'choices') and len(response.choices) > 0:
                        page_text = response.choices[0].message.content
                    
                    full_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    
                    # Remove temporary file
                    os.remove(temp_image_path)
                
                return full_text.strip()
                
            except ImportError:
                print("pdf2image library not found. Please install it to process PDFs:")
                print("pip install pdf2image poppler-utils")
                return "Error: pdf2image library required for PDF processing."
        else:
            # Process a regular image file
            print(f"Processing image file: {file_path}")
            
            # Encode the image to base64
            base64_image = encode_image_to_base64(file_path)
            
            # Process the image with OpenAI Vision
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this image, preserving the layout as much as possible. Include all details such as dates, numbers, and any tabular data."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            # Extract text from the response
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print("Error: No valid response from OpenAI.")
                return None
                
    except Exception as e:
        print(f"Error extracting text with OpenAI: {e}")
        return None

def process_invoice(file_path: str, mistral_api_key: str = None, openai_api_key: str = None, 
                   output_file: str = None, use_openai_for_ocr: bool = True) -> Optional[Dict[str, Any]]:
    """
    Process an invoice image or PDF and extract structured data.
    
    Args:
        file_path (str): Path to the invoice image or PDF
        mistral_api_key (str, optional): Mistral API key
        openai_api_key (str, optional): OpenAI API key
        output_file (str, optional): Path to save the JSON output
        use_openai_for_ocr (bool): Whether to use OpenAI for OCR instead of Mistral
        
    Returns:
        dict: Structured invoice data in JSON format
    """
    # Create console for rich output
    console = Console()
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' not found")
        return None
    
    # Get API keys if not provided
    if not mistral_api_key and not use_openai_for_ocr:
        mistral_api_key = get_stored_api_key(MISTRAL_KEY_NAME)
        if not mistral_api_key:
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        
        if not mistral_api_key:
            console.print("[bold red]Error:[/bold red] Mistral API key not found. Please set it using:")
            console.print("python -c \"import keyring; keyring.set_password('OCR_Tool', 'mistral_api_key', 'your-api-key')\"")
            console.print("Or set the MISTRAL_API_KEY environment variable.")
            return None
    
    if not openai_api_key:
        openai_api_key = get_stored_api_key(OPENAI_KEY_NAME)
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not openai_api_key:
            console.print("[bold red]Error:[/bold red] OpenAI API key not found. Please set it using:")
            console.print("python -c \"import keyring; keyring.set_password('OCR_Tool', 'openai_api_key', 'your-api-key')\"")
            console.print("Or set the OPENAI_API_KEY environment variable.")
            return None
    
    # Extract text from the invoice
    console.print(f"[bold]Processing file:[/bold] {file_path}")
    
    with console.status("[bold green]Extracting text from invoice...[/bold green]", spinner="dots"):
        if use_openai_for_ocr:
            extracted_text = extract_text_with_openai(file_path, openai_api_key)
        else:
            extracted_text = extract_text_with_mistral(file_path, mistral_api_key)
    
    if not extracted_text:
        console.print("[bold red]Error:[/bold red] Failed to extract text from the invoice.")
        return None
    
    # Display the extracted text
    console.print("\n[bold]Extracted Text:[/bold]")
    console.print(Panel(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""), 
                        title="OCR Result (truncated)", 
                        expand=False))
    
    # Parse the invoice text
    with console.status("[bold green]Parsing invoice data...[/bold green]", spinner="dots"):
        parsed_data = parse_invoice_with_openai(extracted_text, openai_api_key)
    
    if not parsed_data:
        console.print("[bold red]Error:[/bold red] Failed to parse invoice data.")
        return None
    
    # Format the JSON output
    json_str = json.dumps(parsed_data, indent=2)
    
    # Display the parsed data
    console.print("\n[bold]Parsed Invoice Data:[/bold]")
    console.print(Syntax(json_str, "json", theme="monokai", line_numbers=True))
    
    # Save the output to a file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        console.print(f"\n[bold green]Invoice data saved to:[/bold green] {output_file}")
    
    return parsed_data

def main():
    """Main function to parse arguments and run the invoice processing."""
    parser = argparse.ArgumentParser(description='Extract and parse invoice data using OCR')
    parser.add_argument('invoice', help='Path to the invoice image or PDF file')
    parser.add_argument('--output', '-o', help='Output file to save the JSON data')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI for OCR instead of Mistral')
    parser.add_argument('--mistral-key', help='Mistral API key (optional)')
    parser.add_argument('--openai-key', help='OpenAI API key (optional)')
    
    args = parser.parse_args()
    
    # Process the invoice
    process_invoice(
        args.invoice,
        mistral_api_key=args.mistral_key,
        openai_api_key=args.openai_key,
        output_file=args.output,
        use_openai_for_ocr=args.use_openai
    )

if __name__ == "__main__":
    main()