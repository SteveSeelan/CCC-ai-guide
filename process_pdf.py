import fitz  # PyMuPDF library
import os
import sys

def extract_text(pdf_path, txt_path, start_page=3, reload=False):
    """
    Extracts text from a large PDF file page by page and saves it to a .txt file.

    Args:
        pdf_path (str): The full path to the input PDF file.
        txt_path (str): The full path for the output text file.
    """

    print(f"Extracting text from {pdf_path} to {txt_path}...")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 and not reload:
        print(f"Text file already exists at {txt_path}. Skipping extraction.")
        return txt_path
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None

    doc = None  # Initialize doc to None
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        print(f"PDF opened successfully. Total pages: {total_pages}")

        # Open the output text file with write permissions and UTF-8 encoding
        with open(txt_path, "w", encoding="utf-8") as outfile:
            # Iterate through each page of the PDF
            for page_num, page in enumerate(doc, start=start_page):
                # Extract text from the current page
                text = page.get_text()
                
                # Write the extracted text to the output file
                outfile.write(text)
                
                # Optional: Write a page separator for clarity
                # outfile.write(f"\n--- Page {page_num} ---\n")

                # Print progress to the console
                # This uses sys.stdout.write and \r to update the line in place
                sys.stdout.write(f"\rProcessing page {page_num}/{total_pages}...")
                sys.stdout.flush()

        # Print a final message to the console on a new line
        print(f"\n\nExtraction complete. Text saved to '{txt_path}'")
        return txt_path
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Ensure the PDF document is closed to free up resources
        if doc:
            doc.close()
