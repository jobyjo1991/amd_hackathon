from PyPDF2 import PdfReader

def extract_clean_text(file_buffer) -> str:
    """
    Parses structural binary streams from uploaded PDFs into clean strings.
    """
    try:
        reader = PdfReader(file_buffer)
        parsed_pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(parsed_pages)
    except Exception as e:
        return f"Extraction Failure: {str(e)}"