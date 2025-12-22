from PyPDF2 import PdfReader
import docx2txt
import mammoth


def extract_text_from_pdf(path_or_file):
    """Accepts a file path or a file-like object and returns extracted PDF text.

    Returns an empty string on error so higher-level code can detect missing content
    and avoid passing error messages to the LLM as resume text.
    """
    text = ""
    try:
        # PdfReader accepts both file paths and file-like objects
        reader = PdfReader(path_or_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        # Log but return empty string (do not return the raw exception text)
        try:
            print(f"Error extracting text from PDF: {e}")
        except Exception:
            pass
        return ""

    return text.strip()


def extract_text_from_docx(path_or_file):
    """Use docx2txt which accepts a path. Accepts file-like or path."""
    try:
        if hasattr(path_or_file, "name"):
            path = path_or_file.name
        else:
            path = path_or_file
        return docx2txt.process(path).strip()
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"


def extract_text_from_doc(path_or_file):
    """Use mammoth to extract raw text from old .doc files (accepts file-like or path)."""
    try:
        if hasattr(path_or_file, "read"):
            # mammoth works with binary file-like objects
            path_or_file.seek(0)
            result = mammoth.extract_raw_text(path_or_file)
        else:
            with open(path_or_file, "rb") as f:
                result = mammoth.extract_raw_text(f)
        return (result.value or "").strip()
    except Exception as e:
        return f"Error extracting text from DOC: {e}"


def parse_uploaded_file(file_path: str, filename: str) -> str:
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif filename.endswith(".doc"):
        return extract_text_from_doc(file_path)
    else:
        raise ValueError("Unsupported file format")
