import re
from pylatexenc.latex2text import LatexNodes2Text


# def remove_added_content(text: str) -> str:
#     patterns = [
#         r"Here is the updated resume.*",
#         r"I have updated your resume.*",
#         r"Sure! Here's.*",
#     ]
#     for pattern in patterns:
#         text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
#     return text.strip()
# def clean_resume_response(text: str) -> str:
#     text = remove_added_content(text)
#     return text.strip()

def clean_resume_response(response: str) -> str:
    """
    Simple cleaning function that removes common intro phrases but is conservative.
    """
    response = response.strip()
    
    # Remove common intro lines (only if they're at the very beginning)
    intro_phrases = [
        "Here's the improved professional resume in markdown format:",
        "Here is the improved professional resume:",
        "Improved Professional Resume:",
        "The improved resume:",
        "Here's the improved resume:",
        "Improved Resume:",
    ]
    
    for phrase in intro_phrases:
        if response.lower().startswith(phrase.lower()):
            response = response[len(phrase):].strip()
            break
    
    # Remove markdown code blocks if they wrap everything
    if response.startswith("```markdown"):
        response = response[11:].strip()
    elif response.startswith("```"):
        response = response[3:].strip()
        
    if response.endswith("```"):
        response = response[:-3].strip()
    
    return response

def remove_added_content(edited: str, original: str) -> str:
    """
    Basic safety check to remove any obvious new sections that weren't in original.
    This is a simple implementation - you might want to enhance it further.
    """
    original_sections = set(re.findall(r'^#+\s+.+', original, flags=re.MULTILINE))
    edited_lines = edited.split('\n')
    cleaned_lines = []
    
    current_section = None
    for line in edited_lines:
        # Check if this is a section header
        if re.match(r'^#+\s+.+', line):
            if line not in original_sections:
                current_section = "REMOVE"
            else:
                current_section = None
        
        if current_section != "REMOVE":
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def parse_markdown_to_plain_text(md_content: str) -> str:
    # 1. Split into lines and filter out Headers/Empty lines
    lines = [
        line.strip() 
        for line in md_content.split('\n') 
        if line.strip() and not line.startswith('#')
    ]
    
    # 2. Join back into a single block of text
    text = "\n".join(lines)
    
    # 3. Strip inline formatting using Regex
    # Removes Bold (**), Italics (* or _), and Inline Code (`)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # Bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # Italics
    text = re.sub(r"_(.*?)_", r"\1", text)         # Italics (underscore)
    text = re.sub(r"`(.*?)`", r"\1", text)         # Inline Code
    
    return text.strip()

def extract_text_from_latex(latex_content: str) -> str:
    """
    Extracts plain text from LaTeX content.
    """
    try:
        l2t = LatexNodes2Text()
        plain_text = l2t.latex_to_text(latex_content)
        return plain_text
    except Exception as e:
        return f"Error extracting text from LaTeX: {e}"


# Backwards-compatibility: allow callers to use `.invoke(...)` on these functions.
try:
    parse_markdown_to_plain_text.invoke = parse_markdown_to_plain_text
except Exception:
    pass

try:
    extract_text_from_latex.invoke = extract_text_from_latex
except Exception:
    pass

# def extract_text_from_latex(latex_text: str) -> str:
#     return LatexNodes2Text().latex_to_text(latex_text)
