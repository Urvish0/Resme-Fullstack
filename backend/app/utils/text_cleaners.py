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

# def clean_resume_response(response: str) -> str:
#     """
#     Simple cleaning function that removes common intro phrases but is conservative.
#     """
#     response = response.strip()

#     # Remove common intro lines (only if they're at the very beginning)
#     intro_phrases = [
#         "Here's the improved professional resume in markdown format:",
#         "Here is the improved professional resume:",
#         "Improved Professional Resume:",
#         "The improved resume:",
#         "Here's the improved resume:",
#         "Improved Resume:",
#     ]

#     for phrase in intro_phrases:
#         if response.lower().startswith(phrase.lower()):
#             response = response[len(phrase):].strip()
#             break

#     # Remove markdown code blocks if they wrap everything
#     if response.startswith("```markdown"):
#         response = response[11:].strip()
#     elif response.startswith("```"):
#         response = response[3:].strip()

#     if response.endswith("```"):
#         response = response[:-3].strip()

#     return response


def clean_resume_response(response: str) -> str:
    """
    Clean LLM response to extract markdown resume content.
    Handles various response formats and ensures consistent markdown output.
    """
    if not response or not response.strip():
        return ""

    # Normalize line endings first
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    response = response.strip()

    # Highly aggressive intro stripping
    # If we find a line starting with #, that's where the resume likely starts
    match = re.search(r'^#\s+.+', response, flags=re.MULTILINE)
    if match:
        response = response[match.start():].strip()
    else:
        # Fallback to common intro phrases
        intro_phrases = [
            "Here's the improved", "Here is the improved", "Improved Professional Resume",
            "The improved resume", "Here's the improved resume", "Improved Resume",
            "Here is your optimized", "Here's your optimized", "Optimized Resume",
            "Sure! Here is", "I have optimized"
        ]
        for phrase in intro_phrases:
            if response.lower().startswith(phrase.lower()):
                response = response[len(phrase):].strip()
                if response.startswith(":") or response.startswith("."):
                    response = response[1:].strip()
                break

    # Handle markdown code block wrapping even if not at the true start
    if "```markdown" in response:
        response = response.split("```markdown", 1)[1].split("```", 1)[0].strip()
    elif "```" in response:
        # Check if the text inside the first code block is large (likely the resume)
        potential_block = response.split("```", 1)[1].split("```", 1)[0].strip()
        if len(potential_block) > len(response) * 0.5:
            response = potential_block

    # --- Whitespace normalization ---
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in response.split("\n")]
    response = "\n".join(lines)

    # Collapse 3+ consecutive blank lines into exactly 2 (one blank line between sections)
    response = re.sub(r"\n{3,}", "\n\n", response)

    return response.strip()


def remove_added_content(edited: str, original: str) -> str:
    """
    Basic safety check to remove any obvious new sections that weren't in original.
    This is a simple implementation - you might want to enhance it further.
    """
    original_sections = set(re.findall(r"^#+\s+.+", original, flags=re.MULTILINE))
    edited_lines = edited.split("\n")
    cleaned_lines = []

    current_section = None
    for line in edited_lines:
        # Check if this is a section header
        if re.match(r"^#+\s+.+", line):
            if line not in original_sections:
                current_section = "REMOVE"
            else:
                current_section = None

        if current_section != "REMOVE":
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# def parse_markdown_to_plain_text(md_content: str) -> str:
#     # 1. Split into lines and filter out Headers/Empty lines
#     lines = [
#         line.strip()
#         for line in md_content.split('\n')
#         if line.strip() and not line.startswith('#')
#     ]

#     # 2. Join back into a single block of text
#     text = "\n".join(lines)

#     # 3. Strip inline formatting using Regex
#     # Removes Bold (**), Italics (* or _), and Inline Code (`)
#     text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # Bold
#     text = re.sub(r"\*(.*?)\*", r"\1", text)       # Italics
#     text = re.sub(r"_(.*?)_", r"\1", text)         # Italics (underscore)
#     text = re.sub(r"`(.*?)`", r"\1", text)         # Inline Code

#     return text.strip()


def parse_markdown_to_plain_text(md_content: str) -> str:
    """
    Parse markdown OR plain text to clean plain text.
    More robust version that handles both markdown and plain text.
    """
    if not md_content:
        return ""

    # 1. Remove HTML tags if present
    md_content = re.sub(r"<[^>]+>", " ", md_content)

    # 2. Replace horizontal whitespace characters with single space
    md_content = re.sub(r"[ \t]+", " ", md_content)

    # 3. Remove markdown formatting if present
    # Headers
    md_content = re.sub(r"^#+\s+", "", md_content, flags=re.MULTILINE)
    # Bold
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    # Italics
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    md_content = re.sub(r"_(.*?)_", r"\1", md_content)
    # Inline code
    md_content = re.sub(r"`(.*?)`", r"\1", md_content)
    # Links
    md_content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md_content)
    # Images
    md_content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", md_content)
    # Lists
    md_content = re.sub(r"^\s*[-*+]\s+", "", md_content, flags=re.MULTILINE)
    # Code blocks
    md_content = re.sub(r"```.*?```", "", md_content, flags=re.DOTALL)

    # 4. Clean up extra whitespace at ends
    md_content = md_content.strip()

    return md_content.strip()


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
