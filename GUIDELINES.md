# Project Guidelines

> **META**: All guidelines must be marked `(UNREVIEWED)` until user confirms review. [Added: 2025-12-13 14:43]
> **META**: All guidelines must include insertion timestamp `[Added: YYYY-MM-DD HH:MM]`. [Added: 2025-12-13 14:43]

---

## File Organization (UNREVIEWED) [Added: 2025-12-13 14:43]

### Multiple Files with Same Basename
When a project contains multiple related files sharing the same basename (e.g., `paper.tex`, `paper.pdf`, `paper.aux`, `paper.log`), these files MUST be organized into a subdirectory with the basename and the `.d` extension.

**Example:**
```
# Before (incorrect):
references/
  string-draft-main.tex
  string-draft-main.pdf
  string-draft-main.aux
  string-draft-main.log

# After (correct):
references/
  string-draft-main.d/
    string-draft-main.tex
    string-draft-main.pdf
    string-draft-main.aux
    string-draft-main.log
```

**Rationale:** This convention improves directory readability and makes it immediately clear which files are related, especially when dealing with LaTeX projects that generate multiple auxiliary files.

---

## Timestamp Format (UNREVIEWED) [Added: 2025-12-13 14:43]
All timestamps in guidelines and TODOs must follow the format: `[Added: YYYY-MM-DD HH:MM]` or `(Completed: YYYY-MM-DD HH:MM)`.
