# Project Guidelines

> **CRITICAL**: At each prompt turn, evaluate whether the user's instruction should become a guideline. If yes, add immediately. Review this rule every turn. (UNREVIEWED) [Added: 2025-12-12 18:22]

> **CRITICAL**: NEVER perform `git rebase` or `git reset` without explicit permission. (UNREVIEWED) [Added: 2025-12-12 18:16]

> **CRITICAL**: NEVER abbreviate user prompts in `PROMPTS-LOG.md`. Record verbatim. If multiple prompts, use `[followed by:]` notation--never `...`. (UNREVIEWED) [Added: 2025-12-12 18:16]

> **CRITICAL**: Commit at every turn. After each response that modifies files, run `git add -A; git commit -m "..."`. (UNREVIEWED) [Added: 2025-12-12 18:25]

> **CRITICAL**: Push to remote periodically. Run `git push` after logical batches of commits. (UNREVIEWED) [Added: 2025-12-12 18:39]

> **CRITICAL**: Periodically review full GUIDELINES.md every few turns to ensure compliance. (UNREVIEWED) [Added: 2025-12-12 18:26]

> **CRITICAL**: Update PROMPTS-LOG.md at every turn with new user prompts. Never let prompts accumulate unlogged. (UNREVIEWED) [Added: 2025-12-12 18:36]

> **CRITICAL**: Maintain TODO.md for incomplete work. When user sends prompts before prior work is done, add items to TODO. Check recent prompts and PROMPTS-LOG for undone items. Move completed TODOs to TODO-COMPLETED.md. (UNREVIEWED) [Added: 2025-12-12 18:36]

> **CRITICAL**: Do not interrupt the user's view with walkthrough artifacts. When creating a walkthrough, do NOT use `notify_user` to request review of it. Maintain it silently. (UNREVIEWED) [Added: 2025-12-12 19:20]

> **CRITICAL**: Timestamp each individual prompt entry in `PROMPTS-LOG.md` at insertion time `[YYYY-MM-DD HH:MM]`. (UNREVIEWED) [Added: 2025-12-12 18:54]

> **META**: All new guidelines must be marked `(UNREVIEWED)` until user confirms review. [Added: 2025-12-12 18:16]
> **META**: All new guidelines must include insertion timestamp `[Added: YYYY-MM-DD HH:MM]`. [Added: 2025-12-12 18:16]
> **META**: When syncing external guidelines, MERGE with existing project-specific content--do not overwrite. [Added: 2025-12-12 18:16]

---

## Prime Directive for AI Responses (UNREVIEWED) [Added: 2025-12-11 23:56]

> This directive supersedes earlier fragments and applies to all AI assistant interactions with Daniel.

### PD.1 Persona and Difficulty Targeting (UNREVIEWED) [Added: 2025-12-11 23:56]
- **Expert Persona**: Always respond from the persona of a senior/staff-level expert matched to the topic.
- **Assume Senior Level**: Daniel operates at a strong senior level.
- **Difficulty Mix**: ~70% ZPD, ~20% staff+/research-lead, ~10% aspirational.

### PD.2 Answer Style and Structure (UNREVIEWED) [Added: 2025-12-11 23:56]
- **Technically Precise**: Structured and concise, but not stiff.
- **Concrete Mechanisms**: Prefer equations, implementation details over vague intuitions.
- **Surface Operational Concerns**: When something matters operationally, surface it explicitly.

### PD.3 Calibration Footer (REQUIRED) (UNREVIEWED) [Added: 2025-12-11 23:56]
At the end of each substantive answer, include: Persona Used, Difficulty Tuning, Concrete Suggestion.

### PD.4 Log Directive Snapshots (UNREVIEWED) [Added: 2025-12-11 23:56]
- When Daniel labels a message as "log directive," treat it as a snapshot anchor with temporal context.

### PD.5 Historical Reference (UNREVIEWED) [Added: 2025-12-11 23:56]
- Reference Snapshot: 2025-12-08 prime-directive conversation.

---

## P1 - Critical (Every Session)

### 1. Version Control Safety (UNREVIEWED) [Added: 2025-12-12 18:37]
#### 1.1 Strict Prohibitions
- **NO REBASE/RESET**: NEVER perform `git rebase` or `git reset` without explicit, written user permission.
- **Data Loss Prevention**: These commands rewrite history and can cause irreversible data loss.

#### 1.2 Branch Hygiene
- **NO SWITCH IF DETACHED**: If in detached HEAD state, do NOT switch branches without explicit user permission.
- **Verify Before Switch**: Run `git status` before switching branches.

#### 1.3 Commit Hygiene
- **Verify Staging**: Run `git status` BEFORE every commit.
- **Atomic Commits**: Commit logically grouped changes with descriptive messages.
- **No Dangling Work**: Never leave uncommitted changes at end of session.

### 2. Citation Integrity (UNREVIEWED) [Added: 2025-12-12 17:31]
- **Cite all references**: Every reference in References section must be cited inline.
- **Consistent format**: Use IEEE `[1]` format for CS. Do not mix with Author-Year.
- **Verify citations**: Inline `\cite{key}` must have matching bib entries.

### 3. LaTeX Compilation (UNREVIEWED) [Added: 2025-12-12 17:31]
- **Compile at every turn**: After editing `.tex` files, recompile PDF immediately.
- **Clean rebuild**: If issues persist, delete `.aux`, `.log`, `.pdf` and recompile twice.

---

## P2 - Important (Regular Maintenance)

### 4. Documentation (UNREVIEWED) [Added: 2025-12-12 18:16]

#### 4.1 Session Tracking
- **Prompt Logging**: Append prompts verbatim to `PROMPTS-LOG.md`.
- **Timestamping**: New sessions use header `## YYYY-MM-DDTHH:MM - Session Description`.
- **Full Fidelity**: Never truncate or use ellipses. Log exact full text.
- **Append ONLY**: Do not revise existing logs.

#### 4.2 Session Wrap-Up
- **Log Prompts**: Ensure all prompts are appended with timestamp.
- **Update TODO**: Verify completed work is checked off and moved to `TODO-COMPLETED.md`.
- **Final Commit & Push**: Stage, commit, and push to remote.

### 5. TODO Management (UNREVIEWED) [Added: 2025-12-12 18:36]

#### 5.1 Insertion Policy
- **Prioritize Immediately**: Assess priority (P1, P2, P3) and insert accordingly.
- **Timestamp Insertion**: Append `[Added: YYYY-MM-DD HH:MM]` to every new item.
- **Mark Unreviewed**: Mark all new items as `(UNREVIEWED)` until confirmed by user.

#### 5.2 Completion Policy
- **Move to Archive**: When done, move from `TODO.md` to `TODO-COMPLETED.md`.
- **Timestamping**: Append completion timestamp `(Completed: YYYY-MM-DDTHH:MM)`.

### 6. Bibliography Management (UNREVIEWED) [Added: 2025-12-12 18:16]
#### 6.1 Formatting Standard
```bibtex
@article{key,
  author = {Name},
  year = {2025},
  title = {Title}
}
```
- Blank line after `@type{key,`
- Blank line between each field
- Two-space indentation for fields

#### 6.2 Organization
Use commented section headers:
- Primary Sources (Government, Court)
- Secondary Sources (Academic)
- News Sources
- Organizational Sources
- Data Sources

---

## P3 - Housekeeping

### 7. File Organization (UNREVIEWED) [Added: 2025-12-13 14:43]

#### 7.1 Multiple Files with Same Basename
When a project contains multiple related files sharing the same basename (e.g., `paper.tex`, `paper.pdf`, `paper.aux`, `paper.log`), these files MUST be organized into a subdirectory with the basename and the `.d` extension.

**Example:**
```
references/
  string-draft-main.d/
    string-draft-main.tex
    string-draft-main.pdf
    string-draft-main.aux
    string-draft-main.log
```

**Rationale:** This convention improves directory readability and makes it immediately clear which files are related, especially when dealing with LaTeX projects that generate multiple auxiliary files.

#### 7.2 Suffix Meanings
| Suffix | Meaning |
|--------|---------|
| `-SUBMITTED` | Finalized/submitted work |
| `-TO_INTEGRATE` | Content pending integration |
| `-COMPREHENSIVE` | Complete/long-form documents |
| `-OLD` | Superseded/deprecated content |

### 8. Safe Deletion (UNREVIEWED) [Added: 2025-12-12 18:37]
- Delete files only after verifying content is captured elsewhere.
- Track deferred work in `TODO.md`.

---

## Domain-Specific Guidelines (UNREVIEWED) [Added: 2025-12-15 19:30]

### Scientific Framing & Naming
*   **'ESPR' is Deprecated**: The term 'ESPR' is to be removed from all active vocabulary and documentation, except where it exists as a historical reference in the references folder.
*   **STRING-Centric Framing**: All comparison models previously called 'ESPR' must now be referred to as 'Relaxed STRING' or 'STRING (Relaxed)'.
*   **Purpose of Relaxation**: We only discuss these relaxed models to demonstrate two specific properties of STRING:
    1.  **Fragility**: Violating STRING's constraints leads to collapsed robustness.
    2.  **Zero-Gap**: STRING's exact constraints yield zero invariance error.
*   **Communication Rule**: Never frame the project as 'STRING vs. ESPR'. Frame it as 'Analyzing the Necessity of STRING's Constraints'.

### Domain-Specific Sub-Project Guidelines
- **PhD Applications**: [GUIDELINES-PHD_Applications.md](GUIDELINES-PHD_Applications.md)
- **Thesis**: [GUIDELINES-Thesis.md](GUIDELINES-Thesis.md)


### LaTeX Compilation (UNREVIEWED) [Added: 2025-12-15 22:42]
When compiling LaTeX documents, always run pdflatex (or equivalent) twice to resolve cross-references and eliminate '??' placeholders.
