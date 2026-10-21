---
description: Add a verified paper to references.md and references.bib
argument-hint: <doi | arXiv id>
---
Add `$ARGUMENTS` to the literature:

1. Fetch the BibTeX from the publisher (DOI) or DBLP/arXiv; never type or generate it. Use the key
   `<firstauthor><year><firstword>`.
2. Append it to `references.bib`, and an annotated entry to `references.md` in the section matching its role,
   following the entry template there (Status, Claim in our own words, Relevance, Used in, Notes).
3. If you could not verify the id or the BibTeX against the source, set `Status: to-verify`.
4. Run `just refs-check`.
