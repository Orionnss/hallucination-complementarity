"""Methods adapted from their authors' code (see ADDING_A_METHOD.md).

Each subpackage wraps an upstream repository that lives, unmodified, under
`original-repos/<name>` at a pinned commit. The upstream code is called, never copied:
`upstream.load_upstream` checks the commit and the hash of every file it imports, so an
edited or moved checkout is refused rather than silently used.
"""
