import re, pathlib

SEP_CHAR = "\u2501"  # ━

for fpath in pathlib.Path("experiments").glob("*.py"):
    text = fpath.read_text(encoding="utf-8")
    # Replace the thick horizontal line in docstrings
    new_text = text.replace(SEP_CHAR * 64, "-" * 64)
    # Replace any remaining ━ in print statements
    new_text = new_text.replace(f'"{SEP_CHAR}"', '"-"')
    fpath.write_text(new_text, encoding="utf-8")
    print(f"Fixed: {fpath}")

print("Done.")
