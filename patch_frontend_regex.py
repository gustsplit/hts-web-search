import re
import os

path = r"c:\Users\gusts\Downloads\hts-web-search\frontend\index.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to match:
#         if (selectedIndex < 0) {
#           alert(t("duty.alertSelect"));
#           return;
#         }
# We use \s+ to match any whitespace including newlines
pattern = r"(\s*)if\s*\(selectedIndex\s*<\s*0\)\s*\{\s*alert\(t\(\"duty\.alertSelect\"\)\);\s*return;\s*\}"

# Replacement using the captured indentation group \1
replacement = r'\1if (selectedIndex < 0) {\n\1  dutyResultEl.innerHTML = `<div class="error-box" style="margin-top:10px;">${t("duty.alertSelect")}</div>`;\n\1  resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });\n\1  return;\n\1}'

new_content, count = re.subn(pattern, replacement, content)

if count > 0:
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully patched {count} occurrences.")
else:
    print("Pattern not found.")
    idx = content.find("selectedIndex < 0")
    if idx != -1:
        print("Found partial match:")
        print(repr(content[idx:idx+100]))
