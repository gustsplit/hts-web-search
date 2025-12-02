import os

path = r"c:\Users\gusts\Downloads\hts-web-search\frontend\index.html"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
patched = False
for line in lines:
    if 'alert(t("duty.alertSelect"));' in line:
        # Preserve indentation
        indent = line[:line.find('alert')]
        new_lines.append(f'{indent}dutyResultEl.innerHTML = `<div class="error-box" style="margin-top:10px;">${{t("duty.alertSelect")}}</div>`;\n')
        new_lines.append(f'{indent}resultsEl.scrollIntoView({{ behavior: "smooth", block: "center" }});\n')
        patched = True
    else:
        new_lines.append(line)

if patched:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Successfully patched.")
else:
    print("Target not found.")
