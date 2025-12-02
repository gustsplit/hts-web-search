import os

path = r"c:\Users\gusts\Downloads\hts-web-search\frontend\index.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = """        if (selectedIndex < 0) {
          alert(t("duty.alertSelect"));
          return;
        }"""

replacement = """        if (selectedIndex < 0) {
          dutyResultEl.innerHTML = `<div class="error-box" style="margin-top:10px;">${t("duty.alertSelect")}</div>`;
          resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });
          return;
        }"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully patched.")
else:
    print("Target not found.")
    idx = content.find("if (selectedIndex < 0)")
    if idx != -1:
        print("Found partial match:")
        print(repr(content[idx:idx+100]))
    else:
        print("Partial match not found.")
