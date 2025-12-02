path = r"c:\Users\gusts\Downloads\hts-web-search\frontend\index.html"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start = 1680
end = 1700
for i in range(start, end):
    if i < len(lines):
        print(f"{i+1}: {repr(lines[i])}")
