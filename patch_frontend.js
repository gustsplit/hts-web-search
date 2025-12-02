const fs = require('fs');
const path = 'c:\\Users\\gusts\\Downloads\\hts-web-search\\frontend\\index.html';

try {
  let content = fs.readFileSync(path, 'utf8');
  const target = 'alert(t("duty.alertSelect"));';
  
  if (content.includes(target)) {
    const lines = content.split('\n');
    const newLines = lines.map(line => {
      if (line.includes(target)) {
        const indent = line.substring(0, line.indexOf('alert'));
        return `${indent}dutyResultEl.innerHTML = \`<div class="error-box" style="margin-top:10px;">\${t("duty.alertSelect")}</div>\`;\n${indent}resultsEl.scrollIntoView({ behavior: "smooth", block: "center" });`;
      }
      return line;
    });
    
    fs.writeFileSync(path, newLines.join('\n'), 'utf8');
    console.log('Successfully patched.');
  } else {
    console.log('Target not found.');
  }
} catch (err) {
  console.error(err);
}
