#!/usr/bin/env node
// Cross-platform syntax check: runs `node --check` on index.js, preload.js,
// and every .js file under js/. Exits non-zero on the first failure.
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const appDir = path.resolve(__dirname, "..");

function collectJsFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectJsFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".js")) {
      files.push(fullPath);
    }
  }
  return files;
}

const targets = [
  path.join(appDir, "index.js"),
  path.join(appDir, "preload.js"),
  ...collectJsFiles(path.join(appDir, "js")),
];

for (const file of targets) {
  const result = spawnSync(process.execPath, ["--check", file], {
    stdio: "inherit",
  });
  if (result.status !== 0) {
    console.error(`Syntax check failed: ${path.relative(appDir, file)}`);
    process.exit(result.status || 1);
  }
}

console.log(`Syntax check passed for ${targets.length} file(s).`);
