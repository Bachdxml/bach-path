#!/usr/bin/env node
/*
 * Unit test for the viewer's mask-degradation notice mapping.
 *
 * apps/desktop has no JS test runner, so this is a dependency-free Node script:
 *   node scripts/test_mask_notice.js
 * It extracts the pure `maskDegradationNoticeText` function from viewer.js (which
 * cannot be `require`d directly because its top level touches the DOM) and
 * verifies the status -> notice mapping required by the spec (Definition of
 * Done #7 / #10): a notice for `downsampled` and `dropped`, none for `full` or
 * legacy runs lacking the field.
 */

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const viewerSource = fs.readFileSync(
  path.join(__dirname, "..", "js", "viewer.js"),
  "utf8"
);

// Pull the pure function out of the browser file by brace-matching.
const marker = "function maskDegradationNoticeText";
const start = viewerSource.indexOf(marker);
assert.ok(start >= 0, "maskDegradationNoticeText not found in viewer.js");
let depth = 0;
let end = -1;
for (let i = viewerSource.indexOf("{", start); i < viewerSource.length; i += 1) {
  const ch = viewerSource[i];
  if (ch === "{") depth += 1;
  else if (ch === "}") {
    depth -= 1;
    if (depth === 0) {
      end = i + 1;
      break;
    }
  }
}
assert.ok(end > start, "could not brace-match maskDegradationNoticeText");

// eslint-disable-next-line no-new-func
const maskDegradationNoticeText = new Function(
  `${viewerSource.slice(start, end)}; return maskDegradationNoticeText;`
)();

// full status -> no notice
assert.strictEqual(maskDegradationNoticeText("full", null), "");
// legacy run (undefined status) -> no notice
assert.strictEqual(maskDegradationNoticeText(undefined, undefined), "");
// downsampled -> notice, includes factor when present
const downsampled = maskDegradationNoticeText("downsampled", 4);
assert.ok(downsampled.length > 0, "downsampled should produce a notice");
assert.ok(/downsampled/i.test(downsampled), "downsampled notice should mention downsampling");
assert.ok(downsampled.includes("4"), "downsampled notice should include the factor");
// downsampled without a usable factor -> still a notice, no factor detail
const downsampledNoFactor = maskDegradationNoticeText("downsampled", null);
assert.ok(downsampledNoFactor.length > 0, "downsampled (no factor) should still notice");
assert.ok(!/×/.test(downsampledNoFactor), "no factor detail when factor missing");
// dropped -> notice
const dropped = maskDegradationNoticeText("dropped", null);
assert.ok(dropped.length > 0, "dropped should produce a notice");
assert.ok(/boxes only/i.test(dropped), "dropped notice should mention boxes only");

console.log("mask notice logic: all assertions passed");
