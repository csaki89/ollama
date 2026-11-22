// Compares two result dirs of ct.sh: node compare.js <dirA> <dirB>
// chat: first generated tokens equal, content; emb: dimension, norm, cosine similarity.
const fs = require("fs"), path = require("path");
const [A, B] = process.argv.slice(2);
const rd = (d, f) => { try { return JSON.parse(fs.readFileSync(path.join(d, f), "utf8")); } catch (e) { return null; } };
let bad = 0;
const files = new Set([...fs.readdirSync(A), ...fs.readdirSync(B)].filter(f => f.endsWith(".json")));
for (const f of [...files].sort()) {
  const a = rd(A, f), b = rd(B, f);
  if (!a || !b) { console.log(`${f}\tMISSING in ${!a ? A : B}`); bad++; continue; }
  if (f.startsWith("chat-")) {
    const ta = a.tokens || [], tb = b.tokens || [];
    let same = 0; while (same < Math.min(ta.length, tb.length) && ta[same] === tb[same]) same++;
    const eq = same === ta.length && ta.length === tb.length && ta.length > 0;
    if (!eq) bad++;
    console.log(`${f}\ttokens ${ta.length}/${tb.length}\tcommon prefix ${same}\t${eq ? "IDENTICAL" : "DIFF"}\t${JSON.stringify((b.content || a.error || "").slice(0, 90))}`);
  } else if (f.startsWith("emb-")) {
    const ea = a.data && a.data[0] && a.data[0].embedding, eb = b.data && b.data[0] && b.data[0].embedding;
    if (!ea || !eb) { console.log(`${f}\tNO EMBEDDING\t${JSON.stringify(a.error || b.error || "").slice(0, 160)}`); bad++; continue; }
    const dot = ea.reduce((s, x, i) => s + x * eb[i], 0);
    const na = Math.hypot(...ea), nb = Math.hypot(...eb), cos = dot / (na * nb);
    let maxd = 0; ea.forEach((x, i) => maxd = Math.max(maxd, Math.abs(x - eb[i])));
    const ok = ea.length === eb.length && cos > 0.9999;
    if (!ok) bad++;
    console.log(`${f}\tdim ${ea.length}/${eb.length}\tnorm ${na.toFixed(4)}/${nb.toFixed(4)}\tcos ${cos.toFixed(7)}\tmax|d| ${maxd.toExponential(2)}\t${ok ? "OK" : "CHECK"}`);
  }
}
console.log(bad ? `${bad} difference(s)` : "all equal");
process.exitCode = bad ? 1 : 0;
