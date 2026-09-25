// Generator for docs/data-flow-deck.pptx -- the sentri-backend data-flow deck, written for a
// non-technical audience (procurement / management / policy). Technical detail and sources
// live in the speaker notes, not on the slides.
//
// Regenerate:  NODE_PATH=<dir with pptxgenjs, react, react-dom, react-icons, sharp>/node_modules \
//              node docs/data-flow-deck.js
// Re-run `make eval` / `make eval-agentic` and update RECALL_SINGLE / AGENT_PASS if they change.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// Results of the repo's own commands, run 2026-09-24.
const RECALL_SINGLE = "20 of 20"; // make eval (rerank): "GATE (single recall) hit_rate=1.000" over 20 single questions
const AGENT_PASS = "7 of 9";      // make eval-agentic: "questions=9 | pass_rate=0.778"

const C = {
  ink: "1B2631", body: "2E3A46", muted: "5B6770", line: "8A96A3",
  tint: "EEF2F5", white: "FFFFFF", soft: "DDE7EE",
  teal: "2B7A78", blue: "3D5A80", orange: "D9731A", green: "3B8254",
  darkCard: "263545", darkLine: "35485C", paleText: "C9D3DC",
};
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Asking questions of the bid documents";

// ---------- helpers ----------
async function iconPng(name, color = "#FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function iconCircle(slide, name, x, y, d, bg) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg, width: 0 } });
  const pad = d * 0.26;
  slide.addImage({ data: await iconPng(name), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}
function title(slide, text) {
  slide.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.75, fontFace: HEAD, fontSize: 28, bold: true,
    color: C.ink, margin: 0, valign: "middle", isTextBox: true });
}
function text(slide, t, x, y, w, h, o = {}) {
  slide.addText(t, { x, y, w, h, fontFace: o.face || BODY, fontSize: o.fs || 16, color: o.color || C.body,
    bold: !!o.bold, italic: !!o.italic, align: o.align || "left", valign: o.valign || "top",
    margin: 0, paraSpaceAfter: o.psa || 0, isTextBox: true });
}
function bullets(slide, items, x, y, w, h, o = {}) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontFace: BODY, fontSize: o.fs || 17, color: o.color || C.body, valign: "top",
      paraSpaceAfter: o.psa || 12, margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, fill = C.tint, line = C.soft) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1,
    fill: { color: fill }, line: { color: line, width: 1 } });
}
function arrow(slide, x1, y1, x2, y2, o = {}) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || C.line, width: o.width || 2.25, endArrowType: "triangle" } });
}

async function build() {
  // ---------- 1. Title ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.ink };
    text(s, "Asking questions of the bid documents", 0.6, 0.55, 8.8, 1.4,
      { face: HEAD, fs: 36, bold: true, color: C.white, valign: "middle" });
    text(s, "How our assistant finds facts in six DPWH construction contracts, and why it says " +
      "“I don’t know” instead of guessing.", 0.6, 2.15, 8.2, 0.9, { fs: 18, color: C.paleText });
    const steps = [["FaFileAlt", "Scanned bids"], ["FaBook", "Catalogued"], ["FaQuestionCircle", "You ask"], ["FaCheckCircle", "Checked answer"]];
    for (let i = 0; i < steps.length; i++) {
      const x = 0.6 + i * 2.0;
      await iconCircle(s, steps[i][0], x, 3.45, 0.7, i === 3 ? C.orange : C.teal);
      text(s, steps[i][1], x - 0.3, 4.25, 1.3, 0.35, { fs: 13, color: C.paleText, align: "center" });
      if (i < 3) arrow(s, x + 0.82, 3.8, x + 1.88, 3.8, { color: "6B7C8C", width: 1.75 });
    }
    s.addNotes("A walkthrough of the sentri-backend pipeline for a non-technical audience. Technical terms: " +
      "RAG (retrieval-augmented generation) over DPWH bid documents, built on LlamaIndex, running locally with Ollama. " +
      "Source: CLAUDE.md.");
  }

  // ---------- 2. The problem ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "The problem: facts are scattered");
    const stats = [["31", "bid files (PDF)"], ["283", "scanned pages"], ["6", "contracts"]];
    stats.forEach(([v, l], i) => {
      const x = 0.5 + i * 3.05;
      card(s, x, 1.35, 2.85, 1.45);
      text(s, v, x, 1.45, 2.85, 0.85, { face: HEAD, fs: 44, bold: true, color: C.teal, align: "center", valign: "middle" });
      text(s, l, x, 2.3, 2.85, 0.4, { fs: 16, color: C.muted, align: "center" });
    });
    bullets(s, [
      "The winner, the amount and the signatories sit on different pages and in different documents.",
      "The contracts use the same templates, so pages from different contracts look alike.",
      "Finding one fact by hand means leafing through a stack of scans.",
    ], 0.6, 3.2, 8.8, 2.0);
    s.addNotes("31 PDFs in source/ with 283 pages in total (pdfinfo). 6 contracts: 24A00153, 24AJ0052, 24BG0272, " +
      "24BJ0005, 24CC0265, 24CM0001 (index_store/manifest.json; make check). rag/enrich.py notes that about 90% of " +
      "OCR chunks do not name their own contract and that the 6 contracts share the same document templates.");
  }

  // ---------- 3. The big picture ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "The big picture");
    const steps = [
      ["FaFileAlt", "Scanned documents", "typed up once", C.blue],
      ["FaBook", "Organised", "catalogued and tagged", C.blue],
      ["FaQuestionCircle", "Your question", "in plain English", C.teal],
      ["FaCheckCircle", "Checked answer", "with sources, or “I don’t know”", C.orange],
    ];
    const W = 2.1, G = 0.2;
    // phase bands
    card(s, 0.5, 1.5, 2 * W + G, 0.45, C.soft, C.soft);
    text(s, "Prepared once, in advance", 0.5, 1.5, 2 * W + G, 0.45, { fs: 14, bold: true, color: C.blue, align: "center", valign: "middle" });
    card(s, 0.5 + 2 * (W + G), 1.5, 2 * W + G, 0.45, "FBE9DA", "FBE9DA");
    text(s, "Every time you ask", 0.5 + 2 * (W + G), 1.5, 2 * W + G, 0.45, { fs: 14, bold: true, color: C.orange, align: "center", valign: "middle" });
    for (let i = 0; i < steps.length; i++) {
      const [ic, h, sub, col] = steps[i];
      const x = 0.5 + i * (W + G);
      await iconCircle(s, ic, x + W / 2 - 0.55, 2.3, 1.1, col);
      text(s, h, x, 3.6, W, 0.45, { fs: 18, bold: true, color: C.ink, align: "center" });
      text(s, sub, x, 4.05, W, 0.7, { fs: 14, color: C.muted, align: "center" });
      if (i < 3) arrow(s, x + W / 2 + 0.65, 2.85, x + W + G + W / 2 - 0.65, 2.85);
    }
    s.addNotes("Offline (make build): OCR JSON in database/ -> stages 1-3 (loader, enrich, relationships) -> vector index " +
      "and corpus manifest in index_store/. Online (make ask / make agent / the SSE API): route -> retrieve + rerank -> " +
      "generate -> Verifier. Sources: CLAUDE.md, scripts/build.sh.");
  }

  // ---------- 4. Preparing the documents ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "Step 1: the scans are typed up once");
    // visual: page -> tagged pieces
    await iconCircle(s, "FaFileAlt", 0.6, 2.35, 1.1, C.blue);
    text(s, "Scanned page", 0.35, 3.55, 1.6, 0.35, { fs: 13, color: C.muted, align: "center" });
    arrow(s, 1.85, 2.9, 2.45, 2.9);
    const pieces = ["Paragraph", "Table", "Signature block"];
    pieces.forEach((p, i) => {
      const y = 1.7 + i * 0.85;
      card(s, 2.55, y, 2.3, 0.7, C.tint, C.blue);
      text(s, p, 2.7, y + 0.06, 2.1, 0.32, { fs: 14, bold: true, color: C.ink });
      text(s, "contract · document · page", 2.7, y + 0.38, 2.1, 0.26, { fs: 11, color: C.teal, italic: true });
    });
    bullets(s, [
      "The scanned pages were already turned into typed text by OCR (software that reads text from an image).",
      "That text is split into 401 pieces, and each is tagged with its contract, document and page.",
      "The typed copy is the official record. The scans are never re-read.",
    ], 5.2, 1.6, 4.3, 3.6, { fs: 17 });
    s.addNotes("database/*.json (38 files) is the authoritative OCR transcription; OCR was done before this repo (no OCR code here). " +
      "Chunks are UUID-keyed under text/table/image/signature with a content field: 289 text, 9 table, 21 image, 82 signature = 401 " +
      "(make check). rag/loader.py makes one node per chunk; rag/enrich.py derives contract_id and doc_type from the file name; " +
      "rag/relationships.py links pieces in reading order (PREV/NEXT). .claude/verify.sh checks the JSON structure every turn and never re-OCRs.");
  }

  // ---------- 5. Catalogue + fact sheet ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "Step 2: build a card catalogue and a fact sheet");
    const panels = [
      ["FaBook", "Card catalogue", C.teal, [
        "Every piece gets a card that captures its meaning.",
        "A search for “who won” also finds “awarded to”.",
        "Cards are grouped in one drawer per contract."]],
      ["FaClipboardList", "Contract fact sheet", C.blue, [
        "One line per contract: name, place, office, contractor, amount.",
        "Filled in by the AI writer (software that reads and writes text).",
        "Anything it can’t find is marked “not stated”."]],
    ];
    for (let i = 0; i < panels.length; i++) {
      const [ic, h, col, items] = panels[i];
      const x = 0.5 + i * 4.6;
      card(s, x, 1.3, 4.4, 3.4);
      await iconCircle(s, ic, x + 0.3, 1.5, 0.8, col);
      text(s, h, x + 1.3, 1.5, 3.0, 0.8, { fs: 20, bold: true, color: C.ink, valign: "middle" });
      bullets(s, items, x + 0.35, 2.55, 3.8, 2.1, { fs: 16, psa: 10 });
    }
    s.addNotes("Catalogue = vector index: each piece is embedded with BAAI/bge-small-en-v1.5 on CPU (rag/index.py) and persisted to " +
      "index_store/. The embedding sees only the text; contract identity is metadata used as a filter (the 'drawer'). 6 synthetic " +
      "signatory-summary nodes are added, one per contract. Fact sheet = corpus manifest (rag/manifest.py): one LLM extraction per " +
      "contract with granite4.1:3b via Ollama, 6 rows in index_store/manifest.json. Fields: contract_name, location, " +
      "implementing_office, contractor, amount (+ doc_types). Runs last in make build and needs Ollama.");
  }

  // ---------- 6. Answering a question ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "How a question gets answered");
    const steps = [
      ["FaMapMarkerAlt", "Which contract?", "Spots the contract number or a place name, like Olongapo City", C.blue],
      ["FaSearch", "Pull the cards", "Takes a few dozen likely pieces from that contract’s drawer", C.teal],
      ["FaUserCheck", "Second reader", "Re-reads them and shortlists the best handful", C.teal],
      ["FaPenFancy", "Write the answer", "Answers from those pieces only, naming the documents", C.teal],
    ];
    const W = 2.1, G = 0.2;
    for (let i = 0; i < steps.length; i++) {
      const [ic, h, sub, col] = steps[i];
      const x = 0.5 + i * (W + G);
      card(s, x, 1.3, W, 2.95);
      await iconCircle(s, ic, x + W / 2 - 0.4, 1.5, 0.8, col);
      text(s, h, x + 0.1, 2.42, W - 0.2, 0.4, { fs: 17, bold: true, color: C.ink, align: "center" });
      text(s, sub, x + 0.15, 2.85, W - 0.3, 1.3, { fs: 13.5, color: C.body, align: "center" });
      if (i < 3) arrow(s, x + W + 0.02, 2.78, x + W + G - 0.02, 2.78, { width: 2 });
    }
    text(s, "Searching one contract’s drawer keeps look-alike pages from other contracts out of the answer.",
      0.5, 4.5, 9, 0.5, { fs: 15, italic: true, color: C.teal, align: "center" });
    s.addNotes("rag/generate.py RagAnswerer: route() resolves contract_id from an explicit id or a unique location token in the " +
      "manifest and strips the id from the search text. Retrieval = vector search, 40 candidates (RERANK_CANDIDATES), with a " +
      "contract_id metadata filter. Second reader = CPU cross-encoder BAAI/bge-reranker-base, 40 -> top 10 (rag/rerank.py). " +
      "Context = contract id + manifest + hits, trimmed so it fits one num_ctx 4096 call, with reading-order neighbours for the top 3. " +
      "Writer = granite4.1:3b, temperature 0.0, prompt says use ONLY the context and name the source documents.");
  }

  // ---------- 7. Harder questions ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "Harder questions are broken into simple ones");
    card(s, 0.5, 2.25, 2.1, 1.0, C.teal, C.teal);
    text(s, "“Who won contracts A, B and C?”", 0.6, 2.25, 1.9, 1.0, { fs: 14, bold: true, color: C.white, align: "center", valign: "middle" });
    ["Who won A?", "Who won B?", "Who won C?"].forEach((q, i) => {
      const y = 1.4 + i * 0.95;
      card(s, 3.05, y, 1.6, 0.7, C.tint, C.teal);
      text(s, q, 3.05, y, 1.6, 0.7, { fs: 14, color: C.ink, align: "center", valign: "middle" });
      arrow(s, 2.6, 2.75, 3.05, y + 0.35, { width: 1.5 });
      arrow(s, 4.65, y + 0.35, 5.1, 2.75, { width: 1.5 });
    });
    card(s, 5.1, 2.25, 1.2, 1.0, C.blue, C.blue);
    text(s, "One list", 5.1, 2.25, 1.2, 1.0, { fs: 15, bold: true, color: C.white, align: "center", valign: "middle" });
    bullets(s, [
      "Each part is answered and fact-checked on its own.",
      "“List all the projects” comes straight from the fact sheet.",
      "It never adds up amounts. Totals are left to people or a stronger system.",
    ], 6.7, 1.4, 2.85, 3.6, { fs: 16, psa: 12 });
    s.addNotes("rag/agent.py AgenticRag. A rule-based router picks one of five routes: simple, fanout (a list of contract ids, or " +
      "'each project', split by rule into one sub-question per contract), semantic (multi-part question split by the LLM, the only " +
      "place the LLM drives control, then combined), enumerate (list-all answered from the manifest, no LLM), analytical " +
      "(corpus-wide pattern, one pass over the whole manifest). Each sub-answer passes the Verifier. The combine prompt forbids " +
      "sums; aggregation is deferred to the handoff model.");
  }

  // ---------- 8. Fact-checker ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "It checks its work instead of guessing");
    await iconCircle(s, "FaPenFancy", 0.7, 1.6, 0.9, C.teal);
    text(s, "Draft answer", 0.4, 2.6, 1.5, 0.4, { fs: 14, bold: true, color: C.ink, align: "center" });
    arrow(s, 1.75, 2.05, 2.55, 2.05);
    await iconCircle(s, "FaUserCheck", 2.65, 1.45, 1.2, C.orange);
    text(s, "Fact-checker", 2.35, 2.72, 1.8, 0.35, { fs: 15, bold: true, color: C.ink, align: "center" });
    text(s, "compares it with the fact sheet", 2.2, 3.05, 2.1, 0.55, { fs: 12, color: C.muted, align: "center" });
    // outcomes
    card(s, 5.0, 1.3, 4.5, 0.95, "E6F2EA", C.green);
    text(s, "Passes → the answer is shown, with its sources", 5.2, 1.3, 4.2, 0.95, { fs: 15, bold: true, color: C.green, valign: "middle" });
    card(s, 5.0, 2.5, 4.5, 1.2, "FBE9DA", C.orange);
    text(s, "Fails → it looks again more widely. Still fails → it gives no answer (in effect “I don’t know”) and says why.",
      5.2, 2.5, 4.2, 1.2, { fs: 15, bold: true, color: C.orange, valign: "middle" });
    arrow(s, 3.95, 1.85, 5.0, 1.77, { width: 1.75 });
    arrow(s, 3.95, 2.3, 5.0, 3.1, { width: 1.75 });
    text(s, "What it catches", 0.5, 3.95, 9, 0.35, { fs: 15, bold: true, color: C.ink });
    const catches = [["FaTimesCircle", "A contract that doesn’t exist"], ["FaMapMarkerAlt", "A contract put in the wrong place or given the wrong contractor"]];
    for (let i = 0; i < catches.length; i++) {
      const x = 0.5 + i * 4.6;
      await iconCircle(s, catches[i][0], x, 4.4, 0.5, C.orange);
      text(s, catches[i][1], x + 0.65, 4.35, 3.8, 0.6, { fs: 14, color: C.body, valign: "middle" });
    }
    s.addNotes("rag/verify.py Verifier checks relational claims against the manifest. BLOCK tier: the answer names a contract_id " +
      "not in the corpus (zero false positives). FLAG tier: a contract bound to another contract's location or contractor; the " +
      "manifest is itself LLM-extracted, so this can occasionally hold back a correct answer, which the project accepts. " +
      "Open-ended claims (materials, clauses, counts) are not checked here. Retry ladder (rag/agent.py): filtered top 10 -> " +
      "widened top 20 -> drop the contract filter (only when no known contract was identified) -> withhold. The SSE service " +
      "only streams an answer after it passes.");
  }

  // ---------- 9. How we know it works ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.white };
    title(s, "How we know it works");
    const cards = [
      ["FaSearch", C.teal, "Finding the right page", RECALL_SINGLE,
        "single-fact test questions: the page holding the answer was found every time."],
      ["FaFlask", C.orange, "Giving the full answer", AGENT_PASS,
        "test questions answered correctly and fact-checked, including both trick questions it should refuse."],
    ];
    for (let i = 0; i < cards.length; i++) {
      const [ic, col, h, v, d] = cards[i];
      const x = 0.5 + i * 4.6;
      card(s, x, 1.3, 4.4, 2.75);
      await iconCircle(s, ic, x + 0.3, 1.5, 0.7, col);
      text(s, h, x + 1.15, 1.5, 3.1, 0.7, { fs: 18, bold: true, color: C.ink, valign: "middle" });
      text(s, v, x + 0.3, 2.3, 3.9, 0.8, { face: HEAD, fs: 40, bold: true, color: col, valign: "middle" });
      text(s, d, x + 0.3, 3.1, 3.85, 0.85, { fs: 14, color: C.body });
    }
    card(s, 0.5, 4.25, 9.0, 0.8, "FBE9DA", "FBE9DA");
    text(s, "Work in progress: one answer about three contracts left one out, and one two-part answer was held back by the fact-checker.",
      0.7, 4.25, 8.6, 0.8, { fs: 14, color: C.ink, valign: "middle" });
    s.addNotes("Left: make eval, rerank mode, run 2026-09-24: 'GATE (single recall) hit_rate=1.000' over the 20 single-fact questions " +
      "in eval/eval_retrieval.json (hit = at least one answer-bearing chunk in the top 10). Broad and complex questions also scored " +
      "hit_rate 1.000 but that is lenient (any one needed page); full-answer coverage is lower (broad 0.366, complex 0.597). " +
      "Right: make eval-agentic, run 2026-09-24: 9 questions, pass_rate 0.778. Failures: fanout 'contracts 24AJ0052, 24BG0272 and " +
      "24CM0001' missed 24CM0001; semantic '24BG0272 contractor + 24AJ0052 District Engineer' was withheld by the Verifier. " +
      "Both withhold (trick) questions passed. This is a single run with no earlier baseline.");
  }

  // ---------- 10. Why it's built this way ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.ink };
    text(s, "Why it’s built this way", 0.5, 0.35, 9, 0.7, { face: HEAD, fs: 30, bold: true, color: C.white, valign: "middle" });
    const cards = [
      ["FaLaptop", "Runs on one modest computer", "No cloud service or data centre needed.", C.teal],
      ["FaLock", "Keeps the data local", "Documents and questions are processed on that computer, not sent to an outside AI service.", C.teal],
      ["FaFileContract", "Trusts the official transcription", "Answers come only from the typed record of the bid documents.", C.teal],
      ["FaCalculator", "Leaves the maths to others", "It finds and checks facts; adding up amounts is left to people or a stronger system.", C.orange],
    ];
    const W = 4.4, H = 1.8;
    for (let i = 0; i < cards.length; i++) {
      const [ic, h, b, col] = cards[i];
      const x = 0.5 + (i % 2) * (W + 0.2), y = 1.3 + Math.floor(i / 2) * (H + 0.2);
      card(s, x, y, W, H, C.darkCard, C.darkLine);
      await iconCircle(s, ic, x + 0.25, y + 0.25, 0.65, col);
      text(s, h, x + 1.1, y + 0.2, W - 1.25, 0.75, { fs: 17, bold: true, color: C.white, valign: "middle" });
      text(s, b, x + 1.1, y + 0.95, W - 1.3, 0.8, { fs: 13.5, color: C.paleText });
    }
    s.addNotes("CPU-only by design: CPU torch for embedding and reranking; generation runs on a local Ollama server " +
      "(granite4.1:3b, num_ctx capped at 4096; rag/config.py notes it fits a 4GB GPU). Models are downloaded once, then run " +
      "locally. database/*.json is the authoritative OCR transcription and is never re-OCR'd. Sums are deferred to the handoff " +
      "model on purpose. Models are config-driven via RAG_* env vars (RAG_EMBED_MODEL, RAG_RERANK_MODEL, RAG_GEN_MODEL), so " +
      "stronger ones can be swapped in without code changes. Sources: CLAUDE.md, rag/config.py, rag/verify.py.");
  }

  const out = path.join(__dirname, "data-flow-deck.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote " + out);
}

build().catch((e) => { console.error(e); process.exit(1); });
