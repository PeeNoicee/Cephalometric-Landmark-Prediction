import { jsPDF } from "jspdf";
import { computeAnalysis } from "./analysis";
import { ANALYSIS_DEFINITIONS } from "./constants";

/**
 * Generate a Cephalometric Analysis Report PDF.
 *
 * @param {Object} opts
 * @param {string}  opts.canvasDataURL  - data URL from the annotated canvas
 * @param {Array}   opts.landmarks      - landmark array
 * @param {number}  opts.pixelSpacing   - mm per pixel
 * @param {string}  opts.fileName       - original file name
 * @param {string}  opts.analysisType   - currently selected analysis
 * @param {string}  opts.diagnosis       - AI-generated diagnosis text (optional)
 */
export function exportPdf({ canvasDataURL, landmarks, pixelSpacing, fileName, analysisType, diagnosis }) {
  const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageW = pdf.internal.pageSize.getWidth();   // 210
  const pageH = pdf.internal.pageSize.getHeight();  // 297
  const margin = 15;
  const contentW = pageW - margin * 2;
  let y = margin;

  // ── Colours ──
  const teal = [0, 172, 168];
  const dark = [15, 23, 42];
  const slate = [100, 116, 139];
  const white = [255, 255, 255];

  // ── Header bar ──
  pdf.setFillColor(...dark);
  pdf.rect(0, 0, pageW, 28, "F");
  pdf.setFillColor(...teal);
  pdf.rect(0, 27, pageW, 1.5, "F");

  pdf.setTextColor(...white);
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(16);
  pdf.text("Cephalometric Analysis Report", margin, 12);
  pdf.setFont("helvetica", "normal");
  pdf.setFontSize(9);
  pdf.setTextColor(160, 174, 192);
  pdf.text("AI-powered landmark detection \u00B7 29 landmarks", margin, 19);

  // Date on right
  const dateStr = new Date().toLocaleDateString("en-US", {
    year: "numeric", month: "long", day: "numeric",
  });
  pdf.setFontSize(9);
  pdf.text(dateStr, pageW - margin, 12, { align: "right" });
  if (fileName) {
    pdf.setFontSize(8);
    pdf.text(fileName, pageW - margin, 18, { align: "right" });
  }

  y = 35;

  // ── X-ray image ──
  if (canvasDataURL) {
    // Get image properties to calculate proper dimensions
    const imgInfo = pdf.getImageProperties(canvasDataURL);
    const aspectRatio = imgInfo.width / imgInfo.height;
    
    // Fixed max dimensions
    const imgMaxH = 110; // 110mm max height
    const imgMaxW = contentW * 0.9; // 90% of page width max
    
    // Calculate display size maintaining aspect ratio
    let displayH = imgMaxH;
    let displayW = displayH * aspectRatio;
    
    // If too wide, scale down proportionally
    if (displayW > imgMaxW) {
      displayW = imgMaxW;
      displayH = displayW / aspectRatio;
    }
    
    // Center the image horizontally
    const x = margin + (contentW - displayW) / 2;
    
    // Add image once at correct position with correct dimensions
    pdf.addImage(canvasDataURL, "PNG", x, y, displayW, displayH);
    y += displayH + 6;
  }

  // ── Section helper ──
  function sectionHeader(title) {
    if (y > pageH - 30) { pdf.addPage(); y = margin; }
    pdf.setFillColor(...teal);
    pdf.rect(margin, y, contentW, 7, "F");
    pdf.setTextColor(...white);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(10);
    pdf.text(title, margin + 3, y + 5);
    y += 10;
  }

  // ── Table helper ──
  function drawTable(rows) {
    const colW = [contentW * 0.35, contentW * 0.2, contentW * 0.25, contentW * 0.2];
    const headers = ["Measurement", "Value", "Normal", "Status"];

    // Header row
    if (y > pageH - 20) { pdf.addPage(); y = margin; }
    pdf.setFillColor(30, 41, 59);
    pdf.rect(margin, y, contentW, 6, "F");
    pdf.setTextColor(200, 210, 220);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(8);
    let x = margin + 2;
    headers.forEach((h, i) => {
      pdf.text(h, x, y + 4);
      x += colW[i];
    });
    y += 7;

    // Data rows
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(8);
    rows.forEach((r, rowIdx) => {
      if (y > pageH - 15) { pdf.addPage(); y = margin; }
      // Alternate row shading
      if (rowIdx % 2 === 0) {
        pdf.setFillColor(241, 245, 249);
        pdf.rect(margin, y - 1, contentW, 6, "F");
      }
      x = margin + 2;

      // Name
      pdf.setTextColor(30, 41, 59);
      pdf.text(r.name, x, y + 3);
      x += colW[0];

      // Value
      const valueStr = r.value !== null ? `${r.value}${r.units}` : "\u2014";
      if (r.status === "normal") pdf.setTextColor(16, 185, 129);
      else if (r.status === "high") pdf.setTextColor(244, 63, 94);
      else if (r.status === "low") pdf.setTextColor(245, 158, 11);
      else pdf.setTextColor(100, 116, 139);
      pdf.setFont("helvetica", "bold");
      pdf.text(valueStr, x, y + 3);
      x += colW[1];

      // Normal
      pdf.setTextColor(...slate);
      pdf.setFont("helvetica", "normal");
      pdf.text(r.normal || "", x, y + 3);
      x += colW[2];

      // Status
      const statusLabel = r.status === "normal" ? "Normal"
        : r.status === "high" ? "Above" : r.status === "low" ? "Below" : "";
      if (r.status === "normal") pdf.setTextColor(16, 185, 129);
      else if (r.status === "high") pdf.setTextColor(244, 63, 94);
      else if (r.status === "low") pdf.setTextColor(245, 158, 11);
      pdf.setFont("helvetica", "bold");
      pdf.text(statusLabel, x, y + 3);

      y += 6;
    });
    y += 4;
  }

  // ── Render analyses ──
  // Put the currently selected analysis first, then the rest
  const analysisOrder = [analysisType, ...Object.keys(ANALYSIS_DEFINITIONS).filter((t) => t !== analysisType)];

  for (const type of analysisOrder) {
    const results = computeAnalysis(type, landmarks, pixelSpacing);
    if (results.length === 0) continue;
    sectionHeader(`${type} Analysis`);
    drawTable(results);
  }

  // ── AI Diagnosis ──
  if (diagnosis) {
    sectionHeader("AI Clinical Diagnosis");
    renderMarkdown(diagnosis);
  }

  /**
   * Render a parsed markdown table (array of header strings + array of row arrays)
   * as a visual jsPDF table with coloured status cells.
   */
  function renderMdTable(headers, rows) {
    if (!headers.length || !rows.length) return;
    if (y > pageH - 20) { pdf.addPage(); y = margin; }

    const numCols = headers.length;
    const colW = contentW / numCols;
    const rowH = 6;

    // Helper: decide fill/text colour from a cell's content
    function cellColor(cell) {
      if (/High|\u2191/.test(cell))   return { text: [220, 38, 38],  bold: false }; // red
      if (/Low|\u2193/.test(cell))    return { text: [217, 119, 6],  bold: false }; // amber
      if (/Normal|\u2713/.test(cell)) return { text: [5, 150, 105],  bold: false }; // green
      if (/\u2190/.test(cell))        return { text: [79, 70, 229],  bold: true  }; // indigo = closest match
      return { text: [30, 41, 59], bold: false };
    }

    // Strip unicode arrows/checkmarks to ASCII for helvetica compatibility
    function cleanCell(cell) {
      return cell
        .replace(/\u2191\s*/g, "")   // ↑
        .replace(/\u2193\s*/g, "")   // ↓
        .replace(/\u2713\s*/g, "")   // ✓
        .replace(/\u2190/g, "<")     // ←
        .replace(/\*\*/g, "")        // bold markers
        .trim();
    }

    // Header row
    pdf.setFillColor(30, 41, 59);
    pdf.rect(margin, y, contentW, rowH, "F");
    pdf.setTextColor(200, 210, 220);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(7.5);
    headers.forEach((h, i) => {
      pdf.text(cleanCell(h), margin + i * colW + 2, y + 4);
    });
    y += rowH;

    // Data rows
    pdf.setFontSize(7.5);
    rows.forEach((row, rowIdx) => {
      if (y > pageH - 15) { pdf.addPage(); y = margin; }
      if (rowIdx % 2 === 0) {
        pdf.setFillColor(241, 245, 249);
        pdf.rect(margin, y, contentW, rowH, "F");
      }
      row.forEach((cell, ci) => {
        const { text: tc, bold } = cellColor(cell);
        pdf.setTextColor(...tc);
        pdf.setFont("helvetica", bold ? "bold" : "normal");
        pdf.text(cleanCell(cell), margin + ci * colW + 2, y + 4);
      });
      y += rowH;
    });
    y += 4;
  }

  /**
   * Render markdown-formatted diagnosis text into the PDF with proper formatting.
   * Handles: ### headings, > blockquotes, bullet lists, numbered lists,
   * markdown pipe tables, *italic* captions, and **bold** inline text.
   */
  function renderMarkdown(md) {
    const textX = margin + 3;
    const bulletX = margin + 6;
    const quoteX = margin + 5;
    const textW = contentW - 8;
    const quoteW = contentW - 10;
    // Replace emoji warning symbol with text for PDF compatibility
    const safeMd = md
      .replace(/\u26a0/g, "[WARNING]")
      .replace(/\u2714/g, "[OK]")
      .replace(/\u{1F6AB}/gu, "[BLOCK]");
    const rawLines = safeMd.split("\n");

    let i = 0;
    while (i < rawLines.length) {
      const raw = rawLines[i];
      const trimmed = raw.trim();

      if (!trimmed) { y += 2; i++; continue; }
      if (y > pageH - 18) { pdf.addPage(); y = margin; }

      // ── Markdown table detection ──────────────────────────────────────
      // A table starts with a header row (|...|) immediately followed by a
      // separator row (|---|) on the next line.
      const isTableRow = (s) => /^\|.+\|$/.test(s);
      const isSeparator = (s) => /^\|[\s\-:|]+\|$/.test(s);
      if (
        isTableRow(trimmed) &&
        i + 1 < rawLines.length &&
        isSeparator(rawLines[i + 1].trim())
      ) {
        // Parse header columns
        const headers = trimmed
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        i += 2; // skip header + separator
        const tableRows = [];
        while (i < rawLines.length && isTableRow(rawLines[i].trim())) {
          const cols = rawLines[i]
            .trim()
            .split("|")
            .slice(1, -1)
            .map((c) => c.trim());
          tableRows.push(cols);
          i++;
        }
        renderMdTable(headers, tableRows);
        continue;
      }

      // ── Blockquote: > text ────────────────────────────────────────────
      if (/^>/.test(trimmed)) {
        const quoteText = trimmed.replace(/^>+\s*/, "");
        if (quoteText) {
          pdf.setFillColor(226, 232, 240);
          pdf.rect(margin, y, 2, 4.5, "F");
          pdf.setTextColor(71, 85, 105);
          pdf.setFont("helvetica", "italic");
          pdf.setFontSize(8);
          const wrapped = pdf.splitTextToSize(stripBold(quoteText), quoteW);
          wrapped.forEach((line) => {
            if (y > pageH - 15) { pdf.addPage(); y = margin; }
            pdf.text(line, quoteX, y + 3);
            y += 4.5;
          });
        } else {
          y += 1;
        }
        i++; continue;
      }

      // ── ### Heading ───────────────────────────────────────────────────
      if (/^#{1,3}\s/.test(trimmed)) {
        const headingText = trimmed.replace(/^#{1,3}\s+/, "");
        y += 2;
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9.5);
        pdf.setTextColor(0, 140, 135);
        pdf.text(stripBold(headingText), textX, y + 3);
        y += 6;
        i++; continue;
      }

      // ── Italic caption line: *text* ───────────────────────────────────
      if (/^\*[^*].*[^*]\*$/.test(trimmed)) {
        const caption = trimmed.replace(/^\*|\*$/g, "");
        pdf.setFont("helvetica", "italic");
        pdf.setFontSize(7.5);
        pdf.setTextColor(100, 116, 139);
        const wrapped = pdf.splitTextToSize(caption, textW);
        wrapped.forEach((line) => {
          if (y > pageH - 15) { pdf.addPage(); y = margin; }
          pdf.text(line, textX, y + 3);
          y += 4;
        });
        i++; continue;
      }

      // ── Numbered list: 1. text ────────────────────────────────────────
      const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
      if (numMatch) {
        renderInlineFormatted(numMatch[2], bulletX, textW - 4, true, numMatch[1] + ". ");
        i++; continue;
      }

      // ── Bullet: - text or  - text (indented) ─────────────────────────
      if (/^[-*]\s/.test(trimmed)) {
        const bulletText = trimmed.replace(/^[-*]\s+/, "");
        renderInlineFormatted(bulletText, bulletX, textW - 4, true);
        i++; continue;
      }

      // ── Normal paragraph ─────────────────────────────────────────────
      renderInlineFormatted(trimmed, textX, textW, false);
      i++;
    }
    y += 3;
  }

  /**
   * Render a line of text that may contain **bold** segments.
   * Handles word-wrapping within the available width.
   */
  function renderInlineFormatted(text, startX, maxW, isBullet, bulletPrefix) {
    pdf.setFontSize(8.5);
    const lineH = 4;

    // Split into segments: alternating normal / bold
    const parts = [];
    const regex = /\*\*(.+?)\*\*/g;
    let last = 0;
    let match;
    while ((match = regex.exec(text)) !== null) {
      if (match.index > last) parts.push({ t: text.slice(last, match.index), bold: false });
      parts.push({ t: match[1], bold: true });
      last = regex.lastIndex;
    }
    if (last < text.length) parts.push({ t: text.slice(last), bold: false });

    // Build word list with formatting
    const words = [];
    if (isBullet && bulletPrefix) {
      words.push({ w: bulletPrefix, bold: true });
    } else if (isBullet) {
      words.push({ w: "\u2022 ", bold: false });
    }
    for (const part of parts) {
      for (const w of part.t.split(/\s+/).filter(Boolean)) {
        words.push({ w, bold: part.bold });
      }
    }

    // Word-wrap and render
    let curX = startX;
    let firstLine = true;
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      pdf.setFont("helvetica", word.bold ? "bold" : "normal");
      pdf.setTextColor(word.bold ? 20 : 50, word.bold ? 30 : 55, word.bold ? 50 : 75);
      const ww = pdf.getTextWidth(word.w + " ");
      if (curX + ww > startX + maxW && curX > startX + 2) {
        y += lineH;
        if (y > pageH - 15) { pdf.addPage(); y = margin; }
        curX = startX;
      }
      pdf.text(word.w, curX, y + 3);
      curX += ww;
    }
    y += lineH;
  }

  function stripBold(s) { return s.replace(/\*\*/g, ""); }

  // ── Footer ──
  const totalPages = pdf.internal.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    pdf.setPage(p);
    pdf.setFontSize(7);
    pdf.setTextColor(160, 174, 192);
    pdf.text(
      `Generated by Cephalometric Landmark Detection AI \u00B7 Page ${p} of ${totalPages}`,
      pageW / 2, pageH - 8, { align: "center" }
    );
  }

  pdf.save(`cephalometric_report_${Date.now()}.pdf`);
}
