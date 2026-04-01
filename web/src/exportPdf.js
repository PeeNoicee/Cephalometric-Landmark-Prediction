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
 */
export function exportPdf({ canvasDataURL, landmarks, pixelSpacing, fileName, analysisType }) {
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
    const imgMaxH = 110;
    pdf.addImage(canvasDataURL, "PNG", margin, y, contentW, imgMaxH);
    y += imgMaxH + 6;
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
