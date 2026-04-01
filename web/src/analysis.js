import { LANDMARK_INDEX, ANALYSIS_DEFINITIONS, DEFAULT_PIXEL_SPACING } from "./constants";

function getLandmark(landmarks, shortName) {
  const idx = LANDMARK_INDEX[shortName];
  if (idx === undefined || !landmarks[idx]) return null;
  return landmarks[idx];
}

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function angleDeg(a, vertex, b) {
  const v1 = { x: a.x - vertex.x, y: a.y - vertex.y };
  const v2 = { x: b.x - vertex.x, y: b.y - vertex.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const cross = v1.x * v2.y - v1.y * v2.x;
  let angle = Math.atan2(Math.abs(cross), dot) * (180 / Math.PI);
  return angle;
}

function angleLines(p1, p2, p3, p4) {
  const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
  const v2 = { x: p4.x - p3.x, y: p4.y - p3.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const cross = v1.x * v2.y - v1.y * v2.x;
  let angle = Math.atan2(Math.abs(cross), dot) * (180 / Math.PI);
  return angle;
}

export function computeAnalysis(analysisType, landmarks, pixelSpacing = DEFAULT_PIXEL_SPACING) {
  const defs = ANALYSIS_DEFINITIONS[analysisType];
  if (!defs || !landmarks || landmarks.length === 0) return [];

  const computed = {};
  const results = [];

  for (const def of defs) {
    let value = null;
    let status = "normal";

    try {
      if (def.type === "angle") {
        const [a, v, b] = def.points.map((s) => getLandmark(landmarks, s));
        if (a && v && b) value = angleDeg(a, v, b);
      } else if (def.type === "angle_lines") {
        const [p1, p2] = def.line1.map((s) => getLandmark(landmarks, s));
        const [p3, p4] = def.line2.map((s) => getLandmark(landmarks, s));
        if (p1 && p2 && p3 && p4) value = angleLines(p1, p2, p3, p4);
      } else if (def.type === "distance") {
        const [a, b] = def.points.map((s) => getLandmark(landmarks, s));
        if (a && b) value = dist(a, b) * pixelSpacing;
      } else if (def.type === "ratio") {
        const [[a1, a2], [b1, b2]] = def.segments.map((seg) =>
          seg.map((s) => getLandmark(landmarks, s))
        );
        if (a1 && a2 && b1 && b2) {
          const d1 = dist(a1, a2);
          const d2 = dist(b1, b2);
          if (d2 > 0) value = (d1 / d2) * 100;
        }
      } else if (def.type === "difference") {
        const aVal = computed[def.a];
        const bVal = computed[def.b];
        if (aVal !== null && aVal !== undefined && bVal !== null && bVal !== undefined) {
          value = aVal - bVal;
        }
      }
    } catch {
      value = null;
    }

    if (value !== null) {
      computed[def.name] = value;
      if (def.range) {
        if (value < def.range[0]) status = "low";
        else if (value > def.range[1]) status = "high";
      }
    }

    results.push({
      name: def.name,
      value: value !== null ? +value.toFixed(1) : null,
      units: def.units,
      normal: def.normal,
      status,
    });
  }

  return results;
}
