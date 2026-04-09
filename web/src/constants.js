export const LANDMARK_NAMES = [
  "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion",
  "Orbitale", "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus",
  "Sella", "Articulare", "Condylion", "Gnathion", "Gonion",
  "Porion", "Lower 2nd PM Cusp Tip", "Lower Incisor Tip", "Lower Molar Cusp Tip",
  "Upper 2nd PM Cusp Tip", "Upper Incisor Apex", "Upper Incisor Tip",
  "Upper Molar Cusp Tip", "Lower Incisor Apex", "Labrale inferius",
  "Labrale superius", "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale",
];

export const LANDMARK_SHORT = [
  "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Prn", "R",
  "S", "Ar", "Co", "Gn", "Go", "Po", "L5", "L1", "L6",
  "U5", "U1A", "U1", "U6", "L1A", "Li", "Ls", "N'", "Pog'", "Sn",
];

export const LANDMARK_INDEX = Object.fromEntries(
  LANDMARK_SHORT.map((s, i) => [s, i])
);

export const LANDMARK_COLORS = [
  '#44FF00',
];

export const TRACING_SEGMENTS = {
  Steiner: [
    ["Cranial Base", "S", "N"],
    ["Maxillary Plane", "ANS", "PNS"],
    ["Mandibular Plane", "Go", "Gn"],
    ["Frankfort Horizontal", "Po", "Or"],
    ["Facial Axis", "N", "Gn"],
    ["E-line", "Prn", "Pog'"],
    ["Nasolabial", "Sn", "Ls"],
  ],
  Ricketts: [
    ["Facial Axis", "N", "Gn"],
    ["Mandibular Plane", "Go", "Gn"],
    ["Lower Facial Height", "ANS", "Me"],
    ["Total Facial Height", "N", "Me"],
    ["Occlusal Plane", "U6", "L6"],
    ["Bisector Occlusal", "U1", "L1"],
    ["Palatal Plane", "ANS", "PNS"],
  ],
  McNamara: [
    ["Cranial Base", "S", "N"],
    ["Maxillary Length", "Co", "A"],
    ["Mandibular Length", "Co", "Gn"],
    ["Anterior Facial Height", "N", "Me"],
    ["Facial Convexity", "A", "N'"],
    ["Mandibular Plane", "Go", "Gn"],
  ],
};

export const TRACING_MEASUREMENT_MAP = {
  Steiner: {
    "Mandibular Plane": "Mandibular Plane",
  },
  Ricketts: {
    "Mandibular Plane":    "Mandibular Plane Angle",
    "Lower Facial Height": "Lower Facial Height %",
    "Total Facial Height": "Lower Facial Height %",
    "Bisector Occlusal":   "Interincisal Angle",
    "Facial Axis":         "Facial Depth Angle",
  },
  McNamara: {
    "Maxillary Length":      "Maxillary Length",
    "Mandibular Length":     "Mandibular Length",
    "Anterior Facial Height":"Anterior Facial Height",
    "Facial Convexity":      "Facial Convexity",
  },
};

export const ANALYSIS_DEFINITIONS = {
  Steiner: [
    { name: "SNA", type: "angle", points: ["S", "N", "A"], units: "\u00B0", normal: "82\u00B0 \u00B1 2\u00B0", range: [80, 84] },
    { name: "SNB", type: "angle", points: ["S", "N", "B"], units: "\u00B0", normal: "80\u00B0 \u00B1 2\u00B0", range: [78, 82] },
    { name: "ANB", type: "difference", a: "SNA", b: "SNB", units: "\u00B0", normal: "2\u00B0 \u00B1 2\u00B0", range: [0, 4] },
    { name: "Mandibular Plane", type: "angle_lines", line1: ["S", "N"], line2: ["Go", "Gn"], units: "\u00B0", normal: "32\u00B0 \u00B1 3\u00B0", range: [29, 35] },
  ],
  Ricketts: [
    { name: "Interincisal Angle", type: "angle_lines", line1: ["U1A", "U1"], line2: ["L1A", "L1"], units: "\u00B0", normal: "130\u00B0 \u00B1 6\u00B0", range: [124, 136] },
    { name: "Lower Facial Height %", type: "ratio", segments: [["ANS", "Me"], ["N", "Me"]], units: "%", normal: "55% \u00B1 5%", range: [50, 60] },
    { name: "Mandibular Plane Angle", type: "angle_lines", line1: ["S", "N"], line2: ["Go", "Gn"], units: "\u00B0", normal: "26\u00B0 \u00B1 4\u00B0", range: [22, 30] },
    { name: "Facial Depth Angle", type: "angle_lines", line1: ["Po", "Or"], line2: ["N", "Pog"], units: "\u00B0", normal: "90\u00B0 \u00B1 3\u00B0", range: [87, 93] },
  ],
  McNamara: [
    { name: "Maxillary Length", type: "distance", points: ["Co", "A"], units: "mm", normal: "91 \u00B1 6 mm", range: [85, 97] },
    { name: "Mandibular Length", type: "distance", points: ["Co", "Gn"], units: "mm", normal: "120 \u00B1 7 mm", range: [113, 127] },
    { name: "Anterior Facial Height", type: "distance", points: ["N", "Me"], units: "mm", normal: "120 \u00B1 5 mm", range: [115, 125] },
    { name: "Facial Convexity", type: "point_to_line", point: "A", line: ["N", "Pog"], units: "mm", normal: "2 \u00B1 2 mm", range: [0, 4] },
  ],
};

export const DEFAULT_PIXEL_SPACING = 0.1;
