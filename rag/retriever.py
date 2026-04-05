"""
Hybrid cephalometric diagnosis engine:
  - Skeletal Pattern & Key Findings: deterministic rule-based (instant)
  - Clinical Summary: ceph-summary (custom Ollama model with baked-in textbook rules)
  - Fallback: deterministic summary if LLM call fails
  - Textbook norms: Steiner, Ricketts, McNamara (per Cephalometric Analysis Guidelines).
"""
import os
import ollama

LLM_MODEL = "ceph-summary"


# ---------------------------------------------------------------------------
# Helper: get measurement by name
# ---------------------------------------------------------------------------
def _get(measurements, name):
    for m in measurements:
        if m["name"] == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Confidence Assessment
# ---------------------------------------------------------------------------
# Landmarks whose mis-detection most heavily contaminates downstream measurements.
# Maps short name → list of affected measurement names.
_HIGH_IMPACT_LANDMARKS = {
    "S":   ["SNA", "SNB", "ANB", "Mandibular Plane"],
    "N":   ["SNA", "SNB", "ANB", "Mandibular Plane", "Facial Depth Angle"],
    "Pog": ["Facial Depth Angle", "Facial Convexity"],
    "Po":  ["Mandibular Plane Angle", "Facial Depth Angle"],
    "Or":  ["Mandibular Plane Angle", "Facial Depth Angle"],
    "A":   ["SNA", "ANB", "Facial Convexity", "Maxillary Length"],
    "B":   ["SNB", "ANB"],
    "Co":  ["Maxillary Length", "Mandibular Length"],
    "Gn":  ["Mandibular Length", "Mandibular Plane"],
    "Go":  ["Mandibular Plane"],
    "Me":  ["Anterior Facial Height", "Mandibular Plane Angle"],
}

# Heatmap confidence below this fraction of the best-detected landmark is flagged.
_LOW_CONF_THRESHOLD = 0.40


def _assess_confidence(measurements, landmark_confidences=None):
    """
    Assesses overall report reliability from cross-analysis contradictions,
    extreme values, and internal landmark consistency.
    Args:
        measurements: list of measurement dicts
        landmark_confidences: dict mapping landmark short name → confidence score (0-1),
                              where 1.0 = most confident landmark in this image.
    Returns:
        dict with keys: level, contradictions, extreme_flags,
                        suspect_landmarks, unreliable_measurements, age_warning
    """
    contradictions = []
    extreme_flags = []
    suspect_landmarks = set()
    unreliable_measurements = set()

    # --- Per-landmark heatmap confidence pre-check ---
    # Flag high-impact landmarks with low heatmap peak values BEFORE
    # cross-analysis checks. This catches errors that may not yet have
    # produced measurement contradictions (e.g. slightly off but not extreme).
    if landmark_confidences:
        for lm_short, conf in landmark_confidences.items():
            if lm_short in _HIGH_IMPACT_LANDMARKS and conf < _LOW_CONF_THRESHOLD:
                suspect_landmarks.add(lm_short)
                unreliable_measurements.update(_HIGH_IMPACT_LANDMARKS[lm_short])
                extreme_flags.append(
                    f"{lm_short} has low heatmap detection confidence "
                    f"({conf:.0%} of best-detected landmark) — placement may be inaccurate"
                )

    snb = _get(measurements, "SNB")
    fda = _get(measurements, "Facial Depth Angle")
    anb = _get(measurements, "ANB")
    facial_conv = _get(measurements, "Facial Convexity")

    # Cross-analysis contradiction: SNB vs Facial Depth Angle
    if snb and fda and snb["value"] is not None and fda["value"] is not None:
        if snb["status"] == "low" and fda["status"] == "high":
            contradictions.append(
                "SNB indicates recessive mandible, but Ricketts Facial Depth indicates a forward chin"
            )
            suspect_landmarks.update(["S", "N", "Pog", "Po", "Or"])
            unreliable_measurements.update(
                ["SNA", "SNB", "ANB", "Facial Depth Angle", "Mandibular Plane Angle"]
            )
        elif snb["status"] == "high" and fda["status"] == "low":
            contradictions.append(
                "SNB indicates prognathic mandible, but Ricketts Facial Depth indicates a retrusive chin"
            )
            suspect_landmarks.update(["S", "N", "Pog", "Po", "Or"])
            unreliable_measurements.update(
                ["SNA", "SNB", "ANB", "Facial Depth Angle", "Mandibular Plane Angle"]
            )

    # Profile consistency: ANB class vs Facial Convexity
    if anb and facial_conv and anb["value"] is not None and facial_conv["value"] is not None:
        if anb["value"] > 4 and facial_conv["status"] == "low":
            contradictions.append(
                "ANB indicates Class II skeletal pattern, but Facial Convexity indicates a Class III profile"
            )
            suspect_landmarks.update(["A", "N", "Pog"])
            unreliable_measurements.update(["ANB", "SNA", "Facial Convexity"])
        elif anb["value"] < 0 and facial_conv["status"] == "high":
            contradictions.append(
                "ANB indicates Class III skeletal pattern, but Facial Convexity indicates a Class II profile"
            )
            suspect_landmarks.update(["A", "N", "Pog"])
            unreliable_measurements.update(["ANB", "SNA", "Facial Convexity"])

    # Extreme value detection (values outside physiological range)
    _extreme_checks = [
        ("SNA", 65, 95, ["S", "N", "A"], ["SNA", "ANB"]),
        ("SNB", 65, 92, ["S", "N", "B"], ["SNB", "ANB"]),
        ("ANB", -10, 15, ["A", "B"], ["ANB"]),
        ("Mandibular Plane", 15, 55, ["Go", "Gn", "S", "N"], ["Mandibular Plane"]),
        ("Mandibular Plane Angle", 10, 50, ["Po", "Or", "Go", "Me"], ["Mandibular Plane Angle"]),
        ("Interincisal Angle", 90, 170, ["U1", "L1"], ["Interincisal Angle"]),
        ("Facial Depth Angle", 75, 105, ["N", "Pog", "Po", "Or"], ["Facial Depth Angle"]),
    ]
    for mname, lo, hi, lms, affected in _extreme_checks:
        m = _get(measurements, mname)
        if m and m["value"] is not None:
            if m["value"] < lo or m["value"] > hi:
                extreme_flags.append(
                    f"{mname} = {m['value']} (physiological range {lo}\u2013{hi})"
                )
                suspect_landmarks.update(lms)
                unreliable_measurements.update(affected)

    # Ricketts age warning: any Ricketts measurement present implies age-dependent norms
    ricketts_names = ["Mandibular Plane Angle", "Facial Depth Angle", "Facial Convexity"]
    age_warning = any(
        _get(measurements, n) is not None and _get(measurements, n)["value"] is not None
        for n in ricketts_names
    )

    # Determine overall confidence level
    if contradictions and len(contradictions) >= 2:
        level = "NEEDS_REVIEW"
    elif contradictions:
        level = "LOW"
    elif len(extreme_flags) >= 2:
        level = "LOW"
    elif extreme_flags:
        level = "MODERATE"
    else:
        level = "HIGH"

    return {
        "level": level,
        "contradictions": contradictions,
        "extreme_flags": extreme_flags,
        "suspect_landmarks": sorted(suspect_landmarks),
        "unreliable_measurements": unreliable_measurements,
        "age_warning": age_warning,
    }


def _build_confidence_header(confidence):
    """Formats the Report Confidence block shown at the top of the diagnostic report."""
    level = confidence["level"]
    label_map = {
        "HIGH":         "**High** \u2014 findings are internally consistent",
        "MODERATE":     "**Moderate** \u2014 minor inconsistencies detected; verify flagged landmarks before relying on derived findings",
        "LOW":          "**Low** \u2014 significant contradictions detected; interpret all findings with caution",
        "NEEDS_REVIEW": "**Needs Review** \u2014 multiple contradictions make this report unreliable for clinical use",
    }
    icon_map = {"HIGH": "\u2713", "MODERATE": "\u26a0", "LOW": "\u26a0", "NEEDS_REVIEW": "\U0001f6ab"}

    lines = [f"### {icon_map[level]} Report Confidence: {label_map[level]}"]

    if confidence["age_warning"]:
        lines.append(
            "\n> **Age Unknown \u2014 Patient age was not provided.** "
            "Ricketts norms for Mandibular Plane Angle, Facial Depth Angle, and Facial Convexity "
            "are all age-adjusted (norms shift every 3 years). "
            "Rather than using a single fixed baseline, the Key Findings section below shows "
            "each measurement classified against **all age checkpoints (9, 12, 15, 18, 21, 25)** "
            "so the clinician can apply the appropriate norm for this patient. "
            "**Providing patient age allows a single definitive age-adjusted interpretation.**"
        )

    if confidence["contradictions"]:
        lines.append("\n**Anatomical Contradictions Detected**:")
        for c in confidence["contradictions"]:
            lines.append(f"  - {c}")
        if confidence["suspect_landmarks"]:
            lines.append(
                f"\n**Landmarks requiring manual verification**: "
                f"{', '.join(confidence['suspect_landmarks'])}"
            )
        if confidence["unreliable_measurements"]:
            lines.append(
                f"\n**Measurements with reduced reliability** "
                f"(derived from suspect landmarks): "
                f"{', '.join(sorted(confidence['unreliable_measurements']))}"
            )

    if confidence["extreme_flags"]:
        lines.append("\n**Values outside physiological range** (strongly suggest landmark errors):")
        for f in confidence["extreme_flags"]:
            lines.append(f"  - {f}")

    if level in ("LOW", "NEEDS_REVIEW"):
        lines.append(
            "\n> **Clinical Action Required**: Manually verify the flagged landmarks and "
            "re-run analysis before using this report for treatment planning."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule-based: Skeletal Pattern
# ---------------------------------------------------------------------------
def _build_skeletal_pattern(measurements):
    lines = []

    anb = _get(measurements, "ANB")
    sna = _get(measurements, "SNA")
    snb = _get(measurements, "SNB")

    # --- Skeletal classification from ANB (Steiner: norm 2°) ---
    if anb and anb["value"] is not None:
        v = anb["value"]
        if v > 4:
            lines.append(f"**Skeletal Class II**: ANB of {v}° (Steiner norm 2°) indicates a significant maxillomandibular discrepancy with the maxilla positioned ahead of the mandible.")
        elif v > 2:
            lines.append(f"**Skeletal Class II tendency**: ANB of {v}° (Steiner norm 2°) indicates a mild Class II skeletal relationship.")
        elif v < 0:
            lines.append(f"**Skeletal Class III**: ANB of {v}° (Steiner norm 2°) indicates the mandible is ahead of the maxilla, suggesting a Class III skeletal relationship.")
        elif v < 2:
            lines.append(f"**Skeletal Class III tendency**: ANB of {v}° (Steiner norm 2°) suggests a mild Class III tendency with the mandible slightly ahead.")
        else:
            lines.append(f"**Skeletal Class I**: ANB of {v}° (Steiner norm 2°) indicates a balanced maxillomandibular relationship.")

    # --- SNA: maxilla to cranial base (Steiner: norm 82°) ---
    if sna and sna["value"] is not None:
        v = sna["value"]
        if sna["status"] == "high":
            lines.append(f"SNA of {v}° (Steiner norm 82°) indicates a relative forward positioning or protrusive maxilla.")
        elif sna["status"] == "low":
            lines.append(f"SNA of {v}° (Steiner norm 82°) indicates a relative backward or recessive location of the maxilla.")
        else:
            lines.append(f"SNA of {v}° (Steiner norm 82°) is within normal limits — maxilla is normally positioned relative to the cranial base.")

    # --- SNB: mandible to cranial base (Steiner: norm 80°) ---
    if snb and snb["value"] is not None:
        v = snb["value"]
        if snb["status"] == "high":
            lines.append(f"SNB of {v}° (Steiner norm 80°) suggests a prognathic mandible.")
        elif snb["status"] == "low":
            lines.append(f"SNB of {v}° (Steiner norm 80°) indicates a recessive mandible.")
        else:
            lines.append(f"SNB of {v}° (Steiner norm 80°) is within normal limits — mandible is normally positioned relative to the cranial base.")

    # --- Source of Class II discrepancy ---
    if anb and anb["value"] is not None and anb["value"] > 2:
        sna_s = sna["status"] if sna else None
        snb_s = snb["status"] if snb else None
        if snb_s == "low" and sna_s in ("normal", None):
            lines.append("The Class II relationship is primarily due to a **recessive mandible** (low SNB) with a normally positioned maxilla.")
        elif sna_s == "high" and snb_s in ("normal", None):
            lines.append("The Class II relationship is primarily due to a **protrusive maxilla** (high SNA) with a normally positioned mandible.")
        elif sna_s == "high" and snb_s == "low":
            lines.append("The Class II relationship involves both a **protrusive maxilla** (high SNA) and a **recessive mandible** (low SNB).")
        elif sna_s == "normal" and snb_s == "normal":
            lines.append("Both SNA and SNB are within normal limits — the elevated ANB may reflect a **dentoalveolar** rather than skeletal discrepancy.")

    # --- Growth pattern from mandibular plane angles ---
    # Steiner GoGn-SN: norm 32° (excessively high or low = unfavorable growth)
    # Ricketts FH-MP: norm at age 9 = 26°, decreases -1°/3yrs; high = open bite, low = deep bite
    # NOTE: MPA measures VERTICAL growth direction only, NOT anteroposterior jaw position.
    steiner_mp = _get(measurements, "Mandibular Plane")
    ricketts_mp = _get(measurements, "Mandibular Plane Angle")

    indicators = []  # (pattern, description)
    if steiner_mp and steiner_mp["value"] is not None:
        v = steiner_mp["value"]
        if steiner_mp["status"] == "high":
            indicators.append(("hyper", f"Steiner GoGn-SN {v}° is above the mean of 32°, suggesting an unfavorable vertical growth pattern"))
        elif steiner_mp["status"] == "low":
            indicators.append(("hypo", f"Steiner GoGn-SN {v}° is below the mean of 32°, suggesting a horizontal growth tendency"))
        else:
            indicators.append(("normal", f"Steiner GoGn-SN {v}° is within normal range (mean 32°)"))

    if ricketts_mp and ricketts_mp["value"] is not None:
        v = ricketts_mp["value"]
        if ricketts_mp["status"] == "high":
            indicators.append(("hyper", f"Ricketts FH-MP {v}° is above normal (norm 26° at age 9, adjusted with age), implying an open bite tendency"))
        elif ricketts_mp["status"] == "low":
            indicators.append(("hypo", f"Ricketts FH-MP {v}° is below normal (norm 26° at age 9, adjusted with age), implying a skeletal deep bite"))
        else:
            indicators.append(("normal", f"Ricketts FH-MP {v}° is within normal range (norm 26° ± 4°)"))

    if indicators:
        patterns = [i[0] for i in indicators]
        details = "; ".join(i[1] for i in indicators)

        if all(p == "hyper" for p in patterns):
            lines.append(f"**Hyperdivergent growth pattern** (vertical growth, long face): {details}.")
        elif all(p == "hypo" for p in patterns):
            lines.append(f"**Hypodivergent growth pattern** (horizontal growth, short face): {details}.")
        elif all(p == "normal" for p in patterns):
            lines.append(f"**Normal growth pattern**: {details}.")
        else:
            # Cross-analysis disagreement — different reference planes and thresholds
            steiner_val = steiner_mp["value"] if steiner_mp and steiner_mp["value"] is not None else None
            ricketts_val = ricketts_mp["value"] if ricketts_mp and ricketts_mp["value"] is not None else None
            steiner_s = steiner_mp["status"] if steiner_mp else None
            ricketts_s = ricketts_mp["status"] if ricketts_mp else None
            threshold_note = ""
            if steiner_s == "normal" and ricketts_s == "high":
                threshold_note = (
                    f" Steiner GoGn-SN {steiner_val}° falls within its wider normal range (32° ± 3°), "
                    f"while Ricketts FH-MP {ricketts_val}° exceeds its threshold (26° ± 4° at age 9). "
                    f"A definitive vertical classification cannot be made — "
                    f"clinical correlation with facial proportions is recommended."
                )
            elif steiner_s == "high" and ricketts_s == "normal":
                threshold_note = (
                    f" Steiner GoGn-SN {steiner_val}° exceeds its threshold (32° ± 3°), "
                    f"while Ricketts FH-MP {ricketts_val}° falls within its normal range (26° ± 4° at age 9). "
                    f"A definitive vertical classification cannot be made — "
                    f"clinical correlation with facial proportions is recommended."
                )
            lines.append(
                f"**Mixed growth pattern indicators**: {details}. "
                f"Note: Steiner uses the SN plane while Ricketts uses Frankfort Horizontal (FH), "
                f"and they have different normal ranges.{threshold_note}"
            )

    # --- Profile ---
    # Ricketts Convexity of Point A (A to N-Pog): norm at age 9 = 2mm, decreases -1mm/3yrs
    # High positive = Class II skeletal pattern; low/negative = Class III tendency
    facial_conv = _get(measurements, "Facial Convexity")
    if anb and anb["value"] is not None:
        v = anb["value"]
        if v > 2:
            profile = "Convex"
        elif v < 0:
            profile = "Concave"
        elif v < 2:
            profile = "Mildly concave"
        else:
            profile = "Straight"

        conv_note = ""
        if facial_conv and facial_conv["value"] is not None:
            fc = facial_conv["value"]
            if facial_conv["status"] == "high":
                conv_note = f", supported by Ricketts Convexity of Point A at {fc}mm (high positive value suggests a Class II skeletal pattern; norm 2mm at age 9, decreases with age)"
            elif facial_conv["status"] == "low":
                conv_note = f", with Ricketts Convexity of Point A at {fc}mm (low/negative value suggests a Class III tendency; norm 2mm at age 9, decreases with age)"
            else:
                conv_note = f", Ricketts Convexity of Point A at {fc}mm within normal range"

        lines.append(f"**{profile} profile**{conv_note}.")

    return "### Skeletal Pattern\n\n" + "\n\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Ricketts age-adjusted norm helpers
# ---------------------------------------------------------------------------
# Ricketts growth increments per textbook:
#   Mandibular Plane Angle: age-9 norm 26°, -1°/3 yrs
#   Facial Depth Angle:     age-9 norm 87°, +1°/3 yrs
#   Facial Convexity:       age-9 norm 2mm, -1mm/3 yrs
_RICKETTS_AGE_NORMS = {
    "Mandibular Plane Angle": {"base_age": 9, "base_norm": 26.0, "delta_per_3yr": -1.0, "sd": 4.0, "units": "\u00b0"},
    "Facial Depth Angle":     {"base_age": 9, "base_norm": 87.0, "delta_per_3yr": +1.0, "sd": 3.0, "units": "\u00b0"},
    "Facial Convexity":       {"base_age": 9, "base_norm":  2.0, "delta_per_3yr": -1.0, "sd": 2.0, "units": "mm"},
}
_RICKETTS_AGE_CHECKPOINTS = [9, 12, 15, 18, 21, 25]


def _ricketts_age_table(name, value):
    """Return a Markdown table of how *value* classifies against Ricketts
    age-adjusted norms at each standard age checkpoint.
    Also returns the age whose norm is closest to the patient value.
    """
    r = _RICKETTS_AGE_NORMS.get(name)
    if not r or value is None:
        return "", None

    u = r["units"]
    rows = []
    closest_age = None
    min_diff = float("inf")

    for age in _RICKETTS_AGE_CHECKPOINTS:
        years_from_base = age - r["base_age"]
        norm = r["base_norm"] + (years_from_base / 3) * r["delta_per_3yr"]
        lo, hi = norm - r["sd"], norm + r["sd"]
        if value < lo:
            status = "\u2193 Low"
        elif value > hi:
            status = "\u2191 High"
        else:
            status = "\u2713 Normal"
        rows.append((age, round(norm, 1), round(lo, 1), round(hi, 1), status))
        if abs(value - norm) < min_diff:
            min_diff = abs(value - norm)
            closest_age = age

    lines = [
        f"| Age | Norm | Normal Range | Patient ({value}{u}) |",
        "|:---:|:----:|:------------:|:-------:|",
    ]
    for age, norm, lo, hi, status in rows:
        marker = " \u2190" if age == closest_age else ""
        lines.append(f"| {age} | {norm}{u} | {lo}\u2013{hi}{u} | {status}{marker} |")
    lines.append(
        f"\n*{value}{u} most closely matches the Ricketts norm for age {closest_age}.*"
    )
    return "\n".join(lines), closest_age


# ---------------------------------------------------------------------------
# Rule-based: Key Findings
# ---------------------------------------------------------------------------
def _build_key_findings(measurements):
    findings = []

    # --- McNamara Proportional Analysis ---
    # Per McNamara philosophy: "A mandible is not 'short' just because it's less
    # than 120mm; it's only short if it's less than 25mm longer than the maxilla."
    # Proportionality to each other > absolute population norms.
    max_len = _get(measurements, "Maxillary Length")
    mand_len = _get(measurements, "Mandibular Length")
    afh = _get(measurements, "Anterior Facial Height")

    if max_len and mand_len and max_len["value"] and mand_len["value"]:
        co_a = max_len["value"]
        co_gn = mand_len["value"]
        diff = co_gn - co_a

        # McNamara Table 10-1 — linear interpolation from anchor points
        # Anchor points: (Co-A, Co-Gn low, Co-Gn high, LAFH low, LAFH high)
        _table = [
            (85, 105, 108, 60, 62),
            (87, 108, 110, 61, 63),
            (89, 111, 113, 63, 65),
            (91, 114, 116, 64, 66),
            (92, 117, 120, 65, 67),
            (94, 121, 124, 66, 67),
            (97, 127, 130, 68, 71),
            (100, 130, 133, 70, 74),
        ]

        # Find bracketing entries and interpolate
        if co_a <= _table[0][0]:
            ideal_lo, ideal_hi = _table[0][1], _table[0][2]
            lafh_lo, lafh_hi = _table[0][3], _table[0][4]
        elif co_a >= _table[-1][0]:
            ideal_lo, ideal_hi = _table[-1][1], _table[-1][2]
            lafh_lo, lafh_hi = _table[-1][3], _table[-1][4]
        else:
            for i in range(len(_table) - 1):
                if _table[i][0] <= co_a <= _table[i + 1][0]:
                    t = (co_a - _table[i][0]) / (_table[i + 1][0] - _table[i][0])
                    ideal_lo = round(_table[i][1] + t * (_table[i + 1][1] - _table[i][1]))
                    ideal_hi = round(_table[i][2] + t * (_table[i + 1][2] - _table[i][2]))
                    lafh_lo = round(_table[i][3] + t * (_table[i + 1][3] - _table[i][3]))
                    lafh_hi = round(_table[i][4] + t * (_table[i + 1][4] - _table[i][4]))
                    break

        ideal_diff_lo = ideal_lo - co_a
        ideal_diff_hi = ideal_hi - co_a

        # McNamara face size category (from Table 10-1)
        if co_a <= 87:
            size_cat = "Small"
        elif co_a <= 96:
            size_cat = "Medium"
        else:
            size_cat = "Large"

        # McNamara Maxillomandibular Differential category
        # Small: diff 20-23mm; Medium: diff 25-27mm; Large: diff 30-33mm
        if diff < 20:
            diff_cat = "below the Small category range (20-23mm)"
        elif diff <= 23:
            diff_cat = "within the Small category range (20-23mm)"
        elif diff < 25:
            diff_cat = "between Small and Medium ranges"
        elif diff <= 27:
            diff_cat = "within the Medium category range (25-27mm)"
        elif diff < 30:
            diff_cat = "between Medium and Large ranges"
        elif diff <= 33:
            diff_cat = "within the Large category range (30-33mm)"
        else:
            diff_cat = "above the Large category range (30-33mm)"

        # ±2mm tolerance for measurement variability
        if ideal_lo - 2 <= co_gn <= ideal_hi + 2:
            findings.append(
                f"**McNamara**: Co-A {co_a:.1f}mm · Co-Gn {co_gn:.1f}mm — jaw lengths **proportional** "
                f"(expected {ideal_lo}\u2013{ideal_hi}mm). Differential: {diff:.1f}mm ({diff_cat})."
            )
        elif co_gn < ideal_lo - 2:
            findings.append(
                f"**McNamara**: Co-A {co_a:.1f}mm · Co-Gn {co_gn:.1f}mm — mandible **short relative to midface** "
                f"(expected {ideal_lo}\u2013{ideal_hi}mm). Differential: {diff:.1f}mm ({diff_cat})."
            )
        else:
            findings.append(
                f"**McNamara**: Co-A {co_a:.1f}mm · Co-Gn {co_gn:.1f}mm — mandible **long relative to midface** "
                f"(expected {ideal_lo}\u2013{ideal_hi}mm). Differential: {diff:.1f}mm ({diff_cat})."
            )

    if afh and afh["value"] is not None and afh["status"] != "normal":
        v = afh["value"]
        word = "increased" if afh["status"] == "high" else "decreased"
        findings.append(f"**Anterior Facial Height**: {v}mm is {word} (norm 120 ± 5mm).")

    # --- Interincisal Angle (Steiner: norm 131° per Table 7-1, or 130° general) ---
    # Acute (< 130°): teeth require uprighting. Obtuse (> 130°): teeth require advancing.
    ia = _get(measurements, "Interincisal Angle")
    if ia and ia["value"] is not None and ia["status"] != "normal":
        v = ia["value"]
        if ia["status"] == "low":
            findings.append(f"**Interincisal Angle**: {v}\u00b0 (\u2193 below 131\u00b0 norm) \u2014 **proclined incisors**, uprighting likely needed.")
        else:
            findings.append(f"**Interincisal Angle**: {v}\u00b0 (\u2191 above 131\u00b0 norm) \u2014 **retroclined incisors**, advancement may be needed.")

    # --- Facial Depth Angle (Ricketts: N-Pog to FH) ---
    # Ricketts norm: 87° at age 9, +1°/3 yrs. System default: 90° ± 3° (≈ age-18 baseline).
    # Classified as high if > 93° (forward chin), low if < 87° (retrusive chin).
    fda = _get(measurements, "Facial Depth Angle")
    if fda and fda["value"] is not None and fda["status"] != "normal":
        v = fda["value"]
        if fda["status"] == "high":
            findings.append(f"**Facial Depth Angle**: {v}\u00b0 (\u2191 above 90\u00b0 default) \u2014 **forward chin**; mandible not the primary cause of any Class II pattern.")
        else:
            findings.append(f"**Facial Depth Angle**: {v}\u00b0 (\u2193 below 90\u00b0 default) \u2014 **retrusive chin**; mandibular contribution to Class II pattern.")

    # --- Lower Facial Height (Ricketts: Xi-ANS to Xi-PM, norm 45° ± 4°) ---
    # Low values indicate a skeletal deep bite.
    lfh = _get(measurements, "Lower Facial Height %")
    if lfh and lfh["value"] is not None and lfh["status"] != "normal":
        v = lfh["value"]
        if lfh["status"] == "high":
            findings.append(f"**Lower Facial Height**: {v}% (\u2191 above 55% norm) \u2014 vertical excess, open bite tendency.")
        else:
            findings.append(f"**Lower Facial Height**: {v}% (\u2193 below 55% norm) \u2014 **skeletal deep bite**.")

    # --- Skeletal vs. Dental Correction Distinction ---
    # Per textbooks: if ANB abnormal but incisors normal → skeletal issue
    # If ANB normal but interincisal angle acute/obtuse → dentoalveolar issue
    anb = _get(measurements, "ANB")
    ia_m = _get(measurements, "Interincisal Angle")
    anb_abnormal = anb and anb["value"] is not None and anb["status"] != "normal"
    ia_abnormal = ia_m and ia_m["value"] is not None and ia_m["status"] != "normal"
    ia_normal = ia_m and ia_m["value"] is not None and ia_m["status"] == "normal"
    anb_normal = anb and anb["value"] is not None and anb["status"] == "normal"

    if anb_abnormal and ia_normal:
        findings.append(
            "**Correction Type**: The skeletal discrepancy (abnormal ANB) with normal incisor "
            "inclination indicates the issue is **primarily skeletal** in origin, requiring "
            "skeletal correction rather than dental compensation."
        )
    elif anb_normal and ia_abnormal:
        findings.append(
            "**Correction Type**: The normal skeletal relationship (ANB within limits) with "
            "abnormal incisor inclination indicates the issue is **dentoalveolar** in origin, "
            "requiring dental correction rather than skeletal intervention."
        )
    elif anb_abnormal and ia_abnormal:
        findings.append(
            "**Correction Type**: Both skeletal (abnormal ANB) and dental (abnormal Interincisal Angle) "
            "components are present, requiring a **combined skeletal and dental** treatment approach."
        )

    # --- Cross-Analysis Validation (Steiner SNA/SNB vs Ricketts Facial Depth) ---
    # Detect physically impossible contradictions that indicate landmark detection errors
    snb = _get(measurements, "SNB")
    fda = _get(measurements, "Facial Depth Angle")
    if snb and fda and snb["value"] is not None and fda["value"] is not None:
        snb_s = snb["status"]
        fda_s = fda["status"]
        if snb_s == "low" and fda_s == "high":
            # CONTRADICTION: SNB says recessive, Facial Depth says forward — impossible
            findings.append(
                "**⚠ Cross-Analysis Contradiction**: Steiner SNB ({snb_v}°) indicates a recessive mandible, "
                "but Ricketts Facial Depth ({fda_v}°) indicates a forward chin position. "
                "It is **anatomically impossible** for the mandible to be simultaneously retrusive and protrusive. "
                "This contradiction strongly suggests a **landmark detection error** — "
                "verify the placement of Sella (S), Nasion (N), Pogonion (Pog), and Porion/Orbitale (FH plane). "
                "Clinical findings from these two measurements should NOT be relied upon until re-verified.".format(
                    snb_v=snb["value"], fda_v=fda["value"]
                )
            )
        elif snb_s == "high" and fda_s == "low":
            # CONTRADICTION: SNB says prognathic, Facial Depth says retrusive — impossible
            findings.append(
                "**⚠ Cross-Analysis Contradiction**: Steiner SNB ({snb_v}°) indicates a prognathic mandible, "
                "but Ricketts Facial Depth ({fda_v}°) indicates a retrusive chin position. "
                "This contradiction strongly suggests a **landmark detection error** — "
                "verify the placement of Sella (S), Nasion (N), Pogonion (Pog), and Porion/Orbitale (FH plane). "
                "Clinical findings from these two measurements should NOT be relied upon until re-verified.".format(
                    snb_v=snb["value"], fda_v=fda["value"]
                )
            )
        elif snb_s == "low" and fda_s == "low":
            findings.append("**Cross-Analysis**: SNB + Facial Depth both confirm **mandibular retrusion**.")
        elif snb_s == "high" and fda_s == "high":
            findings.append("**Cross-Analysis**: SNB + Facial Depth both confirm **mandibular prognathism**.")
        elif snb_s == "low" and fda_s == "normal":
            findings.append("**Cross-Analysis Note**: SNB indicates retrusion but Facial Depth is normal (different reference planes).")
        elif snb_s == "normal" and fda_s == "low":
            findings.append("**Cross-Analysis Note**: Facial Depth suggests retrusive chin but SNB is normal (different reference planes).")

    # --- Steiner Acceptable Compromise ("Chevrons") ---
    # Per Steiner textbook: when ANB deviates from 2°, the "ideal" lower incisor
    # position shifts. The Chevron table defines acceptable 1-to-NB positions:
    #   ANB -2° → 1-NB 5.5mm / 31°   (more proclined acceptable)
    #   ANB  0° → 1-NB 4.75mm / 28°
    #   ANB  2° → 1-NB 4.0mm / 25°   (ideal)
    #   ANB  4° → 1-NB 3.25mm / 22°
    #   ANB  6° → 1-NB 2.5mm / 19°   (more upright acceptable)
    # Since 1-to-NB is not computed, we apply the principle to the interincisal
    # angle: as ANB increases (Class II), incisors naturally compensate with
    # proclination, making an acute IA an acceptable compromise.
    if anb and anb["value"] is not None and ia_m and ia_m["value"] is not None:
        anb_v = anb["value"]
        ia_v = ia_m["value"]
        # Per Chevron logic: each 2° ANB deviation shifts ideal IA by ~3-4°
        # Interpolate: ideal IA ≈ 131° - (ANB - 2°) × 2° (approximate)
        adjusted_ia_norm = 131 - (anb_v - 2) * 2
        if anb_v >= 5 and ia_v < 131 and ia_v >= adjusted_ia_norm - 5:
            findings.append(
                f"**Steiner Chevrons**: ANB {anb_v}\u00b0 (Class II) \u2014 "
                f"interincisal angle {ia_v}\u00b0 may be **acceptable dental compensation** "
                f"(Chevron target: ~{max(2.5, 4.0 - (anb_v - 2) * 0.375):.1f}mm / "
                f"~{max(19, 25 - (anb_v - 2) * 1.5):.0f}\u00b0 from NB)."
            )
        elif anb_v <= -1 and ia_v > 131 and ia_v <= adjusted_ia_norm + 5:
            findings.append(
                f"**Steiner Chevrons**: ANB {anb_v}\u00b0 (Class III) \u2014 "
                f"interincisal angle {ia_v}\u00b0 may be **acceptable dental compensation** "
                f"(Chevron target: ~{min(5.5, 4.0 + abs(anb_v - 2) * 0.375):.1f}mm / "
                f"~{min(31, 25 + abs(anb_v - 2) * 1.5):.0f}\u00b0 from NB)."
            )

    # --- Extreme Value Detection (Landmark Error Flagging) ---
    # Physiologically implausible values suggest landmark detection errors
    _extreme_checks = [
        ("SNA", 65, 95, "SNA", "Sella, Nasion, or Point A"),
        ("SNB", 65, 92, "SNB", "Sella, Nasion, or Point B"),
        ("ANB", -10, 15, "ANB", "Points A and B"),
        ("Mandibular Plane", 15, 55, "GoGn-SN", "Gonion, Gnathion, Sella, or Nasion"),
        ("Mandibular Plane Angle", 10, 50, "FH-MP", "Porion, Orbitale, Gonion, or Menton"),
        ("Interincisal Angle", 90, 170, "Interincisal Angle", "upper and lower incisor landmarks"),
        ("Facial Depth Angle", 75, 105, "Facial Depth", "Nasion, Pogonion, Porion, or Orbitale"),
    ]
    extreme_flags = []
    for name, lo, hi, label, lm_names in _extreme_checks:
        m = _get(measurements, name)
        if m and m["value"] is not None:
            if m["value"] < lo or m["value"] > hi:
                extreme_flags.append(f"{label} = {m['value']}° (physiological range {lo}°-{hi}°; check {lm_names})")
    if extreme_flags:
        findings.append(
            "**⚠ Extreme Value Alert**: The following measurements fall outside physiologically "
            "plausible ranges, strongly suggesting **landmark detection errors**:\n"
            + "\n".join(f"  - {f}" for f in extreme_flags)
        )

    # --- Profile Consistency Check (ANB Class vs Facial Convexity) ---
    # Per Ricketts: Convexity of Point A should agree with ANB classification
    # Positive convexity = Class II; negative = Class III; near zero = Class I
    facial_conv = _get(measurements, "Facial Convexity")
    if anb and anb["value"] is not None and facial_conv and facial_conv["value"] is not None:
        anb_v = anb["value"]
        fc_v = facial_conv["value"]
        fc_s = facial_conv["status"]
        # ANB says Class II but convexity is low/negative (Class III profile)
        if anb_v > 4 and fc_s == "low":
            findings.append(
                f"**⚠ Profile Inconsistency**: ANB of {anb_v}° indicates Class II, but "
                f"Facial Convexity of {fc_v}mm suggests a Class III profile. "
                f"This contradicts expected skeletal relationships and may indicate "
                f"a landmark detection error at Point A, Nasion, or Pogonion."
            )
        # ANB says Class III but convexity is high (Class II profile)
        elif anb_v < 0 and fc_s == "high":
            findings.append(
                f"**⚠ Profile Inconsistency**: ANB of {anb_v}° indicates Class III, but "
                f"Facial Convexity of {fc_v}mm suggests a Class II profile. "
                f"This contradicts expected skeletal relationships and may indicate "
                f"a landmark detection error at Point A, Nasion, or Pogonion."
            )

    # --- ANB vs McNamara Proportional Consistency ---
    # Per McNamara philosophy: "A mandible is not 'short' just because it's less
    # than 120mm; it's only short if the differential is less than expected."
    # If ANB says Class II (mandibular retrusion) but McNamara says mandible is
    # proportionally long (or vice versa), this is a major inconsistency.
    if anb and anb["value"] is not None and max_len and mand_len:
        if max_len["value"] and mand_len["value"]:
            co_a = max_len["value"]
            co_gn = mand_len["value"]
            # Re-run quick proportional check
            _tbl = [
                (85, 105, 108), (87, 108, 110), (89, 111, 113), (91, 114, 116),
                (92, 117, 120), (94, 121, 124), (97, 127, 130), (100, 130, 133),
            ]
            if co_a <= _tbl[0][0]:
                _ilo, _ihi = _tbl[0][1], _tbl[0][2]
            elif co_a >= _tbl[-1][0]:
                _ilo, _ihi = _tbl[-1][1], _tbl[-1][2]
            else:
                for i in range(len(_tbl) - 1):
                    if _tbl[i][0] <= co_a <= _tbl[i + 1][0]:
                        t = (co_a - _tbl[i][0]) / (_tbl[i + 1][0] - _tbl[i][0])
                        _ilo = round(_tbl[i][1] + t * (_tbl[i + 1][1] - _tbl[i][1]))
                        _ihi = round(_tbl[i][2] + t * (_tbl[i + 1][2] - _tbl[i][2]))
                        break

            anb_v = anb["value"]
            mand_long = co_gn > _ihi + 2
            mand_short = co_gn < _ilo - 2

            # Class II (ANB > 4°) + long mandible = inconsistent
            if anb_v > 4 and mand_long:
                findings.append(
                    f"**⚠ Skeletal–Proportional Inconsistency**: ANB {anb_v}\u00b0 suggests Class II, "
                    f"but Co-Gn {co_gn:.1f}mm is **long** relative to Co-A (expected {_ilo}\u2013{_ihi}mm). "
                    f"Class II may reflect **cranial base geometry** rather than true mandibular deficiency."
                )
            # Class III (ANB < 0°) + short mandible = inconsistent
            elif anb_v < 0 and mand_short:
                findings.append(
                    f"**⚠ Skeletal–Proportional Inconsistency**: ANB {anb_v}\u00b0 suggests Class III, "
                    f"but Co-Gn {co_gn:.1f}mm is **short** relative to Co-A (expected {_ilo}\u2013{_ihi}mm). "
                    f"Class III may reflect **cranial base geometry** rather than true mandibular prognathism."
                )

    # --- Ricketts Age-Adjusted Multi-Age Interpretation ---
    # Per Ricketts: norms are age-dependent. Without patient age we cannot pick
    # a single baseline, so we show the patient value vs ALL age checkpoints.
    _ricketts_map = [
        ("Mandibular Plane Angle", "Mandibular Plane Angle (FH-GoGn)"),
        ("Facial Depth Angle",     "Facial Depth Angle (N-Pog to FH)"),
        ("Facial Convexity",       "Facial Convexity (A to N-Pog)"),
    ]
    _age_sections = []
    for mname, mlabel in _ricketts_map:
        m = _get(measurements, mname)
        if m and m["value"] is not None:
            table, closest = _ricketts_age_table(mname, m["value"])
            if table:
                _age_sections.append(
                    f"**{mlabel}: {m['value']}{m['units']}**\n\n{table}"
                )
    if _age_sections:
        findings.append(
            "**Ricketts Age-Adjusted Norms** "
            "(no patient age supplied — all age baselines shown):\n\n"
            + "\n\n".join(_age_sections)
        )

    # --- Catch any remaining abnormal measurements ---
    covered = {
        "ANB", "SNA", "SNB", "Mandibular Plane", "Mandibular Plane Angle",
        "Facial Convexity", "Maxillary Length", "Mandibular Length",
        "Anterior Facial Height", "Interincisal Angle", "Facial Depth Angle",
        "Lower Facial Height %",
    }
    for m in measurements:
        if m["name"] not in covered and m["status"] != "normal" and m["value"] is not None:
            word = "above" if m["status"] == "high" else "below"
            findings.append(f"**{m['name']}**: {m['value']}{m['units']} is {word} normal range ({m['normal']}).")

    if not findings:
        findings.append("All measurements are within normal ranges.")

    return "### Key Findings\n\n" + "\n\n".join(findings)


# ---------------------------------------------------------------------------
# Rule-based: Clinical Summary (deterministic — no LLM)
# ---------------------------------------------------------------------------
def _build_clinical_summary(measurements):
    """Build a deterministic clinical summary from measurement values."""
    sentences = []

    anb = _get(measurements, "ANB")
    sna = _get(measurements, "SNA")
    snb = _get(measurements, "SNB")
    steiner_mp = _get(measurements, "Mandibular Plane")
    ricketts_mp = _get(measurements, "Mandibular Plane Angle")
    ia = _get(measurements, "Interincisal Angle")
    max_len = _get(measurements, "Maxillary Length")
    mand_len = _get(measurements, "Mandibular Length")
    facial_conv = _get(measurements, "Facial Convexity")
    fda = _get(measurements, "Facial Depth Angle")
    lfh = _get(measurements, "Lower Facial Height %")
    afh = _get(measurements, "Anterior Facial Height")

    # --- Sentence 1: Skeletal classification + source ---
    if anb and anb["value"] is not None:
        v = anb["value"]
        if v > 4:
            skel = f"The cephalometric analysis reveals a Skeletal Class II relationship (ANB {v}°, Steiner norm 2°)"
        elif v > 2:
            skel = f"The cephalometric analysis reveals a mild Skeletal Class II tendency (ANB {v}°, Steiner norm 2°)"
        elif v < 0:
            skel = f"The cephalometric analysis reveals a Skeletal Class III relationship (ANB {v}°, Steiner norm 2°)"
        elif v < 2:
            skel = f"The cephalometric analysis reveals a mild Skeletal Class III tendency (ANB {v}°, Steiner norm 2°)"
        else:
            skel = f"The cephalometric analysis reveals a Skeletal Class I relationship (ANB {v}°, Steiner norm 2°)"

        # Add source of discrepancy
        sna_s = sna["status"] if sna else None
        snb_s = snb["status"] if snb else None
        sna_v = sna["value"] if sna else None
        snb_v = snb["value"] if snb else None

        if v > 2:
            if snb_s == "low" and sna_s in ("normal", None):
                skel += f", primarily attributed to mandibular retrusion (SNB {snb_v}°, below the norm of 80°) with a normally positioned maxilla (SNA {sna_v}°)"
            elif sna_s == "high" and snb_s in ("normal", None):
                skel += f", primarily attributed to maxillary protrusion (SNA {sna_v}°, above the norm of 82°) with a normally positioned mandible (SNB {snb_v}°)"
            elif sna_s == "high" and snb_s == "low":
                skel += f", involving both maxillary protrusion (SNA {sna_v}°) and mandibular retrusion (SNB {snb_v}°)"
            elif sna_s == "normal" and snb_s == "normal":
                skel += f" with both SNA ({sna_v}°) and SNB ({snb_v}°) within normal limits, suggesting a dentoalveolar rather than skeletal origin"
        elif v < 0:
            if snb_s == "high" and sna_s in ("normal", None):
                skel += f", primarily attributed to mandibular prognathism (SNB {snb_v}°) with a normally positioned maxilla (SNA {sna_v}°)"
            elif sna_s == "low" and snb_s in ("normal", None):
                skel += f", primarily attributed to maxillary retrusion (SNA {sna_v}°) with a normally positioned mandible (SNB {snb_v}°)"
        else:
            # Class I — describe jaw positions
            parts = []
            if sna and sna_v is not None:
                if sna_s == "normal":
                    parts.append(f"normally positioned maxilla (SNA {sna_v}°)")
                elif sna_s == "high":
                    parts.append(f"slightly protrusive maxilla (SNA {sna_v}°)")
                else:
                    parts.append(f"slightly retrusive maxilla (SNA {sna_v}°)")
            if snb and snb_v is not None:
                if snb_s == "normal":
                    parts.append(f"normally positioned mandible (SNB {snb_v}°)")
                elif snb_s == "high":
                    parts.append(f"slightly prognathic mandible (SNB {snb_v}°)")
                else:
                    parts.append(f"slightly retrusive mandible (SNB {snb_v}°)")
            if parts:
                skel += f" with {' and '.join(parts)}"

        sentences.append(skel + ".")

    # --- Sentence 2: Growth pattern (vertical category — separate from AP) ---
    patterns = []
    if steiner_mp and steiner_mp["value"] is not None:
        patterns.append(("steiner", steiner_mp["status"], steiner_mp["value"]))
    if ricketts_mp and ricketts_mp["value"] is not None:
        patterns.append(("ricketts", ricketts_mp["status"], ricketts_mp["value"]))

    if patterns:
        statuses = [p[1] for p in patterns]
        if all(s == "low" for s in statuses):
            gp = "The vertical growth pattern is hypodivergent (horizontal), indicating a short face tendency"
            if lfh and lfh["value"] is not None and lfh["status"] == "low":
                gp += " consistent with the decreased lower facial height"
            sentences.append(gp + ".")
        elif all(s == "high" for s in statuses):
            gp = "The vertical growth pattern is hyperdivergent, indicating a long face tendency"
            if lfh and lfh["value"] is not None and lfh["status"] == "high":
                gp += " consistent with the increased lower facial height"
            sentences.append(gp + ".")
        elif all(s == "normal" for s in statuses):
            sentences.append("The vertical growth pattern is within normal limits.")
        else:
            sentences.append(
                "The growth pattern indicators show mixed results between Steiner and Ricketts analyses, "
                "attributable to different reference planes (SN plane vs. Frankfort Horizontal)."
            )

    # --- Sentence 3: Proportional analysis (jaw sizes — separate category) ---
    if max_len and mand_len and max_len["value"] and mand_len["value"]:
        co_a = max_len["value"]
        co_gn = mand_len["value"]

        # Re-run interpolation to get verdict
        _table = [
            (85, 105, 108), (87, 108, 110), (89, 111, 113), (91, 114, 116),
            (92, 117, 120), (94, 121, 124), (97, 127, 130), (100, 130, 133),
        ]
        if co_a <= _table[0][0]:
            ideal_lo, ideal_hi = _table[0][1], _table[0][2]
        elif co_a >= _table[-1][0]:
            ideal_lo, ideal_hi = _table[-1][1], _table[-1][2]
        else:
            for i in range(len(_table) - 1):
                if _table[i][0] <= co_a <= _table[i + 1][0]:
                    t = (co_a - _table[i][0]) / (_table[i + 1][0] - _table[i][0])
                    ideal_lo = round(_table[i][1] + t * (_table[i + 1][1] - _table[i][1]))
                    ideal_hi = round(_table[i][2] + t * (_table[i + 1][2] - _table[i][2]))
                    break

        if ideal_lo - 2 <= co_gn <= ideal_hi + 2:
            sentences.append(
                f"McNamara proportional analysis confirms that the effective jaw lengths are well-matched "
                f"(Co-A {co_a:.1f}mm, Co-Gn {co_gn:.1f}mm), falling within the expected range."
            )
        elif co_gn < ideal_lo - 2:
            sentences.append(
                f"McNamara proportional analysis indicates the mandible is short relative to the maxilla "
                f"(Co-A {co_a:.1f}mm, Co-Gn {co_gn:.1f}mm; expected {ideal_lo}-{ideal_hi}mm)."
            )
        else:
            sentences.append(
                f"McNamara proportional analysis indicates the mandible is long relative to the maxilla "
                f"(Co-A {co_a:.1f}mm, Co-Gn {co_gn:.1f}mm; expected {ideal_lo}-{ideal_hi}mm)."
            )

    # --- Sentence 4: Dental findings (separate category) ---
    if ia and ia["value"] is not None:
        v = ia["value"]
        if ia["status"] == "low":
            sentences.append(
                f"Dentally, the acute interincisal angle of {v}° (Steiner norm 131°) indicates "
                f"proclined incisors that may benefit from uprighting."
            )
        elif ia["status"] == "high":
            sentences.append(
                f"Dentally, the obtuse interincisal angle of {v}° (Steiner norm 131°) indicates "
                f"retroclined incisors that may benefit from advancement or correction of axial inclination."
            )
        else:
            sentences.append(
                f"The interincisal angle of {v}° is within normal limits, suggesting acceptable incisor inclination."
            )

    # --- Sentence 5: Skeletal vs Dental distinction ---
    anb_abnormal = anb and anb["value"] is not None and anb["status"] != "normal"
    ia_abnormal = ia and ia["value"] is not None and ia["status"] != "normal"
    ia_normal = ia and ia["value"] is not None and ia["status"] == "normal"
    anb_normal = anb and anb["value"] is not None and anb["status"] == "normal"

    if anb_abnormal and ia_normal:
        sentences.append(
            "The discrepancy is primarily skeletal in origin, as the incisor inclination "
            "is within normal limits despite the abnormal jaw relationship."
        )
    elif anb_normal and ia_abnormal:
        sentences.append(
            "The issue is dentoalveolar in origin, as the skeletal jaw relationship is "
            "normal but the incisor inclination is abnormal."
        )
    elif anb_abnormal and ia_abnormal:
        sentences.append(
            "Both skeletal and dentoalveolar components contribute to the malocclusion."
        )

    # --- Detect cross-analysis contradictions for treatment logic ---
    has_landmark_error = False
    if snb and fda and snb["value"] is not None and fda["value"] is not None:
        snb_s_tx = snb["status"]
        fda_s_tx = fda["status"]
        if (snb_s_tx == "low" and fda_s_tx == "high") or (snb_s_tx == "high" and fda_s_tx == "low"):
            has_landmark_error = True

    # Determine McNamara proportional verdict for treatment cross-reference
    mand_proportional = None  # None = no data, "short", "proportional", "long"
    if max_len and mand_len and max_len["value"] and mand_len["value"]:
        co_a = max_len["value"]
        co_gn = mand_len["value"]
        _table = [
            (85, 105, 108), (87, 108, 110), (89, 111, 113), (91, 114, 116),
            (92, 117, 120), (94, 121, 124), (97, 127, 130), (100, 130, 133),
        ]
        if co_a <= _table[0][0]:
            _lo, _hi = _table[0][1], _table[0][2]
        elif co_a >= _table[-1][0]:
            _lo, _hi = _table[-1][1], _table[-1][2]
        else:
            for i in range(len(_table) - 1):
                if _table[i][0] <= co_a <= _table[i + 1][0]:
                    t = (co_a - _table[i][0]) / (_table[i + 1][0] - _table[i][0])
                    _lo = round(_table[i][1] + t * (_table[i + 1][1] - _table[i][1]))
                    _hi = round(_table[i][2] + t * (_table[i + 1][2] - _table[i][2]))
                    break
        if _lo - 2 <= co_gn <= _hi + 2:
            mand_proportional = "proportional"
        elif co_gn < _lo - 2:
            mand_proportional = "short"
        else:
            mand_proportional = "long"

    # --- Sentence 6: Treatment considerations (cross-referenced) ---
    if has_landmark_error:
        sentences.append(
            "Due to contradictory cross-analysis findings (Steiner SNB vs. Ricketts Facial Depth), "
            "treatment recommendations cannot be reliably made until landmark placement is verified."
        )
    else:
        tx = []
        if anb and anb["value"] is not None and anb["value"] > 2:
            snb_s = snb["status"] if snb else None
            sna_s = sna["status"] if sna else None
            if snb_s == "low" and mand_proportional == "long":
                # CONTRADICTION: SNB says retrusive but McNamara says long
                tx.append(
                    "clarifying the mandibular position discrepancy (Steiner SNB suggests retrusion, "
                    "but McNamara proportional analysis shows a long mandible relative to the maxilla)"
                )
            elif snb_s == "low" and mand_proportional != "long":
                tx.append("addressing the mandibular retrusion")
            if sna_s == "high":
                tx.append("managing maxillary excess")
            if not tx:
                tx.append("addressing the skeletal discrepancy")
        elif anb and anb["value"] is not None and anb["value"] < 0:
            tx.append("addressing the Class III skeletal discrepancy")

        if ia and ia["value"] is not None and ia["status"] == "low":
            tx.append("uprighting the proclined incisors")
        elif ia and ia["value"] is not None and ia["status"] == "high":
            tx.append("advancing the retroclined incisors")

        if tx:
            sentences.append(f"Treatment considerations include {', and '.join(tx)}.")

    if not sentences:
        sentences.append("All cephalometric measurements are within normal limits with no significant skeletal or dental discrepancies identified.")

    return "\n\n".join(sentences)


# ---------------------------------------------------------------------------
# LLM: Clinical Summary via gemma2:9b
# ---------------------------------------------------------------------------
# System prompt is baked into the ceph-summary Ollama Modelfile (rag/Modelfile).
# This eliminates ~500 tokens of system prompt overhead per request.


def _generate_llm_summary(deterministic_summary: str, confidence_level: str = "HIGH") -> str:
    """Use ceph-summary (custom Ollama model) to polish the deterministic summary.

    System prompt with all textbook norms and rules is baked into the model
    via the Modelfile, saving ~500 tokens of overhead per request.
    Conservatism of language is adjusted based on report confidence level.
    """
    if confidence_level in ("LOW", "NEEDS_REVIEW"):
        conservatism_note = (
            "\nIMPORTANT: This report has LOW confidence due to contradictory landmark-derived "
            "measurements. Use hedged language throughout: prefer 'suggests', 'may indicate', "
            "and 'requires verification' over definitive statements. Do NOT present any "
            "finding as a confirmed diagnosis. Explicitly note that findings should be "
            "manually verified before clinical use.\n"
        )
    elif confidence_level == "MODERATE":
        conservatism_note = (
            "\nNote: This report has MODERATE confidence. Prefer 'suggests' and 'may indicate' "
            "over definitive statements for findings involving Ricketts age-adjusted norms "
            "or measurements derived from potentially suspect landmarks.\n"
        )
    else:
        conservatism_note = ""

    user_msg = (
        "Here is the VERIFIED FACT SHEET from the cephalometric analysis. "
        "Rewrite it as a polished clinical narrative paragraph. "
        "Do NOT change any clinical conclusions — only improve the prose flow. "
        "CRITICAL: Copy all numeric values EXACTLY as written. "
        f"Do NOT substitute one measurement's value for another.{conservatism_note}\n\n"
        f"FACT SHEET:\n{deterministic_summary}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        options={
            "num_gpu": 99,
        },
        keep_alive="10m",
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public API: generate_diagnosis
# ---------------------------------------------------------------------------
def generate_diagnosis(measurements: list[dict], landmark_confidences: dict | None = None) -> str:
    """
    Hybrid diagnosis pipeline:
      1. Confidence assessment (heatmap landmark scores + contradictions + extreme values)
      2. Rule-based: Skeletal Pattern (deterministic)
      3. Rule-based: Key Findings (deterministic)
      4. Rule-based: Deterministic Clinical Summary (fact sheet)
      5. LLM: Polish the fact sheet (conservative language scaled to confidence level)
      6. Append clinical disclaimer
    Falls back to deterministic summary if LLM call fails.
    """
    confidence = _assess_confidence(measurements, landmark_confidences)
    confidence_header = _build_confidence_header(confidence)

    skeletal = _build_skeletal_pattern(measurements)
    findings = _build_key_findings(measurements)

    # Build deterministic fact sheet first (always correct)
    deterministic_summary = _build_clinical_summary(measurements)

    # Try to polish with LLM, passing confidence level for conservatism scaling
    try:
        llm_summary = _generate_llm_summary(deterministic_summary, confidence["level"])
        summary = f"### Clinical Summary\n\n{llm_summary}"
    except Exception:
        summary = f"### Clinical Summary\n\n{deterministic_summary}"

    diagnosis = f"{confidence_header}\n\n{skeletal}\n\n{findings}\n\n{summary}"

    disclaimer = (
        "\n\n---\n\n"
        "> ## ⚠ CLINICAL DISCLAIMER — NOT FOR CLINICAL USE\n"
        ">\n"
        "> This AI-generated cephalometric analysis is **for educational and research "
        "purposes only**. It does **NOT** constitute a professional orthodontic or medical diagnosis.\n"
        ">\n"
        "> - **Automated landmark detection** may contain placement errors — always verify key landmarks.\n"
        "> - **Steiner, Ricketts, and McNamara** analyses use different reference planes; differing results between methods are expected.\n"
        "> - **All findings must be reviewed and confirmed** by a qualified orthodontist or oral/maxillofacial specialist.\n"
        "> - Correlate with direct clinical examination, patient history, diagnostic radiographs, study models, and soft tissue assessment before treatment planning.\n"
        ">\n"
        "> *Do not make clinical decisions based solely on this report.*"
    )

    return diagnosis + disclaimer
