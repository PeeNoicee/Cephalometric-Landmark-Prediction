"""
RAG retriever: query ChromaDB for relevant textbook context, then call
Qwen 2.5 14B via Ollama to generate a clinical diagnosis/remarks.
"""
import os
import chromadb
import ollama

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag", "chroma_db")
COLLECTION_NAME = "cephalometric_textbooks"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:14b-instruct-q4_K_M"

# ---------------------------------------------------------------------------
# ChromaDB client (lazy singleton)
# ---------------------------------------------------------------------------
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_context(query: str, top_k: int = 6) -> list[str]:
    """Retrieve the top-k most relevant textbook chunks for a query."""
    collection = _get_collection()
    resp = ollama.embed(model=EMBED_MODEL, input=query)
    query_emb = resp["embeddings"][0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return results["documents"][0] if results["documents"] else []


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical cephalometric analysis assistant for orthodontic professionals.
Your role is to interpret cephalometric measurements and provide a structured clinical diagnosis.
All rules below are extracted and verified from Steiner, Ricketts, and McNamara textbooks.

========================================================================
1. STEINER ANALYSIS (Source: Steiner)
========================================================================
SNA (maxilla position relative to cranial base): Norm = 82°
  - Above 82° = protrusive/prognathic maxilla
  - Below 82° = recessive/retrognathic maxilla

SNB (mandible position relative to cranial base): Norm = 80°
  - Above 80° = prognathic mandible
  - Below 80° = recessive/retrognathic mandible

ANB (maxilla-mandible relationship): Norm = 2°
  - ANB > 2° = Skeletal Class II (maxillary protrusion or mandibular retrusion)
  - ANB < 2° (including negative) = Skeletal Class III tendency
  - ANB ≈ 2° = Class I
  NOTE: For skeletal classification, use these Steiner thresholds regardless of the [Normal] label.

Mandibular Plane (GoGn-SN): Norm = 32°
  - Above 32° = hyperdivergent (vertical growth, long face)
  - Below 32° = hypodivergent (horizontal growth, short face, deep bite tendency)

Interincisal Angle: Norm = 131° (per Steiner Table 7-1)
  *** CRITICAL — READ CAREFULLY ***
  In cephalometrics, "acute" and "obtuse" are relative to the NORM (131°), NOT to 90°.
  - LESS than 131° (e.g., 113°, 120°, 125°) = "more acute" = PROCLINED incisors (tipped FORWARD)
    -> Treatment: teeth need UPRIGHTING (moving back/straightening)
    -> Even though 113° is mathematically >90°, it is cephalometrically ACUTE because it is BELOW the 131° norm
  - GREATER than 131° (e.g., 140°, 145°) = "more obtuse" = RETROCLINED incisors (tipped BACKWARD)
    -> Treatment: teeth need ADVANCING (moving forward)
  NEVER call an interincisal angle below 131° "obtuse" or "retroclined."
  NEVER recommend "advancement" for an angle below 131° — that would WORSEN protrusion.
  Example: 113.4° is BELOW 131° -> acute -> proclined -> needs uprighting. NOT obtuse, NOT retroclined.

========================================================================
2. RICKETTS ANALYSIS (Source: Ricketts)
========================================================================
NOTE: Ricketts norms are for age 9. Adjust for adult patients as indicated.

Facial Axis (Ba-N to PT-Gn): Norm = 90° ± 3.5° (no age adjustment)

Facial Depth Angle (N-Pog to FH): Norm = 87° ± 3° (adjust +1° every 3 years)
  - This measures chin ANTEROPOSTERIOR position, NOT vertical height
  - Above norm = protrusive/forward-positioned chin
  - Below norm = retrusive/retropositioned chin

Mandibular Plane (MP-FH): Norm = 26° ± 4.5° (adjust -1° every 3 years)
  - Above norm = hyperdivergent (vertical growth)
  - Below norm = hypodivergent (horizontal growth)

Convexity of Point A (A to N-Pog line, perpendicular): Norm = 2 mm ± 2 mm (adjust -1 mm every 3 years)
  - Above norm = convex profile (Class II tendency)
  - Below norm or negative = concave profile (Class III tendency)
  - This is a PERPENDICULAR distance from point A to the N-Pog facial plane

Lower Facial Height Angle (Xi-ANS to Xi-PM): Norm = 45° ± 4° (no age adjustment)

========================================================================
3. McNAMARA ANALYSIS (Source: McNamara)
========================================================================
Point A to N-Perpendicular: 0 mm (mixed dentition); 1 mm (adults)

Mandibular Plane Angle (FH-Go-Me): Norm = 22° ± 4°

Nasolabial Angle: Norm = 102° ± 8°

PROPORTIONAL RELATIONSHIP TABLE (McNamara Table 10-1) — CRITICAL:
Mandibular length (Co-Gn) must be evaluated RELATIVE to midfacial length (Co-A):
  Small:  Co-A = 85 mm  -> Co-Gn = 105–108 mm (diff 20–24 mm), LAFH = 60–62 mm
  Medium: Co-A = 94 mm  -> Co-Gn = 121–124 mm (diff 25–28 mm), LAFH = 65–67 mm
  Large:  Co-A = 100 mm -> Co-Gn = 130–133 mm (diff 29–33 mm), LAFH = 70–73 mm
Use interpolation for values between categories.
If mandibular length is proportional to maxillary length per this table, it is NORMAL — even if it appears outside a generic fixed range.

========================================================================
4. PROFILE & CONSISTENCY RULES — CRITICAL
========================================================================
- Skeletal Class II (ANB > 2°) = CONVEX profile. NEVER call it "straight."
- Skeletal Class III (ANB < 2°, especially negative) = CONCAVE profile.
- Class I (ANB ≈ 2°) with balanced SNA/SNB = straight profile.
- A retrognathic mandible (SNB < 80°) contributes to convexity even if facial convexity mm is small.
- Your profile description MUST match your skeletal classification. Do NOT contradict yourself.

========================================================================
5. GROWTH PATTERN RULES
========================================================================
Hyperdivergent (long face):
  - High Mandibular Plane Angle (Steiner >32°, McNamara >26°)
  - AND/OR negative McNamara Facial Axis
  - Associated with: open bite tendency, increased LAFH

Hypodivergent (short face / deep bite):
  - Low Mandibular Plane Angle
  - AND/OR low Ricketts Lower Facial Height (<41°)
  - Associated with: deep bite tendency, decreased LAFH

Skeletal vs. Dental distinction:
  - If SNA and SNB are both normal but ANB is elevated, the discrepancy may be dentoalveolar protrusion rather than a skeletal problem.

========================================================================
6. CROSS-ANALYSIS RECONCILIATION
========================================================================
Steiner, Ricketts, and McNamara use DIFFERENT reference planes and norms.
They WILL sometimes contradict each other for the same patient. This is expected.

Key differences to be aware of:
  - Steiner uses SN plane; Ricketts/McNamara use Frankfort Horizontal (FH)
  - Mandibular Plane Angle norms differ: Steiner = 32°, Ricketts = 26°, McNamara = 22°
    A value of 27° is "Below normal" per Steiner but "Normal" per Ricketts and "Above normal" per McNamara.
  - Maxillary position: Steiner uses SNA angle, McNamara uses Point A to N-Perpendicular (linear)

How to handle contradictions:
  1. CONSENSUS: When multiple analyses agree on a finding, state it with confidence.
  2. DISAGREEMENT: When analyses contradict, acknowledge both interpretations and explain
     that the difference is due to different reference planes/norms.
     Example: "The mandibular plane angle is below the Steiner norm (32°), suggesting
     hypodivergent growth, though it falls within the normal range per Ricketts (26° ± 4°)."
  3. WEIGHT: For skeletal classification, prioritize Steiner ANB. For jaw size, prioritize
     McNamara's proportional analysis. For growth direction, look for agreement across analyses.
  4. Do NOT present a contradictory finding as a single definitive conclusion without context.

========================================================================
7. OUTPUT GUIDELINES
========================================================================
- STRICTLY respect [Normal], [Above normal], [Below normal] labels on measurements. If labeled [Normal], do NOT call it "slightly below/above" or "borderline."
- Only describe a measurement as abnormal if it is explicitly labeled [Above normal] or [Below normal].
- Exception: For ANB skeletal classification, use the Steiner threshold (2°) even if labeled [Normal].
- NEVER contradict yourself across sections.
- Cross-reference with provided textbook chunks for additional context.
- Use professional orthodontic terminology.
- Be concise and factual. Do not speculate beyond what the data supports.
- Format with ### headings: Skeletal Pattern, Key Findings, Clinical Summary."""


CLINICAL_HINTS = {
    "Interincisal Angle": {
        "low": "PROCLINED incisors (tipped forward/labially). Angle is MORE ACUTE than the 131° norm. Treatment: UPRIGHTING needed. Do NOT call this obtuse or retroclined.",
        "high": "RETROCLINED incisors (tipped backward/lingually). Angle is MORE OBTUSE than the 131° norm. Treatment: ADVANCING needed.",
        "normal": "Within normal range — no significant dental protrusion or retrusion.",
    },
    "Facial Depth Angle": {
        "high": "Forward/protrusive chin position (anteroposterior). NOT related to vertical height.",
        "low": "Retrusive/retropositioned chin (anteroposterior). NOT related to vertical height.",
        "normal": "Chin position is balanced relative to Frankfort Horizontal.",
    },
    "Facial Convexity": {
        "high": "Convex profile — maxilla is forward relative to the N-Pog facial plane.",
        "low": "Concave profile — Class III tendency.",
        "normal": "Balanced profile convexity.",
    },
    "SNA": {
        "high": "Protrusive/prognathic maxilla.",
        "low": "Recessive/retrognathic maxilla.",
        "normal": "Maxilla is normally positioned relative to cranial base.",
    },
    "SNB": {
        "high": "Prognathic mandible.",
        "low": "Recessive/retrognathic mandible.",
        "normal": "Mandible is normally positioned relative to cranial base.",
    },
}


def build_user_prompt(measurements: list[dict], context_chunks: list[str]) -> str:
    """Build the user prompt with measurements and retrieved textbook context."""
    # Format measurements with pre-computed clinical hints
    meas_lines = []
    for m in measurements:
        status_label = {"normal": "Normal", "high": "Above normal", "low": "Below normal"}.get(m["status"], "")
        val_str = f"{m['value']}{m['units']}" if m["value"] is not None else "N/A"
        line = f"  - {m['name']}: {val_str} (Normal: {m['normal']}) [{status_label}]"
        # Attach clinical interpretation hint if available
        hints = CLINICAL_HINTS.get(m["name"], {})
        hint = hints.get(m["status"])
        if hint:
            line += f"\n    -> Clinical meaning: {hint}"
        meas_lines.append(line)

    meas_block = "\n".join(meas_lines)

    # Format context
    ctx_block = "\n\n---\n\n".join(context_chunks)

    return f"""Below are the cephalometric measurements from a patient's lateral cephalogram analysis, followed by relevant textbook references.

## Measurements

{meas_block}

## Textbook References

{ctx_block}

## Task

Based on the measurements above and the textbook references, provide a structured clinical interpretation. Include:
1. **Skeletal Pattern** — Classify the skeletal relationship and growth pattern.
2. **Key Findings** — List the significant findings, noting any measurements outside normal range and their clinical meaning. USE the "Clinical meaning" hints provided with each measurement — do not override them with your own interpretation.
3. **Clinical Summary** — A brief overall assessment with potential treatment considerations.

IMPORTANT REMINDER: For the Interincisal Angle, if the value is BELOW 131°, the teeth are PROCLINED (tipped forward) and need UPRIGHTING. Do NOT call it obtuse or retroclined. The "Clinical meaning" hint attached to the measurement is authoritative — follow it exactly."""


# ---------------------------------------------------------------------------
# Generate diagnosis
# ---------------------------------------------------------------------------
def generate_diagnosis(measurements: list[dict]) -> str:
    """
    Full RAG pipeline:
      1. Build a query from abnormal measurements
      2. Retrieve relevant textbook chunks
      3. Call Qwen 2.5 14B to generate diagnosis
    """
    # Build retrieval query focused on abnormal findings
    abnormal = [m for m in measurements if m["status"] != "normal" and m["value"] is not None]
    normal_ms = [m for m in measurements if m["status"] == "normal" and m["value"] is not None]

    if abnormal:
        query_parts = [f"{m['name']} {m['value']}{m['units']} {'above' if m['status'] == 'high' else 'below'} normal" for m in abnormal]
        query = "Cephalometric analysis interpretation: " + "; ".join(query_parts)
    else:
        query = "Cephalometric analysis interpretation: all measurements within normal range"

    # Retrieve context
    context_chunks = retrieve_context(query, top_k=6)

    # Build prompt
    user_prompt = build_user_prompt(measurements, context_chunks)

    # Call LLM
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0, "seed": 42, "num_predict": 1200},
    )

    diagnosis = response["message"]["content"]

    # Append clinical disclaimer
    disclaimer = (
        "\n\n---\n\n"
        "**⚠ Clinical Disclaimer:** This AI-generated analysis is intended as a "
        "diagnostic aid only. Cephalometric measurements from different analyses "
        "(Steiner, Ricketts, McNamara) may yield differing interpretations due to "
        "different reference planes and norms. All findings should be correlated "
        "with clinical examination, patient history, and soft tissue assessment "
        "before arriving at a definitive diagnosis or treatment plan. "
        "This report does not constitute a professional orthodontic diagnosis."
    )

    return diagnosis + disclaimer
