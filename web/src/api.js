const BASE = "";

export async function predictLandmarks(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error("Server unreachable");
  return res.json();
}

export async function generateDiagnosis(measurements) {
  const res = await fetch(`${BASE}/api/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ measurements }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Diagnosis generation failed");
  }
  return res.json();
}
