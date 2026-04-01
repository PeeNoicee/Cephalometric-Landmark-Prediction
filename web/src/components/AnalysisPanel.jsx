import { useMemo } from "react";
import { computeAnalysis } from "../analysis";
import { ArrowDown, ArrowUp, Check } from "lucide-react";

const STATUS_STYLES = {
  normal: { icon: Check, color: "text-emerald-400", bg: "bg-emerald-400/10" },
  low: { icon: ArrowDown, color: "text-amber-400", bg: "bg-amber-400/10" },
  high: { icon: ArrowUp, color: "text-rose-400", bg: "bg-rose-400/10" },
};

export default function AnalysisPanel({ analysisType, landmarks, pixelSpacing }) {
  const results = useMemo(
    () => computeAnalysis(analysisType, landmarks, pixelSpacing),
    [analysisType, landmarks, pixelSpacing]
  );

  if (!landmarks || landmarks.length === 0) {
    return (
      <div className="text-sm text-slate-400 text-center py-8">
        Analyze an image to view cephalometric measurements.
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-sm text-slate-400 text-center py-4">
        No measurements for this analysis type.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {results.map((r) => {
        const style = STATUS_STYLES[r.status] || STATUS_STYLES.normal;
        const Icon = style.icon;
        return (
          <div
            key={r.name}
            className={`flex items-center justify-between px-3 py-2 rounded-lg ${style.bg}`}
          >
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-slate-200 truncate">
                {r.name}
              </div>
              <div className="text-xs text-slate-500">Normal: {r.normal}</div>
            </div>
            <div className="flex items-center gap-2 ml-3">
              <span className={`text-sm font-semibold tabular-nums ${style.color}`}>
                {r.value !== null ? `${r.value}${r.units}` : "—"}
              </span>
              <Icon className={`w-3.5 h-3.5 ${style.color}`} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
