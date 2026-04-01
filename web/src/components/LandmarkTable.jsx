import { LANDMARK_COLORS } from "../constants";

export default function LandmarkTable({ landmarks, highlightedLandmark, onHover }) {
  if (!landmarks || landmarks.length === 0) {
    return (
      <div className="text-sm text-slate-400 text-center py-8">
        No landmarks detected yet.
      </div>
    );
  }

  return (
    <div className="overflow-auto max-h-[420px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-slate-800 z-10">
          <tr className="text-slate-400">
            <th className="text-left py-1.5 px-2 font-medium">#</th>
            <th className="text-left py-1.5 px-2 font-medium">Landmark</th>
            <th className="text-right py-1.5 px-2 font-medium">X</th>
            <th className="text-right py-1.5 px-2 font-medium">Y</th>
          </tr>
        </thead>
        <tbody>
          {landmarks.map((lm, i) => (
            <tr
              key={i}
              className={`border-t border-slate-700/50 cursor-pointer transition-colors
                ${highlightedLandmark === i ? "bg-slate-700/60" : "hover:bg-slate-700/30"}`}
              onMouseEnter={() => onHover?.(i)}
              onMouseLeave={() => onHover?.(null)}
            >
              <td className="py-1 px-2">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: LANDMARK_COLORS[i % LANDMARK_COLORS.length] }}
                />
              </td>
              <td className="py-1 px-2 font-medium text-slate-200">
                <span style={{ color: LANDMARK_COLORS[i % LANDMARK_COLORS.length] }}>
                  {lm.short}
                </span>
                <span className="text-slate-400 ml-1.5">{lm.name}</span>
              </td>
              <td className="py-1 px-2 text-right text-slate-300 tabular-nums">{lm.x.toFixed(1)}</td>
              <td className="py-1 px-2 text-right text-slate-300 tabular-nums">{lm.y.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
