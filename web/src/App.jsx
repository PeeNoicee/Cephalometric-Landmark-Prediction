import { useState, useCallback, useRef } from "react";
import {
  Upload, Search, FileText, Eye, EyeOff, Tag, Tags,
  PenLine, PenOff, Loader2, Server, ChevronDown, Move, RotateCcw,
} from "lucide-react";
import ImageCanvas from "./components/ImageCanvas";
import AnalysisPanel from "./components/AnalysisPanel";
import { predictLandmarks } from "./api";
import { DEFAULT_PIXEL_SPACING } from "./constants";
import { exportPdf } from "./exportPdf";
import "./App.css";

const ANALYSIS_TYPES = ["Steiner", "Ricketts", "McNamara"];

function App() {
  const [imageUrl, setImageUrl] = useState(null);
  const [file, setFile] = useState(null);
  const [landmarks, setLandmarks] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [pixelSpacing, setPixelSpacing] = useState(DEFAULT_PIXEL_SPACING);

  const [analysisType, setAnalysisType] = useState("Steiner");
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showTracing, setShowTracing] = useState(true);
  const [highlightedLandmark, setHighlightedLandmark] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [editCount, setEditCount] = useState(0);

  const fileRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setLandmarks(null);
    setError(null);
    setInferenceTime(null);
    setPixelSpacing(DEFAULT_PIXEL_SPACING);
    const url = URL.createObjectURL(f);
    setImageUrl(url);
  }, []);

  const handleUpload = () => fileRef.current?.click();

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictLandmarks(file);
      setLandmarks(data.landmarks);
      setInferenceTime(data.inference_time_seconds);
      if (data.pixel_spacing_mm) setPixelSpacing(data.pixel_spacing_mm);
      if (data.image_base64) {
        setImageUrl(`data:image/png;base64,${data.image_base64}`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const editedSetRef = useRef(new Set());

  const handleLandmarkMove = useCallback((index, newX, newY) => {
    setLandmarks((prev) => {
      if (!prev) return prev;
      return prev.map((lm, i) =>
        i === index ? { ...lm, x: newX, y: newY, _edited: true } : lm
      );
    });
    if (!editedSetRef.current.has(index)) {
      editedSetRef.current.add(index);
      setEditCount(editedSetRef.current.size);
    }
  }, []);

  const handleResetLandmarks = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictLandmarks(file);
      setLandmarks(data.landmarks);
      if (data.image_base64) {
        setImageUrl(`data:image/png;base64,${data.image_base64}`);
      }
      editedSetRef.current.clear();
      setEditCount(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleExportPDF = useCallback(() => {
    if (!landmarks) return;
    const canvasDataURL = canvasRef.current?.getCanvasDataURL();
    exportPdf({
      canvasDataURL,
      landmarks,
      pixelSpacing,
      fileName: file?.name || "unknown",
      analysisType,
    });
  }, [landmarks, pixelSpacing, file, analysisType]);

  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-5 py-3 bg-slate-900 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center text-white font-bold text-sm">
            AI
          </div>
          <div>
            <h1 className="text-base font-semibold text-white leading-tight">
              Cephalometric Landmark Detection
            </h1>
            <p className="text-xs text-slate-400">
              AI-powered cephalometric analysis &middot; 29 landmarks
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {inferenceTime !== null && (
            <span className="text-xs text-slate-500 mr-2">
              <Server className="w-3 h-3 inline mr-1" />
              {inferenceTime.toFixed(2)}s
            </span>
          )}
          {error && (
            <span className="text-xs text-rose-400 mr-2">{error}</span>
          )}
        </div>
      </header>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        <aside className="w-64 shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
          {/* Actions */}
          <div className="p-4 space-y-2 border-b border-slate-800">
            <input
              ref={fileRef}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm,.dicom"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <button
              onClick={handleUpload}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              Select X-ray
            </button>
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-teal-600 hover:bg-teal-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors cursor-pointer"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
            {landmarks && (
              <button
                onClick={handleExportPDF}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium transition-colors cursor-pointer"
              >
                <FileText className="w-4 h-4" />
                Save as PDF
              </button>
            )}
          </div>

          {/* Display toggles */}
          <div className="p-4 space-y-1 border-b border-slate-800">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
              Display
            </p>
            <Toggle
              icon={showLandmarks ? Eye : EyeOff}
              label="Landmarks"
              active={showLandmarks}
              onClick={() => setShowLandmarks(!showLandmarks)}
            />
            <Toggle
              icon={showLabels ? Tag : Tags}
              label="Labels"
              active={showLabels}
              onClick={() => setShowLabels(!showLabels)}
            />
            <Toggle
              icon={showTracing ? PenLine : PenOff}
              label="Tracing Lines"
              active={showTracing}
              onClick={() => setShowTracing(!showTracing)}
            />
          </div>

          {/* Edit landmarks */}
          {landmarks && (
            <div className="p-4 space-y-1 border-b border-slate-800">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
                Edit Landmarks
              </p>
              <Toggle
                icon={Move}
                label="Edit Mode"
                active={editMode}
                onClick={() => setEditMode(!editMode)}
              />
              {editCount > 0 && (
                <button
                  onClick={handleResetLandmarks}
                  disabled={loading}
                  className="w-full flex items-center gap-2.5 px-3 py-1.5 rounded-md text-sm text-amber-400 hover:bg-slate-800/50 transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset ({editCount} edits)
                </button>
              )}
            </div>
          )}

          {/* Analysis type */}
          <div className="p-4 border-b border-slate-800">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
              Analysis Type
            </p>
            <div className="relative">
              <select
                value={analysisType}
                onChange={(e) => setAnalysisType(e.target.value)}
                className="w-full appearance-none bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {ANALYSIS_TYPES.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-2.5 top-2.5 w-4 h-4 text-slate-500 pointer-events-none" />
            </div>
          </div>

          {/* Status */}
          <div className="p-4 mt-auto">
            <div className="text-xs text-slate-500">
              {file ? (
                <span className="text-slate-400">{file.name}</span>
              ) : (
                "No image loaded"
              )}
            </div>
            {landmarks && (
              <div className="text-xs text-teal-400 mt-1">
                {landmarks.length} landmarks detected
              </div>
            )}
          </div>
        </aside>

        {/* Center: Image canvas */}
        <main
          className="flex-1 relative"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {!imageUrl && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-slate-500 z-10">
              <div className="w-20 h-20 rounded-2xl border-2 border-dashed border-slate-700 flex items-center justify-center">
                <Upload className="w-8 h-8" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-slate-400">
                  Drag & drop an X-ray image here
                </p>
                <p className="text-xs text-slate-600 mt-1">
                  Supports PNG, JPG, BMP, DICOM
                </p>
              </div>
            </div>
          )}
          <ImageCanvas
            ref={canvasRef}
            imageUrl={imageUrl}
            landmarks={landmarks}
            analysisType={analysisType}
            showLandmarks={showLandmarks}
            showLabels={showLabels}
            showTracing={showTracing}
            highlightedLandmark={highlightedLandmark}
            editMode={editMode}
            onLandmarkMove={handleLandmarkMove}
            onHighlight={setHighlightedLandmark}
          />
        </main>

        {/* Right panel — Analysis */}
        <aside className="w-80 shrink-0 bg-slate-900 border-l border-slate-800 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 shrink-0">
            <h2 className="text-sm font-semibold text-slate-200">
              {analysisType} Analysis
            </h2>
            {editCount > 0 && (
              <p className="text-xs text-amber-400 mt-0.5">
                {editCount} landmark{editCount !== 1 ? "s" : ""} edited
              </p>
            )}
          </div>
          <div className="flex-1 overflow-auto p-3">
            <AnalysisPanel
              analysisType={analysisType}
              landmarks={landmarks}
              pixelSpacing={pixelSpacing}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

function Toggle({ icon: Icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-md text-sm transition-colors cursor-pointer ${
        active
          ? "text-slate-200 bg-slate-800"
          : "text-slate-500 hover:text-slate-400 hover:bg-slate-800/50"
      }`}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );
}

export default App;
