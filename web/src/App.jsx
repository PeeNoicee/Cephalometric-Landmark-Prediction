import { useState, useCallback, useRef } from "react";
import {
  Upload, Search, FileText, Eye, EyeOff, Tag, Tags,
  PenLine, PenOff, Loader2, Server, ChevronDown, Move, RotateCcw, BrainCircuit,
  Menu, X, ZoomIn
} from "lucide-react";
import ImageCanvas from "./components/ImageCanvas";
import AnalysisPanel from "./components/AnalysisPanel";
import { predictLandmarks, generateDiagnosis } from "./api";
import { computeAnalysis } from "./analysis";
import { DEFAULT_PIXEL_SPACING, ANALYSIS_DEFINITIONS } from "./constants";
import { exportPdf } from "./exportPdf";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
  const [diagnosis, setDiagnosis] = useState(null);
  const [diagnosisLoading, setDiagnosisLoading] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [showMobileAnalysis, setShowMobileAnalysis] = useState(false);

  const fileRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setLandmarks(null);
    setError(null);
    setInferenceTime(null);
    setPixelSpacing(DEFAULT_PIXEL_SPACING);
    setDiagnosis(null);
    const url = URL.createObjectURL(f);
    setImageUrl(url);
  }, []);

  const handleUpload = () => {
    fileRef.current?.click();
    setShowMobileMenu(false);
  };

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
    setShowMobileMenu(false);
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

  const handleGenerateDiagnosis = useCallback(async () => {
    if (!landmarks) return;
    setShowMobileMenu(false);
    setDiagnosis(null);
    setDiagnosisLoading(true);
    setError(null);
    try {
      // Compute all measurements across all analysis types
      const allMeasurements = [];
      for (const type of Object.keys(ANALYSIS_DEFINITIONS)) {
        const results = computeAnalysis(type, landmarks, pixelSpacing);
        allMeasurements.push(...results.map((r) => ({ ...r, analysisType: type })));
      }
      // Build landmark confidence map: { short_name -> confidence_score }
      const landmarkConfidences = {};
      landmarks.forEach((lm) => {
        if (lm.confidence !== undefined) {
          landmarkConfidences[lm.short] = lm.confidence;
        }
      });
      const data = await generateDiagnosis(allMeasurements, landmarkConfidences);
      setDiagnosis(data.diagnosis);
    } catch (err) {
      setError(err.message);
    } finally {
      setDiagnosisLoading(false);
    }
  }, [landmarks, pixelSpacing]);

  const handleExportPDF = useCallback(() => {
    if (!landmarks) return;
    const canvasDataURL = canvasRef.current?.getOriginalCanvasDataURL();
    exportPdf({
      canvasDataURL,
      landmarks,
      pixelSpacing,
      fileName: file?.name || "unknown",
      analysisType,
      diagnosis,
    });
  }, [landmarks, pixelSpacing, file, analysisType, diagnosis]);

  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
      {/* File input — positioned off-screen so mobile .click() works */}
      <input
        ref={fileRef}
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm,.dicom"
        style={{ position: "fixed", top: "-1000px", left: "-1000px", opacity: 0, width: "1px", height: "1px" }}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <header className="flex items-center justify-between px-3 py-2 bg-slate-900 border-b border-slate-800 shrink-0 md:px-5 md:py-3">
        <div className="flex items-center gap-2 md:gap-3">
          <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center text-white font-bold text-xs md:text-sm">
            AI
          </div>
          <div>
            <h1 className="text-sm md:text-base font-semibold text-white leading-tight">
              AI-Powered Cephalometric Landmark Detection and Diagnosis
            </h1>
          </div>
        </div>
        
        {/* Mobile menu button */}
        <button
          onClick={() => setShowMobileMenu(!showMobileMenu)}
          className="md:hidden p-2 rounded-lg bg-slate-800 text-slate-300"
        >
          <Menu className="w-5 h-5" />
        </button>
        
        <div className="hidden md:flex items-center gap-2">
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
      <div className="flex flex-1 overflow-hidden relative">
        {/* Mobile overlay for menu */}
        {showMobileMenu && (
          <div 
            className="absolute inset-0 bg-black/50 z-40 md:hidden"
            onClick={() => setShowMobileMenu(false)}
          />
        )}

        {/* Left sidebar - hidden on mobile, slide-in when menu open */}
        <aside className={`absolute md:relative z-50 w-64 h-full shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col transition-transform duration-300 md:translate-x-0 ${
          showMobileMenu ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <div className="flex items-center justify-between p-3 border-b border-slate-800 md:hidden">
            <span className="text-sm font-semibold text-slate-200">Menu</span>
            <button 
              onClick={() => setShowMobileMenu(false)}
              className="p-1 rounded-lg bg-slate-800 text-slate-400"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          {/* Actions */}
          <div className="p-4 space-y-2 border-b border-slate-800">
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
                onClick={handleGenerateDiagnosis}
                disabled={diagnosisLoading}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors cursor-pointer"
              >
                {diagnosisLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <BrainCircuit className="w-4 h-4" />
                )}
                {diagnosisLoading ? "Generating..." : diagnosis ? "Regenerate Diagnosis" : "Generate Diagnosis"}
              </button>
            )}
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
          className="flex-1 relative flex flex-col"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {/* Mobile action bar */}
          <div className="md:hidden flex items-center justify-between p-2 bg-slate-900 border-b border-slate-800 shrink-0">
            <div className="flex items-center gap-1 overflow-x-auto">
              <button
                onClick={handleAnalyze}
                disabled={!file || loading}
                className="flex items-center gap-1 px-3 py-2 rounded-lg bg-teal-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium whitespace-nowrap"
              >
                {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                {loading ? "Analyzing..." : "Analyze"}
              </button>
              <button
                onClick={handleGenerateDiagnosis}
                disabled={!landmarks || diagnosisLoading}
                className="flex items-center gap-1 px-3 py-2 rounded-lg bg-violet-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium whitespace-nowrap"
              >
                {diagnosisLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <BrainCircuit className="w-3 h-3" />}
                {diagnosisLoading ? "Generating..." : "Diagnose"}
              </button>
              <button
                onClick={handleExportPDF}
                disabled={!landmarks}
                className="flex items-center gap-1 px-3 py-2 rounded-lg bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium whitespace-nowrap"
              >
                <FileText className="w-3 h-3" />
                Save PDF
              </button>
            </div>
            <button
              onClick={() => setShowMobileAnalysis(!showMobileAnalysis)}
              className="p-2 rounded-lg bg-slate-800 text-slate-300 ml-2 shrink-0"
            >
              <ChevronDown className={`w-4 h-4 transition-transform ${showMobileAnalysis ? 'rotate-180' : ''}`} />
            </button>
          </div>
          {/* Canvas wrapper — flex-1 min-h-0 keeps it below the action bar */}
          <div className="flex-1 min-h-0 relative">
            {!imageUrl && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-slate-500 z-10 px-6">
                <div className="w-20 h-20 rounded-2xl border-2 border-dashed border-slate-700 flex items-center justify-center">
                  <Upload className="w-8 h-8" />
                </div>
                <div className="text-center">
                  <p className="hidden md:block text-sm font-medium text-slate-400">
                    Drag &amp; drop an X-ray image here
                  </p>
                  <p className="md:hidden text-sm font-medium text-slate-400">
                    Tap &ldquo;Select X-ray&rdquo; above to load an image
                  </p>
                  <p className="text-xs text-slate-600 mt-1">
                    PNG, JPG, BMP, DICOM
                  </p>
                </div>
                <button
                  onClick={handleUpload}
                  className="md:hidden mt-3 flex items-center gap-1.5 px-5 py-2.5 rounded-lg bg-cyan-600 text-white text-sm font-medium"
                >
                  <Upload className="w-4 h-4" /> Select X-ray
                </button>
              </div>
            )}
            {imageUrl && landmarks && (
              <div className="md:hidden absolute bottom-12 left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-black/50 text-white text-xs px-3 py-1.5 rounded-full pointer-events-none z-10 whitespace-nowrap">
                <ZoomIn className="w-3 h-3" />
                Pinch to zoom &middot; Drag to pan &middot; Double-tap to reset
              </div>
            )}
            <ImageCanvas
              ref={canvasRef}
              imageUrl={imageUrl}
              landmarks={landmarks}
              analysisType={analysisType}
              pixelSpacing={pixelSpacing}
              showLandmarks={showLandmarks}
              showLabels={showLabels}
              showTracing={showTracing}
              highlightedLandmark={highlightedLandmark}
              editMode={editMode}
              onLandmarkMove={handleLandmarkMove}
              onHighlight={setHighlightedLandmark}
            />
          </div>
        </main>

        {/* Right panel — Analysis */}
        <aside className={`absolute md:relative right-0 z-30 w-full md:w-80 h-full shrink-0 bg-slate-900 border-l border-slate-800 flex flex-col overflow-hidden transition-transform duration-300 md:translate-x-0 ${
          showMobileAnalysis ? 'translate-x-0' : 'translate-x-full md:translate-x-0'
        }`}>
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">
                {analysisType} Analysis
              </h2>
              {editCount > 0 && (
                <p className="text-xs text-amber-400 mt-0.5">
                  {editCount} landmark{editCount !== 1 ? "s" : ""} edited
                </p>
              )}
            </div>
            <button 
              onClick={() => setShowMobileAnalysis(false)}
              className="p-1 rounded-lg bg-slate-800 text-slate-400 md:hidden"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-auto p-3">
            <AnalysisPanel
              analysisType={analysisType}
              landmarks={landmarks}
              pixelSpacing={pixelSpacing}
            />
            {diagnosis && (
              <div className="mt-4 rounded-xl overflow-hidden border border-violet-500/25 bg-slate-900/60 shadow-lg">
                {/* Header bar */}
                <div className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-violet-600/25 to-transparent border-b border-violet-500/20">
                  <BrainCircuit className="w-4 h-4 text-violet-400 shrink-0" />
                  <span className="text-xs font-semibold text-violet-300 uppercase tracking-widest">AI Diagnosis</span>
                </div>
                {/* Content */}
                <div className="diagnosis-markdown p-4 text-xs text-slate-300 leading-relaxed">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table({ children }) {
                        return (
                          <div className="diagnosis-table-wrap">
                            <table>{children}</table>
                          </div>
                        );
                      },
                      td({ children, ...props }) {
                        const text = String(children ?? "");
                        let cls = "";
                        if (/High/.test(text))        cls = "status-high";
                        else if (/Low/.test(text))    cls = "status-low";
                        else if (/Normal/.test(text)) cls = "status-normal";
                        if (text.includes("←")) cls += " status-closest";
                        return <td {...props} className={cls || undefined}>{children}</td>;
                      },
                    }}
                  >{diagnosis}</ReactMarkdown>
                </div>
              </div>
            )}
            {diagnosisLoading && (
              <div className="mt-4 flex items-center gap-2 text-xs text-violet-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                Generating diagnosis...
              </div>
            )}
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
