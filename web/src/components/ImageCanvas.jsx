import { useRef, useEffect, useCallback, useState, forwardRef, useImperativeHandle } from "react";
import { LANDMARK_SHORT, LANDMARK_NAMES, LANDMARK_COLORS, LANDMARK_INDEX, TRACING_SEGMENTS } from "../constants";

const TRACING_LINE_COLORS = [
  "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
  "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
];

const DRAG_THRESHOLD = 15;       // mouse hit radius (px)
const TOUCH_DRAG_THRESHOLD = 32;  // finger hit radius (px) — larger for touch
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.15;

const ImageCanvas = forwardRef(function ImageCanvas({
  imageUrl,
  landmarks,
  analysisType,
  showLandmarks = true,
  showLabels = true,
  showTracing = true,
  highlightedLandmark = null,
  editMode = false,
  onLandmarkMove,
  onHighlight,
}, ref) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  // Expose canvas data URL to parent for PDF export
  useImperativeHandle(ref, () => ({
    getCanvasDataURL: () => canvasRef.current?.toDataURL("image/png"),
    getOriginalCanvasDataURL: () => {
      // Create a temporary canvas to draw the image at original size without zoom/pan
      const canvas = canvasRef.current;
      const img = imgRef.current;
      if (!canvas || !img) return null;

      // Use natural dimensions (original image size, not rendered size)
      const origWidth = img.naturalWidth || img.width;
      const origHeight = img.naturalHeight || img.height;

      const tempCanvas = document.createElement("canvas");
      const tempCtx = tempCanvas.getContext("2d");
      
      // Set canvas to original image dimensions
      tempCanvas.width = origWidth;
      tempCanvas.height = origHeight;

      // Draw the image at original size (no transformations)
      tempCtx.drawImage(img, 0, 0, origWidth, origHeight);

      // Draw landmarks and tracing at original coordinates
      if (landmarks && landmarks.length > 0) {
        // Calculate scale factor for landmarks and labels
        // We want landmarks to be visible regardless of image size
        const scaleFactor = Math.min(origWidth, origHeight) / 1000; // Scale based on image size
        const landmarkRadius = Math.max(3, Math.min(8, 6 * scaleFactor)); // 3-8px radius
        const labelFontSize = Math.max(10, Math.min(16, 12 * scaleFactor)); // 10-16px font
        const lineWidth = Math.max(1, Math.min(3, 2 * scaleFactor)); // 1-3px lines
        
        // Tracing lines
        if (showTracing && analysisType) {
          const segs = TRACING_SEGMENTS[analysisType] || [];
          segs.forEach(([label, from, to], i) => {
            const fromIdx = LANDMARK_INDEX[from];
            const toIdx = LANDMARK_INDEX[to];
            if (fromIdx === undefined || toIdx === undefined) return;
            const a = landmarks[fromIdx];
            const b = landmarks[toIdx];
            if (!a || !b) return;

            tempCtx.beginPath();
            tempCtx.strokeStyle = TRACING_LINE_COLORS[i % TRACING_LINE_COLORS.length];
            tempCtx.lineWidth = lineWidth;
            tempCtx.setLineDash([6 * scaleFactor, 4 * scaleFactor]);
            tempCtx.moveTo(a.x, a.y);
            tempCtx.lineTo(b.x, b.y);
            tempCtx.stroke();
            tempCtx.setLineDash([]);
          });
        }

        // Landmarks
        if (showLandmarks) {
          landmarks.forEach((lm, i) => {
            const color = LANDMARK_COLORS[i % LANDMARK_COLORS.length];
            const isHighlighted = highlightedLandmark === i;
            const radius = isHighlighted ? landmarkRadius * 1.5 : landmarkRadius;

            if (isHighlighted) {
              tempCtx.fillStyle = color + "40"; // 25% opacity
              tempCtx.beginPath();
              tempCtx.arc(lm.x, lm.y, radius + 2 * scaleFactor, 0, 2 * Math.PI);
              tempCtx.fill();
            }

            // Draw landmark point
            tempCtx.fillStyle = color;
            tempCtx.beginPath();
            tempCtx.arc(lm.x, lm.y, radius, 0, 2 * Math.PI);
            tempCtx.fill();
            
            // Add white border for better visibility
            tempCtx.strokeStyle = "#ffffff";
            tempCtx.lineWidth = 1;
            tempCtx.stroke();

            // Labels
            if (showLabels) {
              // Add text shadow/background for better visibility
              const labelText = LANDMARK_SHORT[i] || `P${i + 1}`;
              tempCtx.font = `${labelFontSize}px sans-serif`;
              tempCtx.textAlign = "center";
              tempCtx.textBaseline = "bottom";
              
              // Text background for better visibility
              const metrics = tempCtx.measureText(labelText);
              const textHeight = labelFontSize;
              const padding = 2 * scaleFactor;
              
              tempCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
              tempCtx.fillRect(
                lm.x - metrics.width / 2 - padding,
                lm.y - textHeight - radius - padding * 2,
                metrics.width + padding * 2,
                textHeight + padding * 1.5
              );
              
              // Draw text
              tempCtx.fillStyle = "#ffffff";
              tempCtx.fillText(labelText, lm.x, lm.y - radius - padding);
            }
          });
        }
      }

      return tempCanvas.toDataURL("image/png");
    },
  }));

  // Zoom & pan state (refs to avoid re-render on every frame)
  // panRef stores the absolute top-left position of the image on the canvas
  const zoomRef = useRef(1);
  const panRef = useRef(null); // null = needs centering
  const panningRef = useRef(null); // { startX, startY, startPanX, startPanY }
  const [zoomDisplay, setZoomDisplay] = useState(100);

  const drawParamsRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const draggingRef = useRef(null);
  const lastTapRef = useRef(0);   // timestamp of last tap for double-tap detection
  const drawRef = useRef(null);

  // Reset zoom/pan when image changes
  useEffect(() => {
    zoomRef.current = 1;
    panRef.current = null; // will be centered on next draw
    setZoomDisplay(100);
  }, [imageUrl]);

  // Load image
  useEffect(() => {
    imgRef.current = null;
    setImgLoaded(false);
    if (!imageUrl) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded(true);
      // Force redraw via ref — covers the case where React batches
      // setImgLoaded(false)+setImgLoaded(true) into a no-op
      requestAnimationFrame(() => drawRef.current?.());
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Find nearest landmark within threshold. Pass a larger value for touch input.
  const findNearest = useCallback((canvasX, canvasY, threshold = DRAG_THRESHOLD) => {
    if (!landmarks || landmarks.length === 0) return null;
    const { scale, offsetX, offsetY } = drawParamsRef.current;
    let bestIdx = null;
    let bestDist = Infinity;
    landmarks.forEach((lm, i) => {
      const sx = offsetX + lm.x * scale;
      const sy = offsetY + lm.y * scale;
      const d = Math.hypot(canvasX - sx, canvasY - sy);
      if (d < bestDist && d <= threshold) {
        bestDist = d;
        bestIdx = i;
      }
    });
    return bestIdx;
  }, [landmarks]);

  // Convert canvas coords to image coords
  const canvasToImage = useCallback((canvasX, canvasY) => {
    const { scale, offsetX, offsetY } = drawParamsRef.current;
    if (scale === 0) return null;
    const imgX = (canvasX - offsetX) / scale;
    const imgY = (canvasY - offsetY) / scale;
    return { x: Math.round(imgX * 100) / 100, y: Math.round(imgY * 100) / 100 };
  }, []);

  // Redraw helper — uses a ref so it always calls the latest draw
  const requestDraw = useCallback(() => {
    requestAnimationFrame(() => drawRef.current?.());
  }, []);

  // --- Mouse handlers ---

  const handleMouseDown = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    // Middle-click or right-click → start panning
    if (e.button === 1 || e.button === 2) {
      panningRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startPanX: panRef.current.x,
        startPanY: panRef.current.y,
      };
      canvasRef.current.style.cursor = "grabbing";
      e.preventDefault();
      return;
    }

    // Left-click in edit mode → start landmark drag
    if (e.button === 0 && editMode && landmarks) {
      const idx = findNearest(cx, cy);
      if (idx !== null) {
        draggingRef.current = idx;
        e.preventDefault();
      }
    }
  }, [editMode, landmarks, findNearest]);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    // Panning
    if (panningRef.current) {
      panRef.current = {
        x: panningRef.current.startPanX + (e.clientX - panningRef.current.startX),
        y: panningRef.current.startPanY + (e.clientY - panningRef.current.startY),
      };
      requestDraw();
      return;
    }

    // Landmark dragging
    if (draggingRef.current !== null && editMode && onLandmarkMove) {
      const imgCoords = canvasToImage(cx, cy);
      if (imgCoords) {
        onLandmarkMove(draggingRef.current, imgCoords.x, imgCoords.y);
      }
      return;
    }

    // Hover highlight
    if (landmarks && landmarks.length > 0) {
      const idx = findNearest(cx, cy);
      onHighlight?.(idx);
      canvas.style.cursor = editMode
        ? (idx !== null ? "grab" : "crosshair")
        : (idx !== null ? "pointer" : "default");
    }
  }, [editMode, landmarks, findNearest, canvasToImage, onLandmarkMove, onHighlight, requestDraw]);

  const handleMouseUp = useCallback((e) => {
    if (panningRef.current) {
      panningRef.current = null;
      if (canvasRef.current) canvasRef.current.style.cursor = "";
      return;
    }
    draggingRef.current = null;
  }, []);

  // --- Touch handlers for mobile ---
  const getTouchPos = useCallback((e, touchIndex = 0) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const touch =
      (e.touches && e.touches[touchIndex]) ||
      (e.changedTouches && e.changedTouches[touchIndex]);
    if (!touch) return { x: 0, y: 0 };
    return {
      x: touch.clientX - rect.left,
      y: touch.clientY - rect.top,
    };
  }, []);

  const handleTouchStart = useCallback((e) => {
    e.preventDefault();

    if (e.touches.length === 1) {
      const { x, y } = getTouchPos(e);

      // Double-tap detection → reset zoom
      const now = Date.now();
      if (now - lastTapRef.current < 300) {
        zoomRef.current = 1;
        panRef.current = null;
        setZoomDisplay(100);
        requestDraw();
        lastTapRef.current = 0;
        return;
      }
      lastTapRef.current = now;

      // Edit mode: try to grab a landmark (use larger touch threshold)
      if (editMode && landmarks) {
        const idx = findNearest(x, y, TOUCH_DRAG_THRESHOLD);
        if (idx !== null) {
          draggingRef.current = idx;
          panningRef.current = null; // not panning while dragging
          requestDraw();
          return;
        }
      }

      // Otherwise start panning
      draggingRef.current = null;
      panningRef.current = {
        startX: x,
        startY: y,
        startPanX: panRef.current?.x ?? 0,
        startPanY: panRef.current?.y ?? 0,
      };
    } else if (e.touches.length === 2) {
      // Pinch-zoom start — cancel any active landmark drag / pan
      draggingRef.current = null;
      const t1 = e.touches[0];
      const t2 = e.touches[1];
      const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      // Midpoint of the two fingers (canvas coords)
      const rect = canvasRef.current.getBoundingClientRect();
      const midX = ((t1.clientX + t2.clientX) / 2) - rect.left;
      const midY = ((t1.clientY + t2.clientY) / 2) - rect.top;
      panningRef.current = {
        pinchStartDist: dist,
        pinchStartZoom: zoomRef.current,
        pinchMidX: midX,
        pinchMidY: midY,
        pinchStartPanX: panRef.current?.x ?? 0,
        pinchStartPanY: panRef.current?.y ?? 0,
      };
    }
  }, [editMode, landmarks, findNearest, getTouchPos, requestDraw]);

  const handleTouchMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    e.preventDefault();

    if (e.touches.length === 1) {
      const { x, y } = getTouchPos(e);

      // Landmark drag (edit mode)
      if (draggingRef.current !== null && editMode && onLandmarkMove) {
        const imgCoords = canvasToImage(x, y);
        if (imgCoords) onLandmarkMove(draggingRef.current, imgCoords.x, imgCoords.y);
        requestDraw();
        return;
      }

      // Pan
      if (panningRef.current && !panningRef.current.pinchStartDist) {
        const { startX, startY, startPanX, startPanY } = panningRef.current;
        panRef.current = {
          x: startPanX + (x - startX),
          y: startPanY + (y - startY),
        };
        requestDraw();
      }
    } else if (e.touches.length === 2 && panningRef.current?.pinchStartDist) {
      // Pinch-zoom: scale AND pan simultaneously so the midpoint stays fixed
      const t1 = e.touches[0];
      const t2 = e.touches[1];
      const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      const { pinchStartDist, pinchStartZoom, pinchMidX, pinchMidY,
              pinchStartPanX, pinchStartPanY } = panningRef.current;

      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM,
        pinchStartZoom * (dist / pinchStartDist)));
      const ratio = newZoom / pinchStartZoom;

      // Keep the pinch midpoint fixed on the image
      panRef.current = {
        x: pinchMidX - ratio * (pinchMidX - pinchStartPanX),
        y: pinchMidY - ratio * (pinchMidY - pinchStartPanY),
      };
      zoomRef.current = newZoom;
      setZoomDisplay(Math.round(newZoom * 100));
      requestDraw();
    }
  }, [editMode, getTouchPos, canvasToImage, onLandmarkMove, requestDraw]);

  const handleTouchEnd = useCallback((e) => {
    if (e.touches.length === 0) {
      // All fingers lifted
      draggingRef.current = null;
      panningRef.current = null;
    } else if (e.touches.length === 1 && panningRef.current?.pinchStartDist) {
      // One finger remains after pinch → transition to pan
      const { x, y } = getTouchPos(e);
      panningRef.current = {
        startX: x,
        startY: y,
        startPanX: panRef.current?.x ?? 0,
        startPanY: panRef.current?.y ?? 0,
      };
    }
  }, [getTouchPos]);

  // Zoom with scroll wheel (zoom towards cursor position)
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pan = panRef.current;
    if (!pan) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const oldZoom = zoomRef.current;
    const newZoom = e.deltaY < 0
      ? Math.min(MAX_ZOOM, oldZoom * ZOOM_STEP)
      : Math.max(MIN_ZOOM, oldZoom / ZOOM_STEP);

    // Keep the image point under the cursor fixed:
    // mouseX = pan.x + imgPtX * baseScale * zoom  (for both old and new)
    // => newPan.x = mouseX - (mouseX - pan.x) * newZoom / oldZoom
    const ratio = newZoom / oldZoom;
    panRef.current = {
      x: mouseX - ratio * (mouseX - pan.x),
      y: mouseY - ratio * (mouseY - pan.y),
    };
    zoomRef.current = newZoom;
    setZoomDisplay(Math.round(newZoom * 100));

    requestDraw();
  }, [requestDraw]);

  // Prevent context menu on canvas (right-click is used for pan)
  const handleContextMenu = useCallback((e) => e.preventDefault(), []);

  // Double-click to reset zoom
  const handleDoubleClick = useCallback(() => {
    zoomRef.current = 1;
    panRef.current = null; // will be re-centered on next draw
    setZoomDisplay(100);
    requestDraw();
  }, [requestDraw]);

  // Draw
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imgRef.current;
    if (!canvas || !container || !img) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    canvas.width = cw;
    canvas.height = ch;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, cw, ch);

    const zoom = zoomRef.current;

    // Base scale to fit image in container
    const baseScale = Math.min(cw / img.width, ch / img.height);
    const totalScale = baseScale * zoom;

    const drawW = img.width * totalScale;
    const drawH = img.height * totalScale;

    // If pan is null, center the image (initial state / after reset)
    if (!panRef.current) {
      panRef.current = { x: (cw - drawW) / 2, y: (ch - drawH) / 2 };
    }

    // pan IS the absolute top-left of the image on canvas
    const drawX = panRef.current.x;
    const drawY = panRef.current.y;

    drawParamsRef.current = { scale: totalScale, offsetX: drawX, offsetY: drawY };

    ctx.drawImage(img, drawX, drawY, drawW, drawH);

    if (!landmarks || landmarks.length === 0) return;

    // Tracing lines
    if (showTracing && analysisType) {
      const segs = TRACING_SEGMENTS[analysisType] || [];
      segs.forEach(([label, from, to], i) => {
        const fromIdx = LANDMARK_INDEX[from];
        const toIdx = LANDMARK_INDEX[to];
        if (fromIdx === undefined || toIdx === undefined) return;
        const a = landmarks[fromIdx];
        const b = landmarks[toIdx];
        if (!a || !b) return;

        const ax = drawX + a.x * totalScale;
        const ay = drawY + a.y * totalScale;
        const bx = drawX + b.x * totalScale;
        const by = drawY + b.y * totalScale;

        ctx.beginPath();
        ctx.strokeStyle = TRACING_LINE_COLORS[i % TRACING_LINE_COLORS.length];
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.setLineDash([]);

      });
    }

    // Landmarks
    if (showLandmarks) {
      landmarks.forEach((lm, i) => {
        const sx = drawX + lm.x * totalScale;
        const sy = drawY + lm.y * totalScale;
        const color = LANDMARK_COLORS[i % LANDMARK_COLORS.length];
        const isHighlighted = highlightedLandmark === i;
        const isDragging = draggingRef.current === i;
        const radius = isHighlighted || isDragging ? 6 : 4;

        if (isHighlighted || isDragging) {
          ctx.beginPath();
          ctx.arc(sx, sy, 10, 0, Math.PI * 2);
          ctx.strokeStyle = isDragging ? "#fff" : color;
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.5;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        if (lm._edited) {
          ctx.beginPath();
          ctx.arc(sx, sy, 8, 0, Math.PI * 2);
          ctx.strokeStyle = "#fbbf24";
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Low-confidence warning ring (dashed red)
        if (lm.confidence !== undefined && lm.confidence < 0.40) {
          ctx.beginPath();
          ctx.arc(sx, sy, 9, 0, Math.PI * 2);
          ctx.strokeStyle = "#ef4444";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([3, 2]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        ctx.beginPath();
        ctx.arc(sx, sy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 0.8;
        ctx.stroke();

        if (showLabels) {
          ctx.font = `${isHighlighted || isDragging ? "bold 12px" : "11px"} 'Segoe UI', sans-serif`;
          ctx.fillStyle = lm.confidence !== undefined && lm.confidence < 0.40 ? "#ef4444" : color;
          ctx.strokeStyle = "#000";
          ctx.lineWidth = 2.5;
          ctx.strokeText(LANDMARK_SHORT[i], sx + 6, sy - 6);
          ctx.fillText(LANDMARK_SHORT[i], sx + 6, sy - 6);
        }
      });
    }

    // HUD overlay (top-left)
    if (editMode) {
      ctx.fillStyle = "rgba(251, 191, 36, 0.8)";
      ctx.font = "bold 11px 'Segoe UI', sans-serif";
      ctx.fillText("EDIT MODE", 12, 20);
    }
  }, [imgLoaded, landmarks, analysisType, showLandmarks, showLabels, showTracing, highlightedLandmark, editMode]);

  // Attach wheel + touch events with { passive: false } so preventDefault() works.
  // React JSX event handlers are passive by default in modern browsers, which
  // silently swallows preventDefault() and lets the page scroll/zoom instead.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener("wheel",      handleWheel,      { passive: false });
    canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
    canvas.addEventListener("touchmove",  handleTouchMove,  { passive: false });
    canvas.addEventListener("touchend",   handleTouchEnd,   { passive: false });
    return () => {
      canvas.removeEventListener("wheel",      handleWheel);
      canvas.removeEventListener("touchstart", handleTouchStart);
      canvas.removeEventListener("touchmove",  handleTouchMove);
      canvas.removeEventListener("touchend",   handleTouchEnd);
    };
  }, [handleWheel, handleTouchStart, handleTouchMove, handleTouchEnd]);

  drawRef.current = draw;
  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    const obs = new ResizeObserver(() => draw());
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [draw]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full touch-none"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleContextMenu}
        onDoubleClick={handleDoubleClick}
      />
      {/* Zoom HUD */}
      {imageUrl && zoomDisplay !== 100 && (
        <div className="absolute bottom-3 left-3 bg-black/60 text-white text-xs px-2 py-1 rounded pointer-events-none">
          {zoomDisplay}%
        </div>
      )}
    </div>
  );
});

export default ImageCanvas;
