'use client';

import { useEffect, useRef } from 'react';
import { containViewport, drawHandOverlay, type OverlayHand } from './hand-overlay';

export type NcmOverlayFrame = {
  ncm_preview_jpeg_base64?: string;
  ncm_frame_id?: number;
  landmarks?: number[][];
  hands?: OverlayHand[];
  runtime_prediction?: string;
};

/** Paint a JPEG and its own detector results in one canvas update. */
export default function NcmHandPreview({ frame }: { frame: NcmOverlayFrame }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    const image = new Image();
    const draw = () => {
      if (cancelled || !image.complete || !image.naturalWidth) return;
      const width = Math.max(1, canvas.clientWidth), height = Math.max(1, canvas.clientHeight);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const context = canvas.getContext('2d');
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      const viewport = containViewport(image.naturalWidth, image.naturalHeight, width, height);
      context.drawImage(image, viewport.left, viewport.top, viewport.width, viewport.height);
      drawHandOverlay(context, viewport, frame.landmarks,
        !!frame.runtime_prediction && frame.runtime_prediction !== 'no_gesture', frame.hands);
      canvas.dataset.frameId = String(frame.ncm_frame_id ?? '');
    };
    image.onload = draw;
    image.onerror = () => {
      if (!cancelled) {
        canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
        delete canvas.dataset.frameId;
      }
    };
    if (frame.ncm_preview_jpeg_base64) image.src = `data:image/jpeg;base64,${frame.ncm_preview_jpeg_base64}`;
    else {
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
      delete canvas.dataset.frameId;
    }
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => {
      cancelled = true;
      observer.disconnect();
      image.onload = null;
      image.onerror = null;
    };
  }, [frame]);
  return <canvas ref={canvasRef} className="camera-video active ncm-camera-video ncm-overlay-preview"
    role="img" aria-label="NCM camera with hand bounding boxes and 21 landmarks per detected hand" />;
}
