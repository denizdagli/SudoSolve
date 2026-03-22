import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, X, SwitchCamera, Aperture } from 'lucide-react';

interface CameraInputProps {
  onCapture: (canvas: HTMLCanvasElement) => void;
  onClose: () => void;
}

export default function CameraInput({ onCapture, onClose }: CameraInputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 1280 },
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsReady(true);
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Camera access denied. Please allow camera permissions and try again.');
    }
  }, [facingMode]);

  useEffect(() => {
    startCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, [startCamera]);

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Calculate the square crop area (center of video)
    const size = Math.min(video.videoWidth, video.videoHeight);
    const offsetX = (video.videoWidth - size) / 2;
    const offsetY = (video.videoHeight - size) / 2;

    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext('2d')!;
    
    // Reverse the mirroring applied in the UI so backend/OCR receives correct orientation
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    
    ctx.drawImage(video, offsetX, offsetY, size, size, 0, 0, size, size);

    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }

    onCapture(canvas);
  };

  const toggleCamera = () => {
    setFacingMode(prev => (prev === 'environment' ? 'user' : 'environment'));
  };

  if (error) {
    return (
      <div className="fixed inset-0 z-50 bg-deep-black flex flex-col items-center justify-center p-6">
        <Camera className="w-16 h-16 text-red-400 mb-6" />
        <p className="text-red-400 text-center mb-6 font-mono text-sm">{error}</p>
        <button
          onClick={onClose}
          className="px-8 py-3 bg-surface-high border border-grid-line rounded-xl text-terminal-green font-mono
            hover:bg-terminal-green/10 transition-colors"
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-deep-black flex flex-col">
      {/* Top bar */}
      <div className="flex justify-between items-center p-4 relative z-10">
        <button
          onClick={onClose}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-black/50 backdrop-blur-sm
            border border-white/10 text-white hover:bg-white/10 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
        <span className="font-mono text-xs text-terminal-green/70 tracking-widest uppercase">
          Align Puzzle
        </span>
        <button
          onClick={toggleCamera}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-black/50 backdrop-blur-sm
            border border-white/10 text-white hover:bg-white/10 transition-colors"
        >
          <SwitchCamera className="w-5 h-5" />
        </button>
      </div>

      {/* Camera viewport */}
      <div className="flex-1 relative flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover [transform:scaleX(-1)]"
        />

        {/* Grid overlay */}
        <div className="relative w-[85vw] max-w-[400px] aspect-square z-10">
          {/* Corner brackets */}
          <div className="absolute -inset-1">
            {/* Top-left */}
            <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-terminal-green rounded-tl" />
            {/* Top-right */}
            <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-terminal-green rounded-tr" />
            {/* Bottom-left */}
            <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-terminal-green rounded-bl" />
            {/* Bottom-right */}
            <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-terminal-green rounded-br" />
          </div>

          {/* 9x9 Grid lines */}
          <div className="absolute inset-0 grid grid-cols-9 grid-rows-9">
            {Array.from({ length: 81 }).map((_, i) => {
              const row = Math.floor(i / 9);
              const col = i % 9;
              const isThickRight = (col + 1) % 3 === 0 && col < 8;
              const isThickBottom = (row + 1) % 3 === 0 && row < 8;
              return (
                <div
                  key={i}
                  className={`
                    ${isThickRight ? 'border-r border-r-terminal-green/50' : 'border-r border-r-terminal-green/20'}
                    ${isThickBottom ? 'border-b border-b-terminal-green/50' : 'border-b border-b-terminal-green/20'}
                  `}
                />
              );
            })}
          </div>

          {/* Outer border */}
          <div className="absolute inset-0 border border-terminal-green/60 rounded" />
        </div>

        {/* Dark overlay outside grid */}
        <div className="absolute inset-0 bg-black/40 pointer-events-none" />
      </div>

      {/* Capture button */}
      <div className="p-6 pb-10 flex justify-center relative z-10">
        <button
          onClick={handleCapture}
          disabled={!isReady}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300
            ${isReady
              ? 'bg-terminal-green/20 border-2 border-terminal-green text-terminal-green hover:bg-terminal-green/30 active:scale-90 shadow-[0_0_20px_rgba(0,255,65,0.2)]'
              : 'bg-white/5 border-2 border-white/20 text-white/30'
            }`}
        >
          <Aperture className="w-10 h-10" />
        </button>
      </div>

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
