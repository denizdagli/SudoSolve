/**
 * OCR Processor (v2) — Optimized image recognition.
 * Now primarily handles image capture and sends it to the Python backend for robust processing.
 */

type Grid = number[][];

/**
 * Sends the captured image to the backend API for Sudoku grid recognition.
 * 
 * @param imageSource A blob URL string or an HTMLCanvasElement
 * @returns 9×9 number array (0 = empty cell)
 */
export async function processImage(imageSource: string | HTMLCanvasElement): Promise<Grid> {
  let blob: Blob;

  if (typeof imageSource === 'string') {
    // Convert blob URL to actual Blob
    const response = await fetch(imageSource);
    blob = await response.blob();
  } else {
    // Convert canvas to Blob
    blob = await new Promise<Blob>((resolve, reject) => {
      imageSource.toBlob((b) => {
        if (b) resolve(b);
        else reject(new Error('Canvas conversion failed'));
      }, 'image/jpeg', 0.95);
    });
  }

  const formData = new FormData();
  formData.append('file', blob, 'sudoku.jpg');

  try {
    const response = await fetch('/api/scan', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('API request failed');
    }

    const data = await response.json();
    if (data.status === 'success') {
      return data.grid;
    } else {
      throw new Error(data.message || 'Grid recognition failed');
    }
  } catch (error) {
    console.error('OCR Error:', error);
    // Return empty grid on error
    return Array(9).fill(null).map(() => Array(9).fill(0));
  }
}

/**
 * Capture a square frame from a video element.
 */
export function captureVideoFrame(video: HTMLVideoElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  const size = Math.min(video.videoWidth, video.videoHeight);
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const ox = (video.videoWidth - size) / 2;
  const oy = (video.videoHeight - size) / 2;
  ctx.drawImage(video, ox, oy, size, size, 0, 0, size, size);
  return canvas;
}
