/**
 * OCR Processor — Advanced image processing pipeline for Sudoku grid recognition.
 *
 * Pipeline:
 *   1. Grayscale conversion + Otsu thresholding (pure B/W)
 *   2. Contour-based grid detection + perspective warp to a flat square
 *   3. Cell segmentation into 81 individual images
 *   4. Per-cell Tesseract OCR with digit-only whitelist
 */

import Tesseract from 'tesseract.js';

type Grid = number[][];
type Point = [number, number]; // [x, y]

// ═══════════════════════════════════════════════════════
//  STEP 1 — Grayscale & Otsu Thresholding
// ═══════════════════════════════════════════════════════

/** Convert an RGBA ImageData to a flat grayscale Uint8Array. */
function toGrayscale(imageData: ImageData): Uint8Array {
  const gray = new Uint8Array(imageData.width * imageData.height);
  const d = imageData.data;
  for (let i = 0; i < gray.length; i++) {
    const j = i * 4;
    gray[i] = Math.round(d[j] * 0.299 + d[j + 1] * 0.587 + d[j + 2] * 0.114);
  }
  return gray;
}

/**
 * Compute the optimal threshold using Otsu's method.
 * Maximises inter-class variance between foreground and background.
 */
function otsuThreshold(gray: Uint8Array): number {
  // Build histogram
  const hist = new Array<number>(256).fill(0);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const total = gray.length;
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];

  let sumBg = 0;
  let weightBg = 0;
  let maxVariance = 0;
  let bestThreshold = 0;

  for (let t = 0; t < 256; t++) {
    weightBg += hist[t];
    if (weightBg === 0) continue;

    const weightFg = total - weightBg;
    if (weightFg === 0) break;

    sumBg += t * hist[t];
    const meanBg = sumBg / weightBg;
    const meanFg = (sumAll - sumBg) / weightFg;

    const variance = weightBg * weightFg * (meanBg - meanFg) * (meanBg - meanFg);

    if (variance > maxVariance) {
      maxVariance = variance;
      bestThreshold = t;
    }
  }

  return bestThreshold;
}

/** Apply Otsu thresholding to a grayscale array → binary (0 or 255). */
function applyOtsu(gray: Uint8Array): Uint8Array {
  const threshold = otsuThreshold(gray);
  const binary = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    binary[i] = gray[i] > threshold ? 255 : 0;
  }
  return binary;
}

/** Write a grayscale/binary array back to canvas as RGBA. */
function grayToCanvas(data: Uint8Array, width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < data.length; i++) {
    const j = i * 4;
    imageData.data[j] = data[i];
    imageData.data[j + 1] = data[i];
    imageData.data[j + 2] = data[i];
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

// ═══════════════════════════════════════════════════════
//  STEP 2 — Grid Detection & Perspective Warp
// ═══════════════════════════════════════════════════════

/**
 * Simple 3×3 Gaussian blur on a grayscale array to reduce noise before
 * edge detection.
 */
function gaussianBlur(src: Uint8Array, w: number, h: number): Uint8Array {
  const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  const kSum = 16;
  const dst = new Uint8Array(src.length);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          sum += src[(y + ky) * w + (x + kx)] * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      dst[y * w + x] = Math.round(sum / kSum);
    }
  }
  return dst;
}

/** Sobel edge detection returning gradient magnitude. */
function sobelEdge(gray: Uint8Array, w: number, h: number): Uint8Array {
  const edges = new Uint8Array(gray.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (off: number) => gray[y * w + x + off];
      const row = (dy: number, dx: number) => gray[(y + dy) * w + (x + dx)];

      const gx =
        -row(-1, -1) + row(-1, 1) +
        -2 * row(0, -1) + 2 * row(0, 1) +
        -row(1, -1) + row(1, 1);

      const gy =
        -row(-1, -1) - 2 * row(-1, 0) - row(-1, 1) +
        row(1, -1) + 2 * row(1, 0) + row(1, 1);

      edges[y * w + x] = Math.min(255, Math.round(Math.sqrt(gx * gx + gy * gy)));
    }
  }
  return edges;
}

/**
 * Dilate a binary image to connect nearby edge fragments.
 * Uses a 3×3 square structuring element.
 */
function dilate(binary: Uint8Array, w: number, h: number, iterations = 2): Uint8Array {
  let src = binary;
  for (let iter = 0; iter < iterations; iter++) {
    const dst = new Uint8Array(src.length);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let max = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const v = src[(y + ky) * w + (x + kx)];
            if (v > max) max = v;
          }
        }
        dst[y * w + x] = max;
      }
    }
    src = dst;
  }
  return src;
}

/**
 * Simple flood-fill based connected-component labelling.
 * Returns array of components, each being a list of pixel indices.
 */
function findConnectedComponents(binary: Uint8Array, w: number, h: number): number[][] {
  const visited = new Uint8Array(binary.length);
  const components: number[][] = [];

  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === 0 || visited[i]) continue;

    // BFS flood fill
    const component: number[] = [];
    const queue: number[] = [i];
    visited[i] = 1;

    while (queue.length > 0) {
      const idx = queue.pop()!;
      component.push(idx);

      const y = Math.floor(idx / w);
      const x = idx % w;

      for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const ny = y + dy;
        const nx = x + dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
          const nIdx = ny * w + nx;
          if (binary[nIdx] > 0 && !visited[nIdx]) {
            visited[nIdx] = 1;
            queue.push(nIdx);
          }
        }
      }
    }

    components.push(component);
  }

  return components;
}

/**
 * Given a list of pixel indices, compute the convex-hull-like bounding
 * quadrilateral by finding the 4 corner points (top-left, top-right,
 * bottom-right, bottom-left).
 */
function findCorners(pixels: number[], w: number): Point[] {
  let minSum = Infinity,
    maxSum = -Infinity;
  let minDiff = Infinity,
    maxDiff = -Infinity;
  let tl: Point = [0, 0],
    br: Point = [0, 0],
    tr: Point = [0, 0],
    bl: Point = [0, 0];

  for (const idx of pixels) {
    const y = Math.floor(idx / w);
    const x = idx % w;
    const sum = x + y;
    const diff = x - y;

    if (sum < minSum) { minSum = sum; tl = [x, y]; }
    if (sum > maxSum) { maxSum = sum; br = [x, y]; }
    if (diff > maxDiff) { maxDiff = diff; tr = [x, y]; }
    if (diff < minDiff) { minDiff = diff; bl = [x, y]; }
  }

  return [tl, tr, br, bl];
}

/**
 * Perspective warp: map a quadrilateral in the source image to a square
 * destination. Uses bilinear interpolation.
 *
 * srcCorners: [topLeft, topRight, bottomRight, bottomLeft]
 */
function perspectiveWarp(
  srcCanvas: HTMLCanvasElement,
  srcCorners: Point[],
  outputSize: number
): HTMLCanvasElement {
  const srcCtx = srcCanvas.getContext('2d')!;
  const srcData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const sw = srcCanvas.width;

  const dst = document.createElement('canvas');
  dst.width = outputSize;
  dst.height = outputSize;
  const dstCtx = dst.getContext('2d')!;
  const dstData = dstCtx.createImageData(outputSize, outputSize);

  const [tl, tr, br, bl] = srcCorners;

  for (let dy = 0; dy < outputSize; dy++) {
    for (let dx = 0; dx < outputSize; dx++) {
      const u = dx / (outputSize - 1);
      const v = dy / (outputSize - 1);

      // Bilinear interpolation of source coordinates
      const sx =
        (1 - u) * (1 - v) * tl[0] +
        u * (1 - v) * tr[0] +
        u * v * br[0] +
        (1 - u) * v * bl[0];

      const sy =
        (1 - u) * (1 - v) * tl[1] +
        u * (1 - v) * tr[1] +
        u * v * br[1] +
        (1 - u) * v * bl[1];

      // Nearest-neighbour sampling
      const sxi = Math.round(sx);
      const syi = Math.round(sy);

      if (sxi >= 0 && sxi < srcCanvas.width && syi >= 0 && syi < srcCanvas.height) {
        const si = (syi * sw + sxi) * 4;
        const di = (dy * outputSize + dx) * 4;
        dstData.data[di] = srcData.data[si];
        dstData.data[di + 1] = srcData.data[si + 1];
        dstData.data[di + 2] = srcData.data[si + 2];
        dstData.data[di + 3] = 255;
      }
    }
  }

  dstCtx.putImageData(dstData, 0, 0);
  return dst;
}

/**
 * Attempt to detect the largest quadrilateral (the Sudoku grid) in the image.
 * Falls back to using the full image if detection fails.
 */
function detectAndWarpGrid(sourceCanvas: HTMLCanvasElement, outputSize = 900): HTMLCanvasElement {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  const ctx = sourceCanvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, w, h);

  // 1. Grayscale → Blur → Sobel edges
  const gray = toGrayscale(imageData);
  const blurred = gaussianBlur(gray, w, h);
  const edges = sobelEdge(blurred, w, h);

  // 2. Threshold edges and dilate to connect fragments
  const edgeThreshold = otsuThreshold(edges);
  const binaryEdges = new Uint8Array(edges.length);
  for (let i = 0; i < edges.length; i++) {
    binaryEdges[i] = edges[i] > edgeThreshold * 0.5 ? 255 : 0;
  }
  const dilated = dilate(binaryEdges, w, h, 3);

  // 3. Find connected components and pick the largest
  const components = findConnectedComponents(dilated, w, h);

  // Filter: the grid should be a substantial portion of the image
  const minArea = w * h * 0.05;
  const candidates = components
    .filter(c => c.length > minArea)
    .sort((a, b) => b.length - a.length);

  if (candidates.length === 0) {
    console.warn('Grid detection failed — using full image');
    // Fallback: warp the full image as-is
    return perspectiveWarp(
      sourceCanvas,
      [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
      outputSize
    );
  }

  // 4. Find corners of the largest component
  const corners = findCorners(candidates[0], w);

  // Sanity check: corners should form a reasonable quadrilateral
  const area = quadArea(corners);
  if (area < w * h * 0.04) {
    console.warn('Detected region too small — using full image');
    return perspectiveWarp(
      sourceCanvas,
      [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
      outputSize
    );
  }

  // 5. Perspective warp to a perfect square
  return perspectiveWarp(sourceCanvas, corners, outputSize);
}

/** Compute the area of a quadrilateral using the Shoelace formula. */
function quadArea(corners: Point[]): number {
  const n = corners.length;
  let area = 0;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = corners[i];
    const [x2, y2] = corners[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

// ═══════════════════════════════════════════════════════
//  STEP 3 — Cell Segmentation & OCR
// ═══════════════════════════════════════════════════════

/**
 * Crop a single cell from the warped grid image, applying inset padding
 * to avoid grid lines.
 */
function extractCell(
  src: HTMLCanvasElement,
  row: number,
  col: number,
  cellSize: number
): HTMLCanvasElement {
  const insetRatio = 0.18; // 18% inset to skip grid lines
  const inset = Math.floor(cellSize * insetRatio);

  const sx = col * cellSize + inset;
  const sy = row * cellSize + inset;
  const sw = cellSize - 2 * inset;
  const sh = cellSize - 2 * inset;

  const cell = document.createElement('canvas');
  // Up-scale the cell for better OCR accuracy
  const scale = 3;
  cell.width = sw * scale;
  cell.height = sh * scale;

  const ctx = cell.getContext('2d')!;
  // White background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, cell.width, cell.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, cell.width, cell.height);

  return cell;
}

/**
 * Apply Otsu thresholding to a cell canvas for clean B/W output.
 * Also adds white border padding for Tesseract.
 */
function thresholdCell(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = toGrayscale(imageData);
  const binary = applyOtsu(gray);

  // Invert if needed: we want black digits on white background.
  // Count dark vs light pixels to decide.
  let darkCount = 0;
  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === 0) darkCount++;
  }
  const shouldInvert = darkCount > binary.length * 0.5;

  for (let i = 0; i < binary.length; i++) {
    const v = shouldInvert ? (binary[i] === 0 ? 255 : 0) : binary[i];
    const j = i * 4;
    imageData.data[j] = v;
    imageData.data[j + 1] = v;
    imageData.data[j + 2] = v;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  // Add white border padding (helps Tesseract)
  const padded = document.createElement('canvas');
  const pad = 12;
  padded.width = canvas.width + pad * 2;
  padded.height = canvas.height + pad * 2;
  const pCtx = padded.getContext('2d')!;
  pCtx.fillStyle = '#FFFFFF';
  pCtx.fillRect(0, 0, padded.width, padded.height);
  pCtx.drawImage(canvas, pad, pad);

  return padded;
}

/**
 * Determine if a thresholded cell is empty (no digit).
 * Uses a refined dark-pixel ratio check on the centre region only.
 */
function isCellEmpty(canvas: HTMLCanvasElement): boolean {
  const ctx = canvas.getContext('2d')!;

  // Only inspect the centre 60% to ignore any residual border noise
  const marginX = Math.floor(canvas.width * 0.2);
  const marginY = Math.floor(canvas.height * 0.2);
  const regionW = canvas.width - 2 * marginX;
  const regionH = canvas.height - 2 * marginY;

  const regionData = ctx.getImageData(marginX, marginY, regionW, regionH);
  const d = regionData.data;

  let darkPixels = 0;
  const total = regionW * regionH;

  for (let i = 0; i < d.length; i += 4) {
    if (d[i] < 128) darkPixels++;
  }

  // A digit typically occupies > 4% of the centre area
  return darkPixels / total < 0.04;
}

// ═══════════════════════════════════════════════════════
//  PUBLIC API
// ═══════════════════════════════════════════════════════

/**
 * Full OCR pipeline:
 *   source image → grayscale + Otsu → grid detection + warp → cell segment → OCR
 *
 * @param imageSource  A blob URL string or an HTMLCanvasElement
 * @returns 9×9 number array (0 = empty cell)
 */
export async function processImage(imageSource: string | HTMLCanvasElement): Promise<Grid> {
  // ── Load source into a canvas ──
  let sourceCanvas: HTMLCanvasElement;

  if (typeof imageSource === 'string') {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.crossOrigin = 'anonymous';
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = imageSource;
    });
    sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = img.naturalWidth;
    sourceCanvas.height = img.naturalHeight;
    const ctx = sourceCanvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
  } else {
    sourceCanvas = imageSource;
  }

  console.time('OCR Pipeline');

  // ── Step 1+2: Detect grid & perspective warp to a 900×900 square ──
  console.time('Grid Detection & Warp');
  const warpedSize = 900; // 100px per cell
  const warped = detectAndWarpGrid(sourceCanvas, warpedSize);
  console.timeEnd('Grid Detection & Warp');

  // ── Apply Otsu thresholding to the entire warped grid ──
  console.time('Otsu Threshold');
  const wCtx = warped.getContext('2d')!;
  const wImageData = wCtx.getImageData(0, 0, warped.width, warped.height);
  const wGray = toGrayscale(wImageData);
  const wBinary = applyOtsu(wGray);

  // Write binary back to warped canvas
  for (let i = 0; i < wBinary.length; i++) {
    const j = i * 4;
    wImageData.data[j] = wBinary[i];
    wImageData.data[j + 1] = wBinary[i];
    wImageData.data[j + 2] = wBinary[i];
    wImageData.data[j + 3] = 255;
  }
  wCtx.putImageData(wImageData, 0, 0);
  console.timeEnd('Otsu Threshold');

  // ── Step 3: Segment into 81 cells and OCR each ──
  const cellSize = warpedSize / 9;
  const grid: Grid = Array(9).fill(null).map(() => Array(9).fill(0));

  console.time('Tesseract Init');
  const worker = await Tesseract.createWorker('eng');
  await worker.setParameters({
    tessedit_char_whitelist: '123456789',
    tessedit_pageseg_mode: Tesseract.PSM.SINGLE_CHAR,
  });
  console.timeEnd('Tesseract Init');

  console.time('Cell OCR');
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      // Extract and threshold the individual cell
      const rawCell = extractCell(warped, row, col, cellSize);
      const cell = thresholdCell(rawCell);

      if (isCellEmpty(cell)) {
        grid[row][col] = 0;
        continue;
      }

      try {
        const { data } = await worker.recognize(cell);
        const text = data.text.trim().replace(/\D/g, '');
        const num = parseInt(text.charAt(0), 10);
        grid[row][col] = num >= 1 && num <= 9 ? num : 0;
      } catch {
        grid[row][col] = 0;
      }
    }
  }
  console.timeEnd('Cell OCR');

  await worker.terminate();
  console.timeEnd('OCR Pipeline');

  return grid;
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
