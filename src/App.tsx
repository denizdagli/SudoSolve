import React, { useState, useCallback } from 'react';
import {
  ScanLine,
  Terminal,
  Camera,
  Upload,
  Wand2,
  RotateCcw,
  CheckCircle,
  PlusCircle,
  Share2,
  History as HistoryIcon,
  Grid3X3,
  Loader2,
  ArrowLeft,
  Zap,
  Code2,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { CellValue, AppView, SudokuGrid as SudokuGridType } from './types';
import { solveSudokuWithTiming, validatePuzzle } from './Solver';

import SudokuGridComponent from './components/SudokuGrid';
import CameraInput from './components/CameraInput';

// ─── Helpers ───
const createEmptyGrid = (): CellValue[][] =>
  Array(9).fill(null).map(() => Array(9).fill(null));

const createEmptyFixed = (): boolean[][] =>
  Array(9).fill(null).map(() => Array(9).fill(false));

// Convert CellValue[][] to number[][] (null -> 0)
const toNumberGrid = (grid: CellValue[][]): number[][] =>
  grid.map(row => row.map(cell => cell ?? 0));

// Convert number[][] to CellValue[][] (0 -> null)
const toCellGrid = (grid: number[][]): CellValue[][] =>
  grid.map(row => row.map(cell => cell === 0 ? null : cell));

// Sample puzzle for demo
const samplePuzzle: CellValue[][] = [
  [5, 3, null, null, 7, null, null, null, null],
  [6, null, null, 1, 9, 5, null, null, null],
  [null, 9, 8, null, null, null, null, 6, null],
  [8, null, null, null, 6, null, null, null, 3],
  [4, null, null, 8, null, 3, null, null, 1],
  [7, null, null, null, 2, null, null, null, 6],
  [null, 6, null, null, null, null, 2, 8, null],
  [null, null, null, 4, 1, 9, null, null, 5],
  [null, null, null, null, 8, null, null, 7, 9],
];

export default function App() {
  const [view, setView] = useState<AppView>('home');
  const [grid, setGrid] = useState<CellValue[][]>(createEmptyGrid());
  const [fixedCells, setFixedCells] = useState<boolean[][]>(createEmptyFixed());
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null);
  const [history, setHistory] = useState<SudokuGridType[]>([]);
  const [solveTimeMs, setSolveTimeMs] = useState<number>(0);
  const [processingMessage, setProcessingMessage] = useState('Initializing OCR engine...');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ─── Grid Setup ───
  const setupGrid = useCallback((newGrid: CellValue[][]) => {
    setGrid(newGrid);
    const fixed = newGrid.map(row => row.map(cell => cell !== null));
    setFixedCells(fixed);
    setSelectedCell(null);
    setErrorMessage(null);
    setView('editor');
  }, []);

  // ─── Load sample puzzle ───
  const handleLoadSample = () => {
    setupGrid(samplePuzzle.map(row => [...row]));
  };

  // ─── Manual empty grid ───
  const handleManualEntry = () => {
    setGrid(createEmptyGrid());
    setFixedCells(createEmptyFixed());
    setSelectedCell(null);
    setErrorMessage(null);
    setView('editor');
  };

  // ─── Camera capture → Backend API ───
  const handleCameraCapture = async (canvas: HTMLCanvasElement) => {
    setView('processing');
    setProcessingMessage('Sending to Python backend...');

    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg'));
      if (!blob) throw new Error('Could not create image blob');

      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');

      const res = await fetch('/api/scan', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('API request failed');

      const data = await res.json();
      if (data.status !== 'success') {
        throw new Error(data.message || 'API processing failed');
      }

      console.log('Received Grid:', data.grid);
      const cellGrid = toCellGrid(data.grid);
      setupGrid(cellGrid);
    } catch (err: any) {
      console.error('OCR failed:', err);
      setErrorMessage(err.message || 'OCR processing failed. Try again or enter manually.');
      setView('editor');
      setGrid(createEmptyGrid());
      setFixedCells(createEmptyFixed());
    }
  };

  // ─── Image upload → OCR ───
  const handleImageUpload = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      setView('processing');
      setProcessingMessage('Sending image to Python backend...');

      try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/api/scan', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error('API request failed');

        const data = await res.json();
        if (data.status !== 'success') {
          throw new Error(data.message || 'API processing failed');
        }

        console.log('Received Grid:', data.grid);
        const cellGrid = toCellGrid(data.grid);
        setupGrid(cellGrid);
      } catch (err: any) {
        console.error('OCR failed:', err);
        setErrorMessage(err.message || 'OCR processing failed. Try again or enter manually.');
        setView('editor');
        setGrid(createEmptyGrid());
        setFixedCells(createEmptyFixed());
      }
    };
    input.click();
  };

  // ─── Solve ───
  const handleSolve = () => {
    setErrorMessage(null);
    const numGrid = toNumberGrid(grid);

    // Validate first
    if (!validatePuzzle(numGrid)) {
      setErrorMessage('Invalid puzzle — conflicting numbers detected. Please check your input.');
      return;
    }

    const { solution, timeMs } = solveSudokuWithTiming(numGrid);

    if (!solution) {
      setErrorMessage('No solution found. The puzzle may be unsolvable.');
      return;
    }

    const solvedGrid = toCellGrid(solution);
    setGrid(solvedGrid);
    setSolveTimeMs(timeMs);
    setView('solved');

    const entry: SudokuGridType = {
      id: Date.now().toString(),
      cells: solvedGrid,
      fixedCells: fixedCells,
      solved: true,
      timestamp: Date.now(),
      solveTimeMs: timeMs,
    };
    setHistory(prev => [entry, ...prev]);
  };

  // ─── Cell input (editor) ───
  const handleCellInput = (value: CellValue) => {
    if (!selectedCell) return;
    const [r, c] = selectedCell;

    // Don't allow editing fixed cells
    if (fixedCells[r][c]) return;

    const newGrid = grid.map(row => [...row]);
    newGrid[r][c] = value;
    setGrid(newGrid);
    setErrorMessage(null);
  };

  // ─── Handle cell select ───
  const handleCellSelect = (cell: [number, number]) => {
    setSelectedCell(cell);
  };

  // ─── Reset ───
  const handleReset = () => {
    setGrid(createEmptyGrid());
    setFixedCells(createEmptyFixed());
    setSelectedCell(null);
    setErrorMessage(null);
    setSolveTimeMs(0);
    setView('home');
  };

  return (
    <div className="min-h-screen flex flex-col bg-deep-black text-text-primary relative overflow-hidden">
      {/* Subtle background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-terminal-green/[0.02] rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-python-yellow/[0.015] rounded-full blur-[100px]" />
      </div>

      {/* Header */}
      <header className="relative z-50 px-5 py-4 flex justify-between items-center border-b border-grid-line">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-terminal-green/10 border border-terminal-green/20 flex items-center justify-center">
            <Terminal className="w-4 h-4 text-terminal-green" />
          </div>
          <div>
            <h1 className="font-mono text-sm font-bold tracking-wider text-text-primary">
              SUDO<span className="text-terminal-green">SOLVE</span>
            </h1>
            <p className="font-mono text-[9px] text-text-dim tracking-widest">v1.0.0</p>
          </div>
        </div>

        {view !== 'home' && view !== 'camera' && view !== 'processing' && (
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-high border border-grid-line
              text-text-secondary hover:text-text-primary hover:border-terminal-green/20 transition-all font-mono text-xs"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Home
          </button>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 relative z-10 overflow-y-auto">
        <AnimatePresence mode="wait">

          {/* ═══ HOME ═══ */}
          {view === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="flex flex-col items-center px-5 pt-16 pb-24 max-w-lg mx-auto"
            >
              {/* Logo */}
              <div className="mb-14 text-center">
                <div className="inline-flex items-center justify-center mb-6">
                  <div className="relative">
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-terminal-green/20 to-terminal-green/5
                      border border-terminal-green/30 flex items-center justify-center pulse-glow">
                      <Code2 className="w-10 h-10 text-terminal-green" />
                    </div>
                    <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-python-yellow border-2 border-deep-black" />
                  </div>
                </div>

                <h2 className="font-mono text-4xl md:text-5xl font-extrabold tracking-tight mb-3">
                  Sudo<span className="text-terminal-green">Solve</span>
                </h2>
                <div className="font-mono text-xs text-text-dim tracking-[0.3em] uppercase">
                  <span className="text-python-yellow">$</span> instant sudoku recognition & solving
                </div>
              </div>

              {/* Main CTA — Scan Puzzle */}
              <button
                onClick={() => setView('camera')}
                className="w-full rounded-2xl p-8 mb-4 flex flex-col items-center text-center group
                  bg-surface border border-grid-line hover:border-terminal-green/30
                  transition-all duration-500 relative overflow-hidden"
              >
                <div className="scanline" />
                <div className="w-16 h-16 rounded-2xl bg-terminal-green/10 border border-terminal-green/20
                  flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-500">
                  <ScanLine className="w-8 h-8 text-terminal-green" />
                </div>
                <h3 className="font-mono text-xl font-bold text-text-primary mb-2">Scan Puzzle</h3>
                <p className="font-mono text-xs text-text-dim mb-6 max-w-xs leading-relaxed">
                  Open camera and align the Sudoku grid for automatic OCR recognition
                </p>
                <div className="bg-terminal-green text-deep-black px-8 py-3 rounded-xl font-mono font-bold text-sm
                  tracking-wider uppercase group-hover:shadow-[0_0_20px_rgba(0,255,65,0.3)] transition-all">
                  Initialize Scanner
                </div>
              </button>

              {/* Secondary Actions */}
              <div className="grid grid-cols-2 gap-3 w-full mb-4">
                <button
                  onClick={handleImageUpload}
                  className="rounded-xl p-5 flex flex-col items-start
                    bg-surface border border-grid-line hover:border-python-yellow/30
                    transition-all duration-300 group"
                >
                  <Upload className="w-6 h-6 text-python-yellow mb-3 group-hover:scale-110 transition-transform" />
                  <h4 className="font-mono text-sm font-bold mb-1">Upload</h4>
                  <p className="font-mono text-[10px] text-text-dim leading-relaxed">Import image file</p>
                </button>

                <button
                  onClick={handleManualEntry}
                  className="rounded-xl p-5 flex flex-col items-start
                    bg-surface border border-grid-line hover:border-terminal-green/30
                    transition-all duration-300 group"
                >
                  <Grid3X3 className="w-6 h-6 text-terminal-green mb-3 group-hover:scale-110 transition-transform" />
                  <h4 className="font-mono text-sm font-bold mb-1">Manual</h4>
                  <p className="font-mono text-[10px] text-text-dim leading-relaxed">Enter digits by hand</p>
                </button>
              </div>

              {/* Demo Button */}
              <button
                onClick={handleLoadSample}
                className="w-full rounded-xl p-4 flex items-center justify-center gap-3
                  bg-surface border border-grid-line hover:border-python-yellow/30
                  transition-all duration-300 group"
              >
                <Zap className="w-4 h-4 text-python-yellow" />
                <span className="font-mono text-xs text-text-secondary group-hover:text-python-yellow transition-colors">
                  Load demo puzzle
                </span>
              </button>

              {/* History Link */}
              {history.length > 0 && (
                <button
                  onClick={() => setView('history')}
                  className="mt-6 flex items-center gap-2 font-mono text-xs text-text-dim hover:text-terminal-green transition-colors"
                >
                  <HistoryIcon className="w-4 h-4" />
                  <span>View History ({history.length})</span>
                </button>
              )}
            </motion.div>
          )}

          {/* ═══ CAMERA ═══ */}
          {view === 'camera' && (
            <CameraInput
              onCapture={handleCameraCapture}
              onClose={() => setView('home')}
            />
          )}

          {/* ═══ PROCESSING ═══ */}
          {view === 'processing' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center min-h-[70vh] px-5"
            >
              <div className="relative mb-8">
                <div className="w-24 h-24 rounded-2xl bg-terminal-green/10 border border-terminal-green/20
                  flex items-center justify-center">
                  <Loader2 className="w-12 h-12 text-terminal-green animate-spin" />
                </div>
              </div>

              <div className="text-center">
                <h3 className="font-mono text-xl font-bold mb-3">Processing</h3>
                <div className="font-mono text-sm text-terminal-green mb-2">
                  {processingMessage}
                </div>
                <div className="flex items-center justify-center gap-1 mt-4">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 rounded-full bg-terminal-green"
                      animate={{ opacity: [0.2, 1, 0.2] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.3 }}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ═══ EDITOR ═══ */}
          {view === 'editor' && (
            <motion.div
              key="editor"
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.03 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col items-center px-5 pt-6 pb-8 max-w-lg mx-auto w-full"
            >
              {/* Status Bar */}
              <div className="w-full flex justify-between items-center mb-5">
                <div>
                  <p className="font-mono text-[10px] text-terminal-green uppercase tracking-[0.2em] mb-0.5">
                    <span className="text-python-yellow">$</span> status
                  </p>
                  <h2 className="font-mono text-xl font-bold">Review Grid</h2>
                </div>
                <div className="flex items-center gap-2 bg-terminal-green/10 px-3 py-1.5 rounded-lg border border-terminal-green/20">
                  <span className="w-2 h-2 rounded-full bg-terminal-green animate-pulse" />
                  <span className="font-mono text-[10px] text-terminal-green tracking-wider uppercase">Editing</span>
                </div>
              </div>

              {/* Error Message */}
              {errorMessage && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
                >
                  <p className="font-mono text-xs text-red-400">{errorMessage}</p>
                </motion.div>
              )}

              {/* Grid */}
              <SudokuGridComponent
                grid={grid}
                fixedCells={fixedCells}
                selectedCell={selectedCell}
                onCellSelect={handleCellSelect}
                onCellInput={handleCellInput}
                interactive
              />

              {/* Action Buttons */}
              <div className="flex items-center gap-3 mt-6 w-full max-w-[420px]">
                <button
                  onClick={handleReset}
                  className="flex-1 py-4 px-4 rounded-xl flex items-center justify-center gap-2
                    bg-surface-high border border-grid-line text-text-secondary
                    hover:border-terminal-green/20 hover:text-text-primary transition-all font-mono text-xs"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
                <button
                  onClick={handleSolve}
                  className="flex-[2] py-4 px-6 bg-terminal-green text-deep-black rounded-xl
                    flex items-center justify-center gap-2 font-mono font-extrabold text-sm tracking-wider
                    hover:shadow-[0_0_25px_rgba(0,255,65,0.3)] active:scale-[0.97] transition-all"
                >
                  <Wand2 className="w-5 h-5" />
                  VERIFY & SOLVE
                </button>
              </div>
            </motion.div>
          )}

          {/* ═══ SOLVED ═══ */}
          {view === 'solved' && (
            <motion.div
              key="solved"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="flex flex-col items-center px-5 pt-6 pb-24 max-w-lg mx-auto w-full"
            >
              {/* Success Banner */}
              <div className="text-center mb-6">
                <div className="inline-flex items-center gap-2 bg-terminal-green/10 border border-terminal-green/20
                  px-4 py-2 rounded-lg mb-4">
                  <CheckCircle className="w-4 h-4 text-terminal-green" />
                  <span className="font-mono text-xs text-terminal-green tracking-wider uppercase font-bold">
                    Solved in {solveTimeMs < 1 ? '<1' : solveTimeMs.toFixed(1)}ms
                  </span>
                </div>
                <h2 className="font-mono text-3xl font-extrabold tracking-tight mb-1">
                  Challenge <span className="text-terminal-green">Complete</span>
                </h2>
                <p className="font-mono text-xs text-text-dim">Grid logic fully verified ✓</p>
              </div>

              {/* Solved Grid */}
              <SudokuGridComponent
                grid={grid}
                fixedCells={fixedCells}
                selectedCell={null}
                onCellSelect={() => {}}
                onCellInput={() => {}}
                solved
                interactive={false}
              />

              {/* Actions */}
              <div className="mt-8 flex flex-col gap-3 w-full max-w-[420px]">
                <button
                  onClick={handleReset}
                  className="w-full bg-terminal-green text-deep-black font-mono font-bold py-4 rounded-xl text-sm
                    flex items-center justify-center gap-2 hover:shadow-[0_0_25px_rgba(0,255,65,0.3)]
                    active:scale-[0.97] transition-all tracking-wider"
                >
                  <PlusCircle className="w-5 h-5" />
                  NEW PUZZLE
                </button>
                <div className="flex gap-3">
                  <button className="flex-1 font-mono font-bold py-3.5 rounded-xl flex items-center justify-center gap-2
                    bg-surface-high border border-grid-line text-text-secondary text-xs
                    hover:border-terminal-green/20 active:scale-95 transition-all">
                    <Share2 className="w-4 h-4" />
                    Share
                  </button>
                  <button
                    onClick={() => setView('history')}
                    className="flex-1 font-mono font-bold py-3.5 rounded-xl flex items-center justify-center gap-2
                    bg-surface-high border border-grid-line text-text-secondary text-xs
                    hover:border-terminal-green/20 active:scale-95 transition-all">
                    <HistoryIcon className="w-4 h-4" />
                    History
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {/* ═══ HISTORY ═══ */}
          {view === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col px-5 pt-8 pb-24 max-w-lg mx-auto w-full"
            >
              <h2 className="font-mono text-2xl font-extrabold mb-6">
                <span className="text-python-yellow">$</span> History
              </h2>
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 opacity-40">
                  <HistoryIcon className="w-12 h-12 mb-4 text-text-dim" />
                  <p className="font-mono text-sm text-text-dim">No puzzles solved yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map(item => (
                    <button
                      key={item.id}
                      onClick={() => { setGrid(item.cells); setFixedCells(item.fixedCells); setView('solved'); setSolveTimeMs(item.solveTimeMs || 0); }}
                      className="w-full bg-surface p-4 rounded-xl border border-grid-line flex items-center justify-between
                        hover:border-terminal-green/20 transition-all group"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-surface-high rounded-lg grid grid-cols-3 gap-px p-1.5 border border-grid-line">
                          {Array(9).fill(0).map((_, i) => (
                            <div key={i} className="bg-surface-higher rounded-sm" />
                          ))}
                        </div>
                        <div className="text-left">
                          <h4 className="font-mono text-sm font-bold">Puzzle #{item.id.slice(-4)}</h4>
                          <p className="font-mono text-[10px] text-text-dim">
                            {new Date(item.timestamp).toLocaleDateString()} • {item.solveTimeMs ? `${item.solveTimeMs.toFixed(1)}ms` : 'solved'}
                          </p>
                        </div>
                      </div>
                      <div className="w-8 h-8 rounded-lg bg-terminal-green/10 flex items-center justify-center
                        group-hover:bg-terminal-green/20 transition-colors">
                        <CheckCircle className="w-4 h-4 text-terminal-green" />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Bottom Nav */}
      {view !== 'camera' && view !== 'processing' && (
        <nav className="fixed bottom-0 left-0 w-full z-50 glass-panel border-t border-grid-line">
          <div className="flex justify-around items-center max-w-lg mx-auto px-6 py-3 pb-safe">
            <button
              onClick={() => setView('home')}
              className={`flex flex-col items-center justify-center px-6 py-2 rounded-xl transition-all duration-300 ${
                view !== 'history'
                  ? 'text-terminal-green bg-terminal-green/10'
                  : 'text-text-dim hover:text-text-secondary'
              }`}
            >
              <ScanLine className="w-5 h-5" />
              <span className="font-mono text-[9px] uppercase tracking-[0.15em] mt-1 font-bold">Scan</span>
            </button>
            <button
              onClick={() => setView('history')}
              className={`flex flex-col items-center justify-center px-6 py-2 rounded-xl transition-all duration-300 ${
                view === 'history'
                  ? 'text-terminal-green bg-terminal-green/10'
                  : 'text-text-dim hover:text-text-secondary'
              }`}
            >
              <HistoryIcon className="w-5 h-5" />
              <span className="font-mono text-[9px] uppercase tracking-[0.15em] mt-1 font-bold">History</span>
            </button>
          </div>
        </nav>
      )}
    </div>
  );
}
