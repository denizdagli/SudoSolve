export type CellValue = number | null;

export type AppView = 'home' | 'camera' | 'processing' | 'editor' | 'solved' | 'history';

export interface SudokuGrid {
  id: string;
  cells: CellValue[][];
  fixedCells: boolean[][]; // true = pre-filled (OCR or manual), false = solved by algorithm
  solved: boolean;
  timestamp: number;
  solveTimeMs?: number;
}

/** A 9x9 grid of numbers where 0 = empty */
export type NumberGrid = number[][];
