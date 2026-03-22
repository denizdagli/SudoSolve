/**
 * Sudoku Solver — Recursive Backtracking Algorithm
 * Takes a 9x9 grid (0 = empty) and returns the solved grid.
 */

type Grid = number[][];

function isValid(grid: Grid, row: number, col: number, num: number): boolean {
  // Check row
  for (let c = 0; c < 9; c++) {
    if (grid[row][c] === num) return false;
  }

  // Check column
  for (let r = 0; r < 9; r++) {
    if (grid[r][col] === num) return false;
  }

  // Check 3x3 box
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;
  for (let r = boxRow; r < boxRow + 3; r++) {
    for (let c = boxCol; c < boxCol + 3; c++) {
      if (grid[r][c] === num) return false;
    }
  }

  return true;
}

function findEmpty(grid: Grid): [number, number] | null {
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      if (grid[r][c] === 0) return [r, c];
    }
  }
  return null;
}

function solveRecursive(grid: Grid): boolean {
  const empty = findEmpty(grid);
  if (!empty) return true; // All cells filled — solved!

  const [row, col] = empty;

  for (let num = 1; num <= 9; num++) {
    if (isValid(grid, row, col, num)) {
      grid[row][col] = num;

      if (solveRecursive(grid)) return true;

      grid[row][col] = 0; // Backtrack
    }
  }

  return false; // No valid number found — trigger backtrack
}

/**
 * Solves a Sudoku puzzle using recursive backtracking.
 * @param puzzle - 9x9 array where 0 represents empty cells
 * @returns The solved 9x9 array, or null if no solution exists
 */
export function solveSudoku(puzzle: Grid): Grid | null {
  // Deep copy to avoid mutating the original
  const grid: Grid = puzzle.map(row => [...row]);

  const startTime = performance.now();
  const solved = solveRecursive(grid);
  const endTime = performance.now();

  console.log(`Solver completed in ${(endTime - startTime).toFixed(2)}ms`);

  return solved ? grid : null;
}

/**
 * Validates that a puzzle is well-formed (no conflicting pre-filled values).
 */
export function validatePuzzle(grid: Grid): boolean {
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const val = grid[r][c];
      if (val !== 0) {
        // Temporarily clear cell and check if value is valid
        grid[r][c] = 0;
        const valid = isValid(grid, r, c, val);
        grid[r][c] = val;
        if (!valid) return false;
      }
    }
  }
  return true;
}

/**
 * Gets the solve time for display purposes.
 */
export function solveSudokuWithTiming(puzzle: Grid): { solution: Grid | null; timeMs: number } {
  const grid: Grid = puzzle.map(row => [...row]);
  const startTime = performance.now();
  const solved = solveRecursive(grid);
  const endTime = performance.now();
  const timeMs = endTime - startTime;

  return {
    solution: solved ? grid : null,
    timeMs,
  };
}
