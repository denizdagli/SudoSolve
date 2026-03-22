import React from 'react';
import { Eraser } from 'lucide-react';
import type { CellValue } from '../types';

interface SudokuGridProps {
  grid: CellValue[][];
  fixedCells: boolean[][];
  selectedCell: [number, number] | null;
  onCellSelect: (cell: [number, number]) => void;
  onCellInput: (value: CellValue) => void;
  solved?: boolean;
  interactive?: boolean;
}

export default function SudokuGrid({
  grid,
  fixedCells,
  selectedCell,
  onCellSelect,
  onCellInput,
  solved = false,
  interactive = true,
}: SudokuGridProps) {
  return (
    <div className="w-full flex flex-col items-center gap-6">
      {/* Sudoku Board */}
      <div className="sudoku-board aspect-square w-full max-w-[420px]">
        <div className="grid grid-cols-9 w-full h-full border-2 border-terminal-green/60 rounded-lg overflow-hidden">
          {grid.map((row, r) =>
            row.map((cell, c) => {
              const isSelected = selectedCell?.[0] === r && selectedCell?.[1] === c;
              const isFixed = fixedCells[r][c];
              const isSameRow = selectedCell?.[0] === r;
              const isSameCol = selectedCell?.[1] === c;
              const isSameBox =
                selectedCell &&
                Math.floor(selectedCell[0] / 3) === Math.floor(r / 3) &&
                Math.floor(selectedCell[1] / 3) === Math.floor(c / 3);
              const isHighlighted = !isSelected && (isSameRow || isSameCol || isSameBox);

              // Border classes for 3x3 box separation
              const borderRight = (c + 1) % 3 === 0 && c < 8 ? 'border-r-2 border-r-terminal-green/40' : 'border-r border-r-grid-line';
              const borderBottom = (r + 1) % 3 === 0 && r < 8 ? 'border-b-2 border-b-terminal-green/40' : 'border-b border-b-grid-line';

              return (
                <button
                  key={`${r}-${c}`}
                  onClick={() => interactive && onCellSelect([r, c])}
                  disabled={!interactive}
                  className={`
                    flex items-center justify-center font-mono text-lg sm:text-xl md:text-2xl font-bold
                    transition-all duration-150 relative
                    ${borderRight} ${borderBottom}
                    ${isSelected
                      ? 'bg-terminal-green/20 ring-1 ring-inset ring-terminal-green shadow-[0_0_12px_rgba(0,255,65,0.15)]'
                      : isHighlighted
                        ? 'bg-white/[0.03]'
                        : 'bg-transparent hover:bg-white/[0.04]'
                    }
                    ${isFixed
                      ? 'text-python-yellow'
                      : cell
                        ? 'text-terminal-green'
                        : 'text-transparent'
                    }
                    ${solved && !isFixed ? 'text-terminal-green/80' : ''}
                    ${!interactive ? 'cursor-default' : 'cursor-pointer'}
                  `}
                >
                  {cell || ''}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Number Pad */}
      {interactive && !solved && (
        <div className="w-full max-w-[420px]">
          <div className="grid grid-cols-5 gap-2">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
              <button
                key={num}
                onClick={() => onCellInput(num)}
                className="h-14 flex items-center justify-center rounded-xl font-mono text-xl font-bold
                  bg-surface-high border border-grid-line text-terminal-green
                  hover:bg-terminal-green/10 hover:border-terminal-green/30
                  active:scale-90 transition-all duration-150"
              >
                {num}
              </button>
            ))}
            <button
              onClick={() => onCellInput(null)}
              className="h-14 flex items-center justify-center rounded-xl font-mono text-xl font-bold
                bg-surface-high border border-grid-line text-red-400/70
                hover:bg-red-500/10 hover:border-red-400/30
                active:scale-90 transition-all duration-150"
            >
              <Eraser className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
