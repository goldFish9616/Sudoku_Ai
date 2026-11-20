#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import copy
import math

# Points for completing 0, 1, 2 or 3 regions with one move
SCORE_TABLE = [0, 1, 3, 7]


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Alpha-beta minimax agent with iterative deepening.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Main entry point (anytime): iterative deepening over alpha-beta
    # ------------------------------------------------------------------
    def compute_best_move(self, game_state: GameState) -> None:
        # Work on our own copy: the search mutates the state (play / undo)
        state = copy.deepcopy(game_state)
        me = state.current_player

        moves = self.legal_moves(state)
        if not moves:
            return

        best_move = moves[0]
        self.propose_move(best_move)

        empties = sum(1 for v in state.board.squares if v == SudokuBoard.empty)
        depth = 1
        while depth <= empties:
            move = self.search_root(state, depth, me, best_move)
            if move is not None:
                best_move = move
                self.propose_move(best_move)
            depth += 1

    def search_root(self, state: GameState, depth: int, me: int, previous_best: Move):
        moves = self.order_moves(state.board, self.legal_moves(state))
        if previous_best is not None:
            moves.sort(key=lambda mv: not (mv.square == previous_best.square
                                           and mv.value == previous_best.value))

        best_value = -math.inf
        best_move = None
        alpha, beta = -math.inf, math.inf
        for move in moves:
            points = self.play(state, move)
            value = self.alphabeta(state, depth - 1, alpha, beta, me)
            self.undo(state, move, points)
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, best_value)
        return best_move

    # ------------------------------------------------------------------
    # Alpha-beta minimax
    # ------------------------------------------------------------------
    def alphabeta(self, state: GameState, depth: int, alpha: float, beta: float, me: int) -> float:
        if depth == 0 or self.is_board_full(state):
            return self.evaluate(state, me)

        moves = self.legal_moves(state)
        if not moves:
            # Rules: a player without allowed empty cells skips the turn
            state.current_player = 3 - state.current_player
            value = self.alphabeta(state, depth - 1, alpha, beta, me)
            state.current_player = 3 - state.current_player
            return value

        moves = self.order_moves(state.board, moves)

        if state.current_player == me:  # maximizing
            value = -math.inf
            for move in moves:
                points = self.play(state, move)
                value = max(value, self.alphabeta(state, depth - 1, alpha, beta, me))
                self.undo(state, move, points)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta cut-off
            return value
        else:  # minimizing
            value = math.inf
            for move in moves:
                points = self.play(state, move)
                value = min(value, self.alphabeta(state, depth - 1, alpha, beta, me))
                self.undo(state, move, points)
                beta = min(beta, value)
                if alpha >= beta:
                    break  # alpha cut-off
            return value

    # ------------------------------------------------------------------
    # Make / unmake a move on the (copied) state
    # ------------------------------------------------------------------
    def play(self, state: GameState, move: Move) -> int:
        board = state.board
        board.put(move.square, move.value)
        points = SCORE_TABLE[self.completed_regions(board, move.square)]
        player = state.current_player
        state.scores[player - 1] += points
        occupied = state.occupied_squares1 if player == 1 else state.occupied_squares2
        if occupied is not None:
            occupied.append(move.square)
        state.current_player = 3 - player
        return points

    def undo(self, state: GameState, move: Move, points: int) -> None:
        state.current_player = 3 - state.current_player
        player = state.current_player
        occupied = state.occupied_squares1 if player == 1 else state.occupied_squares2
        if occupied is not None:
            occupied.pop()
        state.scores[player - 1] -= points
        state.board.put(move.square, SudokuBoard.empty)

    @staticmethod
    def completed_regions(board: SudokuBoard, square) -> int:
        i, j = square
        N, m, n = board.N, board.m, board.n  # region: m rows x n columns
        sq = board.squares
        count = 0
        if all(sq[i * N + c] != SudokuBoard.empty for c in range(N)):
            count += 1
        if all(sq[r * N + j] != SudokuBoard.empty for r in range(N)):
            count += 1
        r0, c0 = (i // m) * m, (j // n) * n
        if all(sq[r * N + c] != SudokuBoard.empty
               for r in range(r0, r0 + m) for c in range(c0, c0 + n)):
            count += 1
        return count

    @staticmethod
    def order_moves(board: SudokuBoard, moves):
        """Move ordering: moves that complete more regions are searched first.
        A move completes a region if its cell is the only empty cell of that region."""
        N, m, n = board.N, board.m, board.n
        sq = board.squares
        empty = SudokuBoard.empty
        # One pass over the board: count empty cells per row, column and block
        row_empty = [0] * N
        col_empty = [0] * N
        block_empty = [0] * N
        blocks_per_row = N // n  # number of blocks next to each other
        for k in range(N * N):
            if sq[k] == empty:
                i, j = divmod(k, N)
                row_empty[i] += 1
                col_empty[j] += 1
                block_empty[(i // m) * blocks_per_row + j // n] += 1

        def gain(move):
            i, j = move.square
            return ((row_empty[i] == 1) + (col_empty[j] == 1)
                    + (block_empty[(i // m) * blocks_per_row + j // n] == 1))

        moves.sort(key=gain, reverse=True)
        return moves

    @staticmethod
    def is_board_full(state: GameState) -> bool:
        return SudokuBoard.empty not in state.board.squares

    # ------------------------------------------------------------------
    # That is possible for you to just make those functions, for now they are stubs and to some stuuf
    # ------------------------------------------------------------------
    def legal_moves(self, state: GameState):
        """Legal = empty cell, allowed for the player, respects C0, not taboo."""
        board = state.board
        N, m, n = board.N, board.m, board.n
        squares = state.player_squares()  # already only empty cells
        if squares is None:  # classic mode: every empty cell is allowed
            squares = [(i, j) for i in range(N) for j in range(N)
                       if board.get((i, j)) == SudokuBoard.empty]
        taboo = {(t.square, t.value) for t in state.taboo_moves}

        moves = []
        for (i, j) in squares:
            used = {board.get((i, c)) for c in range(N)}
            used |= {board.get((r, j)) for r in range(N)}
            r0, c0 = (i // m) * m, (j // n) * n
            used |= {board.get((r, c)) for r in range(r0, r0 + m) for c in range(c0, c0 + n)}
            for value in range(1, N + 1):
                if value not in used and ((i, j), value) not in taboo:
                    moves.append(Move((i, j), value))
        return moves

    # ------------------------------------------------------------------
    # This function(now!!) selects the permitted empty squares (player_squares(); in classic mode, all are empty),
    # removes values already present in the row, column and block (C0 rule), and taboo moves.
    # ------------------------------------------------------------------

    def evaluate(self, state: GameState, me: int) -> float:
        """Score difference from the point of view of player `me`."""
        return state.scores[me - 1] - state.scores[2 - me]

    # ------------------------------------------------------------------
    # Now(!!!) it is our points minus the opponent’s points. scores[2 - me] is the
    # opponent’s index (for me = 1 it is 1, for me = 2 it is 0).
    # ------------------------------------------------------------------