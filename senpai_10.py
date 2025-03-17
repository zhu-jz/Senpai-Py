# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:29:14 2024

@author: ZHU
"""

from __future__ import annotations

import copy
import io
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, Lock
from typing import List, Optional


class Util:
    class Timer:
        def __init__(self):
            self.reset()

        def reset(self):
            self.p_elapsed: int = 0  # milliseconds
            self.p_running: bool = False
            self.p_start: Optional[float] = None

        def _now(self) -> float:
            """Get the current time in seconds since the epoch."""
            return time.time()
        
        def _time(self) -> int:
            """Calculate elapsed time in milliseconds since the timer was started."""
            current = self._now()
            elapsed_seconds = current - self.p_start
            return int(elapsed_seconds * 1000)

        def start(self):
            """Start the timer."""
            self.p_start = self._now()
            self.p_running = True

        def stop(self):
            """Stop the timer and accumulate the elapsed time."""
            if self.p_running:
                elapsed = self._time()
                self.p_elapsed += elapsed
                self.p_running = False

        def elapsed(self) -> int:
            """
            Get the total elapsed time in milliseconds.
            If the timer is running, include the time since it was last started.
            """
            total = self.p_elapsed
            if self.p_running:
                total += self._time()
            return total

    class Lockable:
        def __init__(self):
            self.p_mutex = Lock()

        def lock(self):
            """Acquire the mutex lock."""
            self.p_mutex.acquire()

        def unlock(self):
            """Release the mutex lock."""
            self.p_mutex.release()

    class Waitable(Lockable):
        def __init__(self):
            super().__init__()
            self.p_cond = Condition(self.p_mutex)

        def wait(self):
            """
            Wait for a condition to be signaled.
            Assumes that the mutex lock has already been acquired.
            """
            self.p_cond.wait()

        def signal(self):
            """Notify one waiting thread."""
            self.p_cond.notify()

    class GlibcRand:
        def __init__(self, seed: int = 1) -> None:
            """
            Initialize the state for the GLIBC random() algorithm.
            This builds the initial state r[0..343] using:
            r[0] = seed
            for i in 1..30:  r[i] = (16807 * r[i-1]) % 2147483647
            for i in 31..33: r[i] = r[i-31]
            for i in 34..343: r[i] = (r[i-31] + r[i-3]) mod 2^32
            """
            self.r: list[int] = [0] * 344  # Pre-allocate state list for indices 0 to 343
            self.r[0] = seed
            for i in range(1, 31):
                self.r[i] = (16807 * self.r[i - 1]) % 2147483647
            for i in range(31, 34):
                self.r[i] = self.r[i - 31]
            for i in range(34, 344):
                self.r[i] = (self.r[i - 31] + self.r[i - 3]) & 0xFFFFFFFF  # modulo 2^32

            self.index: int = 344  # Next index to compute

        def rand(self) -> int:
            """
            Compute the next pseudo-random number.
            The algorithm computes:
                new_val = (r[index-31] + r[index-3]) mod 2^32,
            appends it to the state, and returns new_val >> 1 (a 31-bit value).
            """
            new_val: int = (self.r[self.index - 31] + self.r[self.index - 3]) & 0xFFFFFFFF
            self.r.append(new_val)
            self.index += 1
            return new_val >> 1

    @staticmethod
    def round(x: float) -> int:
        """Round a floating-point number to the nearest integer."""
        return int(math.floor(x + 0.5))

    @staticmethod
    def div(a: int, b: int) -> int:
        """
        Perform integer division with flooring.
        In C++, integer division truncates towards zero (like rounding towards 0)
        In Python, integer division always rounds down (towards negative infinity)
        """
        assert b > 0, "Divider must be positive"
        return a // b

    @staticmethod
    def sqrt(n: int) -> int:
        """Calculate the integer square root of a number."""
        return int(math.sqrt(float(n)))

    @staticmethod
    def is_square(n: int) -> bool:
        """Check if a number is a perfect square."""
        i = Util.sqrt(n)
        return i * i == n
    
    @staticmethod
    def rand_float() -> float:
        """Generate a random floating-point number in the range [0.0, 1.0)."""
        return Util.rng.rand() / 2147483648.0

    @staticmethod
    def rand_int(n: int) -> int:
        """
        Generate a random integer in the range [0, n).
        Asserts that n is positive.
        """
        assert n > 0, "n must be positive"
        return int(Util.rand_float() * n)

    @staticmethod
    def string_find(s: str, c: str) -> int:
        """
        Find the index of character `c` in string `s`.
        Returns -1 if `c` is not found.
        """
        return s.find(c)

    @staticmethod
    def string_case_equal(s0: str, s1: str) -> bool:
        """
        Compare two strings for case-insensitive equality.
        Returns True if they are equal (ignoring case), else False.
        """
        if len(s0) != len(s1):
            return False
        for a, b in zip(s0, s1):
            if a.lower() != b.lower():
                return False
        return True

    @staticmethod
    def to_bool(s: str) -> bool:
        """
        Convert a string to a boolean.
        Accepts "true" or "false" (case-insensitive).
        Exits the program if the input is invalid.
        """
        if Util.string_case_equal(s, "true"):
            return True
        elif Util.string_case_equal(s, "false"):
            return False
        else:
            print(f'not a boolean: "{s}"', file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def to_int(s: str) -> int:
        """
        Convert a string to an integer.
        Exits the program if the conversion fails.
        """
        try:
            return int(s)
        except ValueError:
            print(f'Error converting to int: "{s}"', file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def to_string(n: int) -> str:
        """Convert an integer to its string representation."""
        return str(n)

    @staticmethod
    def to_string_double(x: float) -> str:
        """Convert a floating-point number to its string representation."""
        return str(x)

    @staticmethod
    def log(s: str):
        """
        Append a string to the log file.
        Each entry is written on a new line.
        """
        with open("log.txt", "a") as log_file:
            log_file.write(s + "\n")

    @classmethod
    def init(cls):
        """Initialize the Util class."""
        cls.rng = cls.GlibcRand(seed=1)


class Input:
    class INPUT(Util.Waitable):    
        def __init__(self):
            super().__init__()
            self.p_has_input = False
            self.p_eof = False
            self.p_line = ""
    
        def has_input(self) -> bool:
            """Check if there is input available."""
            return self.p_has_input
    
        def get_line(self) -> tuple[bool, str]:
            """
            Retrieve a line of input.
            Blocks until input is available.
            Returns a tuple (line_ok, line), where:
                - line_ok is False if EOF has been reached.
                - line contains the input line if available.
            """
            self.lock()
            try:
                while not self.p_has_input:
                    self.wait()
    
                line_ok = not self.p_eof
                if line_ok:
                    line = self.p_line
                else:
                    line = ''
    
                self.p_has_input = False
                self.signal()
                return (line_ok, line)
            finally:
                self.unlock()
    
        def set_eof(self):
            """Set the end-of-file flag and notify waiting threads."""
            self.lock()
            try:
                while self.p_has_input:
                    self.wait()
    
                self.p_eof = True
                self.p_has_input = True
                self.signal()
            finally:
                self.unlock()
    
        def set_line(self, line: str):
            """Set a new line of input and notify waiting threads."""
            self.lock()
            try:
                while self.p_has_input:
                    self.wait()
    
                self.p_line = line
                self.p_has_input = True
                self.signal()
            finally:
                self.unlock()
    
    @staticmethod
    def input_program(input_obj):
        """
        Function to run in a separate thread.
        Continuously reads lines from standard input and passes them to the INPUT instance.
        """
        for line in sys.stdin:
            input_obj.set_line(line.rstrip('\n'))
        input_obj.set_eof()
    
    @classmethod
    def init(cls):
        """
        Initialize the input handling by starting the input thread.
        This should be called at the beginning of the program.
        """
        cls.input_instance = cls.INPUT()
        cls.input_thread = threading.Thread(target=cls.input_program, args=(cls.input_instance,))
        cls.input_thread.daemon = True  # Daemonize thread to exit with the main program
        cls.input_thread.start()


class Side:
    SIZE = 2
    WHITE = 0
    BLACK = 1

    @staticmethod
    def opposit(sd: int) -> int:
        """Return the opposite side."""
        return sd ^ 1


class Square:
    FILE_SIZE = 8
    RANK_SIZE = 8
    SIZE = FILE_SIZE * RANK_SIZE
    
    # Files
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H = range(8)
    
    # Ranks
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 = range(8)
    
    # Squares (A1 to H8)
    NONE = -1
    A1, A2, A3, A4, A5, A6, A7, A8, \
    B1, B2, B3, B4, B5, B6, B7, B8, \
    C1, C2, C3, C4, C5, C6, C7, C8, \
    D1, D2, D3, D4, D5, D6, D7, D8, \
    E1, E2, E3, E4, E5, E6, E7, E8, \
    F1, F2, F3, F4, F5, F6, F7, F8, \
    G1, G2, G3, G4, G5, G6, G7, G8, \
    H1, H2, H3, H4, H5, H6, H7, H8 = range(64)

    INC_LEFT = -8
    INC_RIGHT = +8

    CASTLING_DELTA = 16
    DOUBLE_PAWN_DELTA = 2
    
    @staticmethod
    def make(fl: int, rk: int, sd: int = None) -> int:
        """Create a square from file and rank, optionally adjusted by side."""
        assert fl < 8, f"File {fl} out of range"
        assert rk < 8, f"Rank {rk} out of range"

        if sd is None:
            return (fl << 3) | rk
        else:
            return Square.make(fl, (rk ^ -sd) & 7)

    @staticmethod
    def file(sq: int) -> int:
        """Get the file of the square."""
        return sq >> 3

    @staticmethod
    def rank(sq: int, sd: int = None) -> int:
        """Get the rank of the square, optionally adjusted by side."""
        if sd is None:
            return sq & 7
        else:
            return (sq ^ -sd) & 7

    @staticmethod
    def opposit_file(sq: int) -> int:
        """Get the opposite file."""
        return sq ^ 0o70  # 070 in octal is 56 in decimal

    @staticmethod
    def opposit_rank(sq: int) -> int:
        """Get the opposite rank."""
        return sq ^ 0o07  # 007 in octal is 7 in decimal

    @staticmethod
    def is_promotion(sq: int) -> bool:
        """Check if the square is a promotion rank."""
        rk = Square.rank(sq)
        return rk == Square.RANK_1 or rk == Square.RANK_8

    @staticmethod
    def colour(sq: int) -> int:
        """Determine the colour of the square."""
        return ((sq >> 3) ^ sq) & 1

    @staticmethod
    def same_colour(s0: int, s1: int) -> bool:
        """Check if two squares are the same colour."""
        diff = s0 ^ s1
        return (((diff >> 3) ^ diff) & 1) == 0

    @staticmethod
    def same_line(s0: int, s1: int) -> bool:
        """Check if two squares share the same file or rank."""
        return Square.file(s0) == Square.file(s1) or Square.rank(s0) == Square.rank(s1)

    @staticmethod
    def file_distance(s0: int, s1: int) -> int:
        """Calculate the file distance between two squares."""
        return abs(Square.file(s1) - Square.file(s0))

    @staticmethod
    def rank_distance(s0: int, s1: int) -> int:
        """Calculate the rank distance between two squares."""
        return abs(Square.rank(s1) - Square.rank(s0))

    @staticmethod
    def distance(s0: int, s1: int) -> int:
        """Calculate the Chebyshev distance between two squares."""
        return max(Square.file_distance(s0, s1), Square.rank_distance(s0, s1))

    @staticmethod
    def pawn_inc(sd: int) -> int:
        """Get the pawn increment based on side."""
        return 1 if sd == Side.WHITE else -1

    @staticmethod
    def stop(sq: int, sd: int) -> int:
        """Get the square in front of the pawn."""
        return sq + Square.pawn_inc(sd)

    @staticmethod
    def promotion(sq: int, sd: int) -> int:
        """Get the promotion square for a pawn."""
        return Square.make(Square.file(sq), Square.RANK_8, sd)

    @staticmethod
    def is_valid_88(s88: int) -> bool:
        """Check if a square in 0x88 representation is valid."""
        return (s88 & ~0x77) == 0

    @staticmethod
    def to_88(sq: int) -> int:
        """Convert square to 0x88 representation."""
        return sq + (sq & 0o70)  # 070 is octal for 56

    @staticmethod
    def from_88(s88: int) -> int:
        """Convert from 0x88 representation to square index."""
        assert Square.is_valid_88(s88), f"Invalid 0x88 square: {s88}"
        return (s88 + (s88 & 7)) >> 1

    @staticmethod
    def from_fen(sq: int) -> int:
        """Convert from FEN square to internal representation."""
        return Square.make(sq & 7, (sq >> 3) ^ 7)

    @staticmethod
    def from_string(s: str) -> int:
        """Convert from algebraic notation to internal square index."""
        assert len(s) == 2, "Square string must be of length 2"
        return Square.make(ord(s[0]) - ord('a'), ord(s[1]) - ord('1'))

    @staticmethod
    def to_string(sq: int) -> str:
        """Convert internal square index to algebraic notation."""
        file_char = chr(ord('a') + Square.file(sq))
        rank_char = chr(ord('1') + Square.rank(sq))
        return f"{file_char}{rank_char}"


class Wing:
    SIZE = 2
    KING = 0
    QUEEN = 1

    # Shelter files for pawn-shelter evaluation
    shelter_file = [Square.FILE_G, Square.FILE_B]


class Piece:
    SIZE = 7
    SIDE_SIZE = 12

    # Pieces
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5
    NONE = 6

    # Side_Piece enumeration
    WHITE_PAWN = 0
    BLACK_PAWN = 1
    WHITE_KNIGHT = 2
    BLACK_KNIGHT = 3
    WHITE_BISHOP = 4
    BLACK_BISHOP = 5
    WHITE_ROOK = 6
    BLACK_ROOK = 7
    WHITE_QUEEN = 8
    BLACK_QUEEN = 9
    WHITE_KING = 10
    BLACK_KING = 11

    # Piece values
    PAWN_VALUE = 100
    KNIGHT_VALUE = 325
    BISHOP_VALUE = 325
    ROOK_VALUE = 500
    QUEEN_VALUE = 975
    KING_VALUE = 10000  # for SEE

    Char = "PNBRQK?"
    Fen_Char = "PpNnBbRrQqKk"

    @staticmethod
    def is_minor(pc: int) -> bool:
        """Check if the piece is a minor piece."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        return pc == Piece.KNIGHT or pc == Piece.BISHOP

    @staticmethod
    def is_major(pc: int) -> bool:
        """Check if the piece is a major piece."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        return pc == Piece.ROOK or pc == Piece.QUEEN

    @staticmethod
    def is_slider(pc: int) -> bool:
        """Check if the piece is a sliding piece."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        return Piece.BISHOP <= pc <= Piece.QUEEN

    @staticmethod
    def score(pc: int) -> int:
        """Get the score of the piece for MVV/LVA."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        assert pc != Piece.NONE, "NONE piece has no score"
        return pc

    @staticmethod
    def value(pc: int) -> int:
        """Get the value of the piece."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        values = [
            Piece.PAWN_VALUE,
            Piece.KNIGHT_VALUE,
            Piece.BISHOP_VALUE,
            Piece.ROOK_VALUE,
            Piece.QUEEN_VALUE,
            Piece.KING_VALUE,
            0  # NONE
        ]
        return values[pc]

    @staticmethod
    def make(pc: int, sd: int) -> int:
        """Create a side-specific piece identifier."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        assert pc != Piece.NONE, "Cannot make NONE piece"
        assert sd < Side.SIZE, f"Side {sd} out of range"
        return (pc << 1) | sd

    @staticmethod
    def piece(p12: int) -> int:
        """Extract the piece type from side-specific piece."""
        assert p12 < Piece.SIDE_SIZE, f"Side-Piece index {p12} out of range"
        return p12 >> 1

    @staticmethod
    def side(p12: int) -> int:
        """Extract the side from side-specific piece."""
        assert p12 < Piece.SIDE_SIZE, f"Side-Piece index {p12} out of range"
        return p12 & 1

    @staticmethod
    def from_char(c: str) -> int:
        """Convert a character to a piece type."""
        return Util.string_find(Piece.Char, c)

    @staticmethod
    def to_char(pc: int) -> str:
        """Convert a piece type to its character representation."""
        assert pc < Piece.SIZE, f"Piece index {pc} out of range"
        return Piece.Char[pc]

    @staticmethod
    def from_fen(c: str) -> int:
        """Convert a FEN character to a side-specific piece."""
        return Util.string_find(Piece.Fen_Char, c)

    @staticmethod
    def to_fen(p12: int) -> str:
        """Convert a side-specific piece to its FEN character."""
        assert p12 < Piece.SIDE_SIZE, f"Side-Piece index {p12} out of range"
        return Piece.Fen_Char[p12]


class Move:
    # Constants
    FLAGS_BITS = 9
    FLAGS_SIZE = 1 << FLAGS_BITS
    FLAGS_MASK = FLAGS_SIZE - 1

    BITS = FLAGS_BITS + 12
    SIZE = 1 << BITS
    MASK = SIZE - 1

    SCORE_BITS = 32 - BITS
    SCORE_SIZE = 1 << SCORE_BITS
    SCORE_MASK = SCORE_SIZE - 1

    # Move Enums
    NONE = 0
    NULL_ = 1
    
    class SEE:
        def __init__(self):
            self.p_board: Optional['Board'] = None
            self.p_to: int = 0
            self.p_all: int = 0

            self.p_val: int = 0
            self.p_side: int = 0

        def init(self, t: int, sd: int):
            """
            Initialize the SEE evaluation.
            """
            self.p_to = t
            self.p_all = self.p_board.all_pieces()

            pc = self.p_board.square(t)

            self.p_val = Piece.value(pc)
            self.p_side = sd

        def move(self, f: int) -> int:
            """
            Execute a move in the SEE evaluation.
            """
            assert Bit.is_set(self.p_all, f), "Bit not set in p_all."
            self.p_all = Bit.clear_bit(self.p_all, f)

            pc = self.p_board.square(f)
            assert pc != Piece.NONE and self.p_board.square_side(f) == self.p_side, "Invalid piece or side."

            val = self.p_val
            self.p_val = Piece.value(pc)

            if pc == Piece.PAWN and Square.is_promotion(self.p_to):
                delta = Piece.QUEEN_VALUE - Piece.PAWN_VALUE
                val += delta
                self.p_val += delta

            if val == Piece.KING_VALUE:
                self.p_all = 0  # Erase all attackers

            self.p_side = Side.opposit(self.p_side)

            return val

        def see_rec(self, alpha: int, beta: int) -> int:
            """
            Recursive SEE evaluation.
            """
            assert alpha < beta, "Alpha must be less than beta."

            s0 = 0

            if s0 > alpha:
                alpha = s0
                if s0 >= beta:
                    return s0

            if self.p_val <= alpha:
                return self.p_val

            f = self.pick_lva()

            if f == Square.NONE:
                return s0

            cap = self.move(f)
            s1 = cap - self.see_rec(cap - beta, cap - alpha)

            return max(s0, s1)

        def pick_lva(self) -> int:
            """
            Pick the least valuable attacker.
            """
            sd = self.p_side

            for pc in range(Piece.PAWN, Piece.KING + 1):
                fs = self.p_board.piece(pc, sd) & Attack.pseudo_attacks_to(pc, sd, self.p_to) & self.p_all

                while fs:
                    b = fs & -fs
                    f = Bit.first(b)

                    if (self.p_all & Attack.Between[f][self.p_to]) == 0:
                        return f

                    fs &= fs - 1

            return Square.NONE

        def see(self, mv: int, alpha: int, beta: int, bd: 'Board.BOARD') -> int:
            """
            Perform SEE evaluation on a move.
            """
            assert alpha < beta, "Alpha must be less than beta."

            self.p_board = bd

            f = Move.from_sq(mv)
            t = Move.to_sq(mv)

            pc = self.p_board.square(f)
            sd = self.p_board.square_side(f)

            self.init(t, sd)
            cap = self.move(f)  # Assumes queen promotion

            if pc == Piece.PAWN and Square.is_promotion(t):
                delta = Piece.QUEEN_VALUE - Piece.value(Move.prom(mv))
                cap -= delta
                self.p_val -= delta

            return cap - self.see_rec(cap - beta, cap - alpha)

    @staticmethod
    def make_flags(pc: int, cp: int, pp: int = Piece.NONE) -> int:
        """
        Create move flags by encoding piece, captured piece, and promotion piece.
        Flags occupy 9 bits: pc (3 bits), cp (3 bits), pp (3 bits).
        """
        assert 0 <= pc < Piece.SIZE, "Piece type out of range."
        assert 0 <= cp < Piece.SIZE, "Captured piece type out of range."
        assert 0 <= pp < Piece.SIZE, "Promotion piece type out of range."

        return (pc << 6) | (cp << 3) | pp

    @staticmethod
    def make(f: int, t: int, pc: int, cp: int, pp: int = Piece.NONE) -> int:
        """
        Encode a move into an integer.
        Bit layout:
        [30-18] Piece (3 bits)
        [17-15] Captured Piece (3 bits)
        [14-12] Promotion Piece (3 bits)
        [11-6] From Square (6 bits)
        [5-0] To Square (6 bits)
        """
        assert 0 <= f < Square.SIZE, "From square out of range."
        assert 0 <= t < Square.SIZE, "To square out of range."
        assert 0 <= pc < Piece.SIZE, "Piece type out of range."
        assert 0 <= cp < Piece.SIZE, "Captured piece type out of range."
        assert 0 <= pp < Piece.SIZE, "Promotion piece type out of range."
        assert pc != Piece.NONE, "Piece type cannot be NONE in a move."
        assert pp == Piece.NONE or pc == Piece.PAWN, "Only pawns can promote."

        return (pc << 18) | (cp << 15) | (pp << 12) | (f << 6) | t

    @staticmethod
    def from_sq(mv: int) -> int:
        """
        Extract the 'from' square from a move.
        """
        assert mv != Move.NONE, "Cannot extract from_sq from NONE."
        assert mv != Move.NULL_, "Cannot extract from_sq from NULL_."
        return (mv >> 6) & 0o77  # 077 octal == 63 decimal

    @staticmethod
    def to_sq(mv: int) -> int:
        """
        Extract the 'to' square from a move.
        """
        assert mv != Move.NONE, "Cannot extract to_sq from NONE."
        assert mv != Move.NULL_, "Cannot extract to_sq from NULL_."
        return mv & 0o77  # 077 octal == 63 decimal

    @staticmethod
    def piece(mv: int) -> int:
        """
        Extract the piece type from a move.
        """
        assert mv != Move.NONE, "Cannot extract piece from NONE."
        assert mv != Move.NULL_, "Cannot extract piece from NULL_."
        return (mv >> 18) & 0b111  # Extract 3 bits

    @staticmethod
    def cap(mv: int) -> int:
        """
        Extract the captured piece type from a move.
        """
        assert mv != Move.NONE, "Cannot extract captured_piece from NONE."
        assert mv != Move.NULL_, "Cannot extract captured_piece from NULL_."
        return (mv >> 15) & 0b111  # Extract 3 bits

    @staticmethod
    def prom(mv: int) -> int:
        """
        Extract the promotion piece type from a move.
        """
        assert mv != Move.NONE, "Cannot extract promotion_piece from NONE."
        assert mv != Move.NULL_, "Cannot extract promotion_piece from NULL_."
        return (mv >> 12) & 0b111  # Extract 3 bits

    @staticmethod
    def flags(mv: int) -> int:
        """
        Extract the flags from a move.
        """
        assert mv != Move.NONE, "Cannot extract flags from NONE."
        assert mv != Move.NULL_, "Cannot extract flags from NULL_."
        return (mv >> 12) & 0o777  # 0777 octal == 511 decimal

    @staticmethod
    def to_can(mv: int) -> str:
        """
        Convert a move to coordinate (CAN) notation.
        Example: e2e4, e7e8q
        """
        assert mv != Move.NONE, "Cannot convert NONE move to CAN."
        assert mv != Move.NULL_, "Cannot convert NULL_ move to CAN."

        from_sq = Move.from_sq(mv)
        to_sq = Move.to_sq(mv)
        prom = Move.prom(mv)

        move_str = Square.to_string(from_sq) + Square.to_string(to_sq)

        if prom != Piece.NONE:
            move_str += Piece.to_char(prom).lower()

        return move_str

    @staticmethod
    def is_capture(mv: int) -> bool:
        """
        Check if the move is a capture.
        """
        return Move.cap(mv) != Piece.NONE

    @staticmethod
    def is_en_passant(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Check if the move is an en passant capture.
        """
        return Move.piece(mv) == Piece.PAWN and Move.to_sq(mv) == bd.ep_sq()

    @staticmethod
    def is_recapture(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Check if the move is a recapture.
        """
        return Move.to_sq(mv) == bd.recap() and Move.is_win(mv, bd)

    @staticmethod
    def is_promotion(mv: int) -> bool:
        """
        Check if the move is a promotion.
        """
        return Move.prom(mv) != Piece.NONE

    @staticmethod
    def is_queen_promotion(mv: int) -> bool:
        """
        Check if the move is a queen promotion.
        """
        return Move.prom(mv) == Piece.QUEEN

    @staticmethod
    def is_under_promotion(mv: int) -> bool:
        """
        Check if the move is an under-promotion.
        """
        pp = Move.prom(mv)
        return pp != Piece.NONE and pp != Piece.QUEEN

    @staticmethod
    def is_tactical(mv: int) -> bool:
        """
        Check if the move is tactical (capture or promotion).
        """
        return Move.is_capture(mv) or Move.is_promotion(mv)

    @staticmethod
    def is_pawn_push(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Check if the move is a pawn push.
        """
        if Move.is_tactical(mv):
            return False

        f = Move.from_sq(mv)
        t = Move.to_sq(mv)

        pc = bd.square(f)
        sd = bd.square_side(f)

        return (pc == Piece.PAWN and
                Square.rank(t, sd) >= Square.RANK_6 and
                Pawn.is_passed(t, sd, bd) and
                not Move.is_capture(mv))

    @staticmethod
    def is_castling(mv: int) -> bool:
        """
        Check if the move is castling.
        """
        return Move.piece(mv) == Piece.KING and \
               abs(Move.to_sq(mv) - Move.from_sq(mv)) == Square.CASTLING_DELTA

    @staticmethod
    def is_conversion(mv: int) -> bool:
        """
        Check if the move involves a conversion (capture, pawn move, or castling).
        """
        return Move.is_capture(mv) or Move.piece(mv) == Piece.PAWN or Move.is_castling(mv)

    @staticmethod
    def make_move(f: int, t: int, pp: int, bd: 'Board.BOARD') -> int:
        """
        Create a move with additional board context.
        """
        pc = bd.square(f)
        cp = bd.square(t)

        if pc == Piece.PAWN and t == bd.ep_sq():
            cp = Piece.PAWN

        if pc == Piece.PAWN and Square.is_promotion(t) and pp == Piece.NONE:
            pp = Piece.QUEEN

        return Move.make(f, t, pc, cp, pp)

    @staticmethod
    def from_string(s: str, bd: 'Board.BOARD') -> int:
        """
        Convert a string representation of a move to its integer encoding.
        """
        assert len(s) >= 4, "Move string must be at least 4 characters."

        f = Square.from_string(s[:2])
        t = Square.from_string(s[2:4])
        pp = Piece.from_char(s[4].upper()) if len(s) > 4 else Piece.NONE

        return Move.make_move(f, t, pp, bd)
    
    @staticmethod
    def see(mv: int, alpha: int, beta: int, bd: 'Board.BOARD') -> int:
        """
        Static Exchange Evaluation (SEE) for a move.
        """
        see_evaluator = Move.SEE()
        return see_evaluator.see(mv, alpha, beta, bd)

    @staticmethod
    def see_max(mv: int) -> int:
        """
        Calculate the maximum SEE (Static Exchange Evaluation) score for a move.
        """
        assert Move.is_tactical(mv), "SEE max called on non-tactical move."

        sc = Piece.value(Move.cap(mv))

        pp = Move.prom(mv)
        if pp != Piece.NONE:
            sc += Piece.value(pp) - Piece.PAWN_VALUE

        return sc

    @staticmethod
    def is_safe(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if making a move is safe based on SEE.
        """
        pc = Move.piece(mv)
        cp = Move.cap(mv)
        pp = Move.prom(mv)

        if pc == Piece.KING:
            return True
        elif Piece.value(cp) >= Piece.value(pc):
            return True
        elif pp != Piece.NONE and pp != Piece.QUEEN:
            return False
        else:
            return Move.see(mv, -1, 0, bd) >= 0

    @staticmethod
    def is_win(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if making a move results in a win based on SEE.
        """
        assert Move.is_tactical(mv), "is_win called on non-tactical move."

        pc = Move.piece(mv)
        cp = Move.cap(mv)
        pp = Move.prom(mv)

        if pc == Piece.KING:
            return True
        elif Piece.value(cp) > Piece.value(pc):
            return True
        elif pp != Piece.NONE and pp != Piece.QUEEN:
            return False
        else:
            return Move.see(mv, 0, +1, bd) > 0

    @staticmethod
    def is_legal_debug(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Debug function to check move legality by making and undoing the move.
        """
        bd.move(mv)
        b = Attack.is_legal(bd)
        bd.undo()
        return b

    @staticmethod
    def is_legal(mv: int, bd: 'Board.BOARD', attacks: 'Attack.Attacks') -> bool:
        """
        Check if a move is legal based on board state and attack information.
        """
        sd = bd.turn()

        f = Move.from_sq(mv)
        t = Move.to_sq(mv)

        if Move.is_en_passant(mv, bd):
            return Move.is_legal_debug(mv, bd)

        if Move.piece(mv) == Piece.KING:
            return not Attack.is_attacked(t, Side.opposit(sd), bd)

        if not Bit.is_set(attacks.pinned, f):
            return True

        if Bit.is_set(Attack.ray(bd.king(sd), f), t):
            return True

        return False

    @staticmethod
    def is_check_debug(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Debug function to check if a move results in a check.
        """
        bd.move(mv)
        b = Attack.is_in_check(bd)
        bd.undo()
        return b

    @staticmethod
    def is_check(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if making a move results in a check.
        """
        if Move.is_promotion(mv) or Move.is_en_passant(mv, bd) or Move.is_castling(mv):
            return Move.is_check_debug(mv, bd)

        f = Move.from_sq(mv)
        t = Move.to_sq(mv)

        pc = Move.prom(mv) if Move.prom(mv) != Piece.NONE else Move.piece(mv)
        sd = bd.square_side(f)

        king = bd.king(Side.opposit(sd))

        if Attack.attack(pc, sd, t, king, bd):
            return True

        if Attack.attack_behind(king, f, sd, bd) and not Bit.is_set(Attack.ray(king, f), t):
            return True

        return False


class Bit:
    # Bitboards for left, right, front, rear
    p_left = [0 for _ in range(8)]
    p_right = [0 for _ in range(8)]
    p_front = [0 for _ in range(8)]
    p_rear = [0 for _ in range(8)]

    # Side-specific front and rear bitboards
    p_side_front = [[0 for _ in range(8)] for _ in range(Side.SIZE)]
    p_side_rear = [[0 for _ in range(8)] for _ in range(Side.SIZE)]

    @staticmethod
    def bit(n: int) -> int:
        """Return a bitboard with only the nth bit set."""
        assert 0 <= n < 64, "Bit index out of range"
        return 1 << n

    @staticmethod
    def set_bit(b: int, n: int) -> int:
        """Set the nth bit in the bitboard."""
        assert 0 <= n < 64, "Bit index out of range"
        return b | Bit.bit(n)

    @staticmethod
    def clear_bit(b: int, n: int) -> int:
        """Clear the nth bit in the bitboard."""
        assert 0 <= n < 64, "Bit index out of range"
        assert b & ~Bit.bit(n) == b & ~Bit.bit(n) & 0xFFFFFFFFFFFFFFFF
        return b & ~Bit.bit(n)

    @staticmethod
    def is_set(b: int, n: int) -> bool:
        """Check if the nth bit is set in the bitboard."""
        assert 0 <= n < 64, "Bit index out of range"
        return (b & Bit.bit(n)) != 0

    @staticmethod
    def first(b: int) -> int:
        """Return the index of the least significant set bit."""
        assert b != 0, "Bitboard is empty"
        return (b & -b).bit_length() - 1  # Equivalent to __builtin_ctzll(b)

    @staticmethod
    def rest(b: int) -> int:
        """Remove the least significant set bit."""
        assert b != 0, "Bitboard is empty"
        return b & (b - 1)

    @staticmethod
    def count(b: int) -> int:
        """Count the number of set bits in the bitboard."""
        return bin(b).count('1')

    @staticmethod
    def single(b: int) -> bool:
        """Check if only a single bit is set in the bitboard."""
        assert b != 0, "Bitboard is empty"
        return (b & (b - 1)) == 0

    @staticmethod
    def file(fl: int) -> int:
        """Return a bitboard for the given file."""
        assert 0 <= fl < 8, "File index out of range"
        return 0xFF << (fl * 8)

    @staticmethod
    def rank(rk: int) -> int:
        """Return a bitboard for the given rank."""
        assert 0 <= rk < 8, "Rank index out of range"
        return 0x0101010101010101 << rk

    @staticmethod
    def files(fl: int) -> int:
        """Return a bitboard for the given file shifted left and right by one file."""
        assert 0 <= fl < 8, "File index out of range"
        file_bb = Bit.file(fl)
        return ((file_bb << 8) | file_bb | (file_bb >> 8)) & ((1 << 64) - 1)

    @staticmethod
    def left(fl: int) -> int:
        """Return the left bitboard for the given file."""
        assert 0 <= fl < 8, "File index out of range"
        return Bit.p_left[fl]

    @staticmethod
    def right(fl: int) -> int:
        """Return the right bitboard for the given file."""
        assert 0 <= fl < 8, "File index out of range"
        return Bit.p_right[fl]

    @staticmethod
    def front(rk: int) -> int:
        """Return the front bitboard for the given rank."""
        assert 0 <= rk < 8, "Rank index out of range"
        return Bit.p_front[rk]

    @staticmethod
    def rear(rk: int) -> int:
        """Return the rear bitboard for the given rank."""
        assert 0 <= rk < 8, "Rank index out of range"
        return Bit.p_rear[rk]

    @staticmethod
    def front_side(sq: int, sd: int) -> int:
        """Return the side-specific front bitboard for a square and side."""
        rk = Square.rank(sq)
        return Bit.p_side_front[sd][rk]

    @staticmethod
    def rear_side(sq: int, sd: int) -> int:
        """Return the side-specific rear bitboard for a square and side."""
        rk = Square.rank(sq)
        return Bit.p_side_rear[sd][rk]

    @staticmethod
    def init():
        """Initialize all bitboards."""
        # Initialize p_left and p_rear
        bf = 0
        br = 0
        for i in range(8):
            Bit.p_left[i] = bf
            Bit.p_rear[i] = br
            bf |= Bit.file(i)
            br |= Bit.rank(i)

        # Initialize p_right and p_front
        bf = 0
        br = 0
        for i in range(7, -1, -1):
            Bit.p_right[i] = bf
            Bit.p_front[i] = br
            bf |= Bit.file(i)
            br |= Bit.rank(i)

        # Initialize side-specific front and rear bitboards
        for rk in range(8):
            Bit.p_side_front[Side.WHITE][rk] = Bit.front(rk)
            Bit.p_side_front[Side.BLACK][rk] = Bit.rear(rk)
            Bit.p_side_rear[Side.WHITE][rk] = Bit.rear(rk)
            Bit.p_side_rear[Side.BLACK][rk] = Bit.front(rk)


class Hash:
    # Constants
    TURN = Piece.SIDE_SIZE * Square.SIZE
    CASTLE = TURN + 1
    EN_PASSANT = CASTLE + 4
    SIZE = EN_PASSANT + 8

    # Initialize random keys
    p_rand = [0 for _ in range(SIZE)]

    @staticmethod
    def rand_64() -> int:
        """Generate a random 64-bit integer."""
        rand = 0
        for _ in range(4):
            part = Util.rand_int(1 << 16)  # 16-bit random part
            rand = (rand << 16) | part
        return rand

    @staticmethod
    def rand_key(index: int) -> int:
        """Retrieve the random key for the given index."""
        assert 0 <= index < Hash.SIZE, "Hash index out of range"
        return Hash.p_rand[index]

    @staticmethod
    def piece_key(p12: int, sq: int) -> int:
        """Retrieve the Zobrist key for a specific piece at a specific square."""
        return Hash.rand_key(p12 * Square.SIZE + sq)

    @staticmethod
    def turn_key(turn: int) -> int:
        """Retrieve the Zobrist key for the current turn."""
        if turn == Side.WHITE:
            return 0
        else:
            return Hash.rand_key(Hash.TURN)

    @staticmethod
    def turn_flip() -> int:
        """Retrieve the Zobrist key to flip the turn."""
        return Hash.rand_key(Hash.TURN)

    @staticmethod
    def flag_key(flag: int) -> int:
        """Retrieve the Zobrist key for castling flags."""
        assert 0 <= flag < 4, "Castling flag out of range"
        return Hash.rand_key(Hash.CASTLE + flag)

    @staticmethod
    def en_passant_key(sq: int) -> int:
        """Retrieve the Zobrist key for en passant square."""
        if sq == Square.NONE:
            return 0
        else:
            return Hash.rand_key(Hash.EN_PASSANT + Square.file(sq))

    @staticmethod
    def index(key: int) -> int:
        """Convert hash key to int64."""
        return int(key)

    @staticmethod
    def lock(key: int) -> int:
        """Convert hash key to uint32 (upper 32 bits)."""
        return (key >> 32) & 0xFFFFFFFF
    
    @staticmethod
    def init():
        """Initialize the random keys."""
        for i in range(Hash.SIZE):
            Hash.p_rand[i] = Hash.rand_64()


class Castling:
    @dataclass
    class Info:
        kf: int  # King from square
        kt: int  # King to square
        rf: int  # Rook from square
        rt: int  # Rook to square

    # Castling information for White and Black, Kingside and Queenside
    info: List['Castling.Info'] = [
        Info(Square.E1, Square.G1, Square.H1, Square.F1),  # White Kingside
        Info(Square.E1, Square.C1, Square.A1, Square.D1),  # White Queenside
        Info(Square.E8, Square.G8, Square.H8, Square.F8),  # Black Kingside
        Info(Square.E8, Square.C8, Square.A8, Square.D8),  # Black Queenside
    ]

    # Flags mask for each square
    flags_mask: List[int] = [0 for _ in range(Square.SIZE)]

    # Zobrist hash keys for castling flags
    flags_key: List[int] = [0 for _ in range(1 << 4)]  # 16 possible flags

    @staticmethod
    def index(sd: int, wg: int) -> int:
        """
        Combine side and wing to get the castling index.
        sd: Side (0 for WHITE, 1 for BLACK)
        wg: Wing (0 for KING, 1 for QUEEN)
        """
        return sd * Wing.SIZE + wg

    @staticmethod
    def side(index: int) -> int:
        """Retrieve the side from the castling index."""
        return index // Wing.SIZE

    @staticmethod
    def flag(flags: int, index: int) -> bool:
        """Check if a specific castling flag is set."""
        assert index < 4, "Castling index out of range"
        return bool((flags >> index) & 1)

    @staticmethod
    def set_flag(flags: int, index: int) -> int:
        """Set a specific castling flag."""
        assert index < 4, "Castling index out of range"
        return flags | (1 << index)

    @staticmethod
    def flags_key_debug(flags: int) -> int:
        """
        Generate the Zobrist hash key for the given castling flags.
        Each set flag XORs the corresponding hash key.
        """
        key = 0
        for index in range(4):
            if Castling.flag(flags, index):
                key ^= Hash.flag_key(index)
        return key

    @classmethod
    def init(cls):
        """Initialize the flags_mask and flags_key arrays."""
        # Initialize flags_mask for relevant squares
        for sq in range(Square.SIZE):
            cls.flags_mask[sq] = 0

        # Set relevant flags for castling
        cls.flags_mask[Square.E1] = Castling.set_flag(cls.flags_mask[Square.E1], 0)
        cls.flags_mask[Square.E1] = Castling.set_flag(cls.flags_mask[Square.E1], 1)
        cls.flags_mask[Square.H1] = Castling.set_flag(cls.flags_mask[Square.H1], 0)
        cls.flags_mask[Square.A1] = Castling.set_flag(cls.flags_mask[Square.A1], 1)

        cls.flags_mask[Square.E8] = Castling.set_flag(cls.flags_mask[Square.E8], 2)
        cls.flags_mask[Square.E8] = Castling.set_flag(cls.flags_mask[Square.E8], 3)
        cls.flags_mask[Square.H8] = Castling.set_flag(cls.flags_mask[Square.H8], 2)
        cls.flags_mask[Square.A8] = Castling.set_flag(cls.flags_mask[Square.A8], 3)

        # Initialize flags_key for all possible flag combinations
        for flags in range(1 << 4):
            cls.flags_key[flags] = cls.flags_key_debug(flags)


class Stage:
    SIZE = 2  # Middle Game and End Game

    # Stage enumeration
    MG = 0  # Middle Game
    EG = 1  # End Game


class Material:
    # Phase values for each piece type
    PAWN_PHASE = 0
    KNIGHT_PHASE = 1
    BISHOP_PHASE = 1
    ROOK_PHASE = 2
    QUEEN_PHASE = 4

    # Total game phase
    TOTAL_PHASE = (
        PAWN_PHASE * 16 +
        KNIGHT_PHASE * 4 +
        BISHOP_PHASE * 4 +
        ROOK_PHASE * 4 +
        QUEEN_PHASE * 2
    )

    # Phase values indexed by piece type
    p_phase = [
        PAWN_PHASE,      # PAWN
        KNIGHT_PHASE,    # KNIGHT
        BISHOP_PHASE,    # BISHOP
        ROOK_PHASE,      # ROOK
        QUEEN_PHASE,     # QUEEN
        0,               # KING
        0                # NONE
    ]

    # Piece power for force evaluation
    p_power = [0, 1, 1, 2, 4, 0, 0]

    # Piece scores for different stages
    p_score = [
        [85, 95],    # PAWN
        [325, 325],  # KNIGHT
        [325, 325],  # BISHOP
        [460, 540],  # ROOK
        [975, 975],  # QUEEN
        [0, 0],      # KING
        [0, 0]       # NONE
    ]

    # Weight interpolation table
    phase_weight = [0] * (TOTAL_PHASE + 1)
    
    @staticmethod
    def phase(pc: int) -> int:
        """
        Get the phase value for a given piece type.
        pc: Piece type index
        """
        assert pc < Piece.SIZE, "Piece type out of range"
        return Material.p_phase[pc]

    @staticmethod
    def power(pc: int) -> int:
        """
        Get the power value for a given piece type.
        """
        assert 0 <= pc < len(Material.p_power), "Piece type out of range."
        return Material.p_power[pc]

    @staticmethod
    def score(pc: int, stage: int) -> int:
        """
        Get the score of a piece based on the current game stage.
        """
        assert 0 <= pc < Piece.SIZE, "Piece type out of range."
        assert 0 <= stage < Stage.SIZE, "Stage out of range."
        return Material.p_score[pc][stage]

    @staticmethod
    def force(sd: int, bd: 'Board.BOARD') -> int:
        """
        Calculate the force for draw evaluation based on material.
        """
        force_val = 0
        for pc in range(Piece.KNIGHT, Piece.QUEEN + 1):
            force_val += bd.count(pc, sd) * Material.power(pc)
        return force_val

    @staticmethod
    def lone_king(sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a side has only a lone king.
        """
        return (
            bd.count(Piece.KNIGHT, sd) == 0 and
            bd.count(Piece.BISHOP, sd) == 0 and
            bd.count(Piece.ROOK, sd) == 0 and
            bd.count(Piece.QUEEN, sd) == 0
        )

    @staticmethod
    def lone_bishop(sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a side has only a lone bishop.
        """
        return (
            bd.count(Piece.KNIGHT, sd) == 0 and
            bd.count(Piece.BISHOP, sd) == 1 and
            bd.count(Piece.ROOK, sd) == 0 and
            bd.count(Piece.QUEEN, sd) == 0
        )

    @staticmethod
    def lone_king_or_bishop(sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a side has only a lone king or a king and one bishop.
        """
        return (
            bd.count(Piece.KNIGHT, sd) == 0 and
            bd.count(Piece.BISHOP, sd) <= 1 and
            bd.count(Piece.ROOK, sd) == 0 and
            bd.count(Piece.QUEEN, sd) == 0
        )

    @staticmethod
    def lone_king_or_minor(sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a side has only a lone king or a king and one minor piece.
        """
        return (
            bd.count(Piece.KNIGHT, sd) + bd.count(Piece.BISHOP, sd) <= 1 and
            bd.count(Piece.ROOK, sd) == 0 and
            bd.count(Piece.QUEEN, sd) == 0
        )

    @staticmethod
    def two_knights(sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a side has exactly two knights and no other pieces.
        """
        return (
            bd.count(Piece.KNIGHT, sd) == 2 and
            bd.count(Piece.BISHOP, sd) == 0 and
            bd.count(Piece.ROOK, sd) == 0 and
            bd.count(Piece.QUEEN, sd) == 0
        )

    @staticmethod
    def interpolation(mg: int, eg: int, bd: 'Board.BOARD') -> int:
        """
        Interpolate between middle game (mg) and end game (eg) scores based on game phase.
        """
        phase = min(bd.phase(), Material.TOTAL_PHASE)
        assert 0 <= phase <= Material.TOTAL_PHASE, "Phase out of range."

        weight = Material.phase_weight[phase]
        return (mg * weight + eg * (256 - weight) + 128) >> 8

    @staticmethod
    def init():
        """
        Initialize the phase_weight table using a sigmoid function.
        """
        for i in range(Material.TOTAL_PHASE + 1):
            x = float(i) / (Material.TOTAL_PHASE / 2) - 1.0
            y = 1.0 / (1.0 + math.exp(-x * 5.0))
            Material.phase_weight[i] = Util.round(y * 256.0)


class Board:
    @dataclass
    class Copy:
        key: int = 0
        pawn_key: int = 0
        flags: int = 0
        ep_sq: int = 0
        moves: int = 0
        recap: int = 0
        phase: int = 0

    @dataclass
    class Undo:
        copy: 'Board.Copy' = field(default_factory=lambda: Board.Copy())
        move: int = 0
        cap_sq: int = 0
        castling: bool = False

    start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

    class BOARD:
        # # Constants
        # SCORE_NONE = -10000
    
        def __init__(self):
            # Bitboards for each piece type
            self.p_piece: List[int] = [0 for _ in range(Piece.SIZE)]
            # Bitboards for each side
            self.p_side: List[int] = [0 for _ in range(Side.SIZE)]
            # Combined bitboard of all pieces
            self.p_all: int = 0
    
            # King positions for each side
            self.p_king: List[int] = [Square.NONE for _ in range(Side.SIZE)]
            # Count of each side-specific piece
            self.p_count: List[int] = [0 for _ in range(Piece.SIDE_SIZE)]
    
            # Piece on each square
            self.p_square: List[int] = [Piece.NONE for _ in range(Square.SIZE)]
            # Current turn (0 for WHITE, 1 for BLACK)
            self.p_turn: int = Side.WHITE
    
            # Current game state
            self.p_copy: 'Board.Copy' = Board.Copy()
    
            # Move stack for undo operations
            self.p_root: int = 0
            self.p_sp: int = 0  # Stack pointer
            self.p_stack: List['Board.Undo'] = [Board.Undo() for _ in range(1024)]
        
        def assign(self, bd: 'Board.BOARD'):
            for pc in range(Piece.SIZE):
                self.p_piece[pc] = bd.p_piece[pc]
        
            for sd in range(Side.SIZE):
                self.p_side[sd] = bd.p_side[sd]
                self.p_king[sd] = bd.p_king[sd]
        
            self.p_all = bd.p_all
        
            for p12 in range(Piece.SIDE_SIZE):
                self.p_count[p12] = bd.p_count[p12]
        
            for sq in range(Square.SIZE):
                self.p_square[sq] = bd.p_square[sq]
        
            self.p_turn = bd.p_turn
            self.p_copy = copy.deepcopy(bd.p_copy)
        
            self.p_root = bd.p_root
            self.p_sp = bd.p_sp
        
            for sp in range(bd.p_sp):
                self.p_stack[sp] = copy.deepcopy(bd.p_stack[sp])
        
            assert self.moves() == bd.moves()
        
        def piece(self, pc: int, sd: int = None) -> int:
            """Return the bitboard for a given piece type, and optionally for a specific side."""
            assert pc < Piece.SIZE, "Piece type out of range"
            assert pc != Piece.NONE, "NONE piece has no bitboard"
            if sd is None:
                return self.p_piece[pc]
            else:
                return self.p_piece[pc] & self.p_side[sd]

        def count(self, pc: int, sd: int) -> int:
            """Return the count of a specific piece type and side."""
            assert pc < Piece.SIZE, "Piece type out of range"
            assert pc != Piece.NONE, "NONE piece has no count"
            return self.p_count[Piece.make(pc, sd)]
    
        def side(self, sd: int) -> int:
            """Return the bitboard for a given side."""
            return self.p_side[sd]
    
        def pieces(self, sd: int) -> int:
            """Return the bitboard of all pieces for a given side, excluding pawns."""
            assert self.p_side[sd] & ~self.piece(Piece.PAWN, sd) == self.p_side[sd] & ~self.piece(Piece.PAWN, sd) & 0xFFFFFFFFFFFFFFFF
            return self.p_side[sd] & ~self.piece(Piece.PAWN, sd)
    
        def all_pieces(self) -> int:
            """Return the bitboard of all pieces on the board."""
            return self.p_all
    
        def empty(self) -> int:
            """Return the bitboard of all empty squares."""
            return ~self.p_all & 0xFFFFFFFFFFFFFFFF
    
        def square(self, sq: int) -> int:
            """Return the piece type on a given square."""
            return self.p_square[sq]
    
        def square_side(self, sq: int) -> int:
            """Return the side occupying a given square."""
            assert self.p_square[sq] != Piece.NONE, "Square is empty"
            return (self.p_side[Side.BLACK] >> sq) & 1  # Assumes side::BLACK is at index 1
    
        def square_is(self, sq: int, pc: int, sd: int) -> bool:
            """Check if a square has a specific piece and side."""
            assert pc < Piece.SIZE, "Piece type out of range"
            assert pc != Piece.NONE, "NONE piece has no square"
            
            if not (0 <= sq < 64):
                return False
            
            return self.p_square[sq] == pc and self.square_side(sq) == sd
    
        def king(self, sd: int) -> int:
            """Return the king's square for a given side."""
            sq = self.p_king[sd]
            assert sq == Bit.first(self.piece(Piece.KING, sd)), "King position mismatch"
            return sq
    
        def turn(self) -> int:
            """Return the current turn."""
            return self.p_turn
    
        def key(self) -> int:
            """Return the current Zobrist key."""
            key = self.p_copy.key
            key ^= Castling.flags_key[self.p_copy.flags]
            key ^= Hash.en_passant_key(self.p_copy.ep_sq)
            return key
    
        def pawn_key(self) -> int:
            """Return the current pawn Zobrist key."""
            return self.p_copy.pawn_key
    
        def eval_key(self) -> int:
            """Return the evaluation Zobrist key."""
            key = self.p_copy.key
            key ^= Hash.turn_key(self.p_turn)
            key ^= Castling.flags_key[self.p_copy.flags]
            return key
    
        def flags(self) -> int:
            """Return the current castling flags."""
            return self.p_copy.flags
    
        def ep_sq(self) -> int:
            """Return the current en passant square."""
            return self.p_copy.ep_sq
    
        def moves(self) -> int:
            """Return the number of moves made."""
            return self.p_copy.moves
    
        def recap(self) -> int:
            """Return the last recaptured square."""
            return self.p_copy.recap
    
        def phase(self) -> int:
            """Return the current game phase."""
            return self.p_copy.phase
    
        def ply(self) -> int:
            """Return the current ply (half-move) count."""
            assert self.p_sp >= self.p_root, "Stack pointer below root"
            return self.p_sp - self.p_root
    
        def last_move(self) -> int:
            """Return the last move made."""
            return Move.NONE if self.p_sp == 0 else self.p_stack[self.p_sp - 1].move
    
        def is_draw(self) -> bool:
            """Determine if the game is a draw."""
            if self.p_copy.moves > 100:
                return True  # Fifty-move rule (needs actual implementation)
    
            key = self.p_copy.key
    
            assert self.p_copy.moves <= self.p_sp, "Move count exceeds stack pointer"
    
            # Detect repetition
            for i in range(4, self.p_copy.moves, 2):
                if self.p_stack[self.p_sp - i].copy.key == key:
                    return True
    
            return False
    
        def set_root(self):
            """Set the current stack pointer as the root."""
            self.p_root = self.p_sp
    
        def clear(self):
            """Clear the board to an empty state."""
            for pc in range(Piece.SIZE):
                self.p_piece[pc] = 0
    
            for sd in range(Side.SIZE):
                self.p_side[sd] = 0
    
            for sq in range(Square.SIZE):
                self.p_square[sq] = Piece.NONE
    
            for sd in range(Side.SIZE):
                self.p_king[sd] = Square.NONE
    
            for p12 in range(Piece.SIDE_SIZE):
                self.p_count[p12] = 0
    
            self.p_turn = Side.WHITE
            self.p_copy.key = 0
            self.p_copy.pawn_key = 0
            self.p_copy.flags = 0
            self.p_copy.ep_sq = Square.NONE
            self.p_copy.moves = 0
            self.p_copy.recap = Square.NONE
            self.p_copy.phase = 0
            self.p_root = 0
            self.p_sp = 0
    
        def clear_square(self, pc: int, sd: int, sq: int, update_copy: bool):
            """
            Remove a piece from a square.
            pc: Piece type
            sd: Side
            sq: Square index
            update_copy: Whether to update the game state copy
            """
            assert pc < Piece.SIZE, "Piece type out of range"
            assert pc != Piece.NONE, "Cannot clear NONE piece"
            assert 0 <= sq < Square.SIZE, "Square index out of range"
            assert pc == self.p_square[sq], "Piece mismatch on square"
    
            assert Bit.is_set(self.p_piece[pc], sq), "Piece bitboard mismatch"
            self.p_piece[pc] = Bit.clear_bit(self.p_piece[pc], sq)
    
            assert Bit.is_set(self.p_side[sd], sq), "Side bitboard mismatch"
            self.p_side[sd] = Bit.clear_bit(self.p_side[sd], sq)
    
            assert self.p_square[sq] != Piece.NONE, "Square already empty"
            self.p_square[sq] = Piece.NONE
    
            p12 = Piece.make(pc, sd)
            assert self.p_count[p12] > 0, "Piece count underflow"
            self.p_count[p12] -= 1
    
            if update_copy:
                key = Hash.piece_key(p12, sq)
                self.p_copy.key ^= key
                if pc == Piece.PAWN:
                    self.p_copy.pawn_key ^= key
    
                self.p_copy.phase -= Material.phase(pc)
    
        def set_square(self, pc: int, sd: int, sq: int, update_copy: bool):
            """
            Place a piece on a square.
            pc: Piece type
            sd: Side
            sq: Square index
            update_copy: Whether to update the game state copy
            """
            assert pc < Piece.SIZE, "Piece type out of range"
            assert pc != Piece.NONE, "Cannot set NONE piece"
            assert 0 <= sq < Square.SIZE, "Square index out of range"
            assert self.p_square[sq] == Piece.NONE, "Square already occupied"
    
            assert not Bit.is_set(self.p_piece[pc], sq), "Piece already set on bitboard"
            self.p_piece[pc] = Bit.set_bit(self.p_piece[pc], sq)
    
            assert not Bit.is_set(self.p_side[sd], sq), "Side already set on bitboard"
            self.p_side[sd] = Bit.set_bit(self.p_side[sd], sq)
    
            self.p_square[sq] = pc
    
            if pc == Piece.KING:
                self.p_king[sd] = sq
    
            p12 = Piece.make(pc, sd)
            self.p_count[p12] += 1
    
            if update_copy:
                key = Hash.piece_key(p12, sq)
                self.p_copy.key ^= key
                if pc == Piece.PAWN:
                    self.p_copy.pawn_key ^= key
    
                self.p_copy.phase += Material.phase(pc)
    
        def move_square(self, pc: int, sd: int, f: int, t: int, update_copy: bool):
            """
            Move a piece from one square to another.
            pc: Piece type
            sd: Side
            f: From square
            t: To square
            update_copy: Whether to update the game state copy
            """
            self.clear_square(pc, sd, f, update_copy)
            self.set_square(pc, sd, t, update_copy)
    
        def flip_turn(self):
            """Switch the turn to the opposite side."""
            self.p_turn = Side.opposit(self.p_turn)
            self.p_copy.key ^= Hash.turn_flip()
    
        def update(self):
            """
            Update the combined bitboards and validate the board state.
            Debug assertions are included if necessary.
            """
            self.p_all = self.p_side[Side.WHITE] | self.p_side[Side.BLACK]
    
            # Debugging assertions can be added here if needed
    
        def can_castle(self, index: int) -> bool:
            """
            Check if castling is possible for a given castling index.
            index: Castling index (0-3)
            """
            sd = Castling.side(index)
            return (self.square_is(Castling.info[index].kf, Piece.KING, sd) and
                    self.square_is(Castling.info[index].rf, Piece.ROOK, sd))
    
        def pawn_is_attacked(self, sq: int, sd: int) -> bool:
            """
            Check if a pawn is attacking a given square.
            sq: Square to check
            sd: Side of the attacking pawn
            """
            fl = Square.file(sq)
            sq -= Square.pawn_inc(sd)
    
            return ((fl != Square.FILE_A and self.square_is(sq + Square.INC_LEFT, Piece.PAWN, sd)) or
                    (fl != Square.FILE_H and self.square_is(sq + Square.INC_RIGHT, Piece.PAWN, sd)))
    
        def init_fen(self, fen: str):
            """
            Initialize the board from a FEN string.
            fen: FEN string
            """
            self.clear()
            pos = 0
            sq = 0
            length = len(fen)
    
            # Piece placement
            while pos < length:
                c = fen[pos]
                pos += 1
    
                if c == ' ':
                    break
                elif c == '/':
                    continue
                elif c.isdigit():
                    sq += int(c)
                else:
                    p12 = Piece.from_fen(c)
                    pc = Piece.piece(p12)
                    sd = Piece.side(p12)
                    self.set_square(pc, sd, Square.from_fen(sq), True)
                    sq += 1
    
            assert sq == Square.SIZE, "FEN piece placement does not cover all squares"
    
            # Turn
            if pos < length:
                c = fen[pos]
                pos += 1
                self.p_turn = Util.string_find("wb", c)
                if pos < length and fen[pos] == ' ':
                    pos += 1
    
            self.p_copy.key ^= Hash.turn_key(self.p_turn)
    
            # Castling flags
            self.p_copy.flags = 0
            if pos < length:
                while pos < length:
                    c = fen[pos]
                    pos += 1
                    if c == ' ':
                        break
                    if c == '-':
                        continue
    
                    index = Util.string_find("KQkq", c)
                    if index != -1 and self.can_castle(index):
                        self.p_copy.flags = Castling.set_flag(self.p_copy.flags, index)
    
            # En-passant square
            self.p_copy.ep_sq = Square.NONE
            if pos < length:
                ep_string = ""
                while pos < length:
                    c = fen[pos]
                    pos += 1
                    if c == ' ':
                        break
                    ep_string += c
    
                if ep_string != "-":
                    sq = Square.from_string(ep_string)
                    if self.pawn_is_attacked(sq, self.p_turn):
                        self.p_copy.ep_sq = sq
    
            self.update()
    
        def move(self, mv: int):
            """
            Execute a move on the board.
            mv: Encoded move integer
            """
            assert mv != Move.NONE and mv != Move.NULL_, "Invalid move"
    
            sd = self.p_turn
            xd = Side.opposit(sd)
    
            f = Move.from_sq(mv)
            t = Move.to_sq(mv)
    
            pc = Move.piece(mv)
            cp = Move.cap(mv)
            pp = Move.prom(mv)
    
            assert self.p_square[f] == pc, "Piece mismatch on from-square"
            assert self.square_side(f) == sd, "Side mismatch on from-square"
    
            assert self.p_sp < len(self.p_stack), "Move stack overflow"
            undo = self.p_stack[self.p_sp]
            self.p_sp += 1
            
            undo.copy = copy.deepcopy(self.p_copy)
            undo.move = mv
            undo.castling = False
    
            self.p_copy.moves += 1
            self.p_copy.recap = Square.NONE
    
            # Handle captures
            assert cp != Piece.KING, "Cannot capture the king"
    
            if pc == Piece.PAWN and t == self.p_copy.ep_sq:
                cap_sq = t - Square.pawn_inc(sd)
                assert self.p_square[cap_sq] == cp, "Invalid en-passant capture"
                assert cp == Piece.PAWN, "En-passant must capture a pawn"
    
                undo.cap_sq = cap_sq
                self.clear_square(cp, xd, cap_sq, True)
            elif cp != Piece.NONE:
                assert self.p_square[t] == cp, "Captured piece mismatch"
                assert self.square_side(t) == xd, "Captured piece side mismatch"
    
                undo.cap_sq = t
                self.clear_square(cp, xd, t, True)
            else:
                assert self.p_square[t] == cp, "Non-capture move inconsistency"
    
            # Handle promotion
            if pp != Piece.NONE:
                assert pc == Piece.PAWN, "Only pawns can promote"
                self.clear_square(Piece.PAWN, sd, f, True)
                self.set_square(pp, sd, t, True)
            else:
                self.move_square(pc, sd, f, t, True)
    
            # Handle castling rook movement
            if pc == Piece.KING and abs(t - f) == Square.CASTLING_DELTA:
                undo.castling = True
    
                wg = Wing.KING if t > f else Wing.QUEEN
                index = Castling.index(sd, wg)
    
                assert Castling.flag(self.p_copy.flags, index), "Castling flag not set"
    
                assert f == Castling.info[index].kf, "King from-square mismatch"
                assert t == Castling.info[index].kt, "King to-square mismatch"
    
                self.move_square(Piece.ROOK, sd, Castling.info[index].rf, Castling.info[index].rt, True)
    
            # Switch turn
            self.flip_turn()
    
            # Update castling flags
            assert self.p_copy.flags & ~Castling.flags_mask[f] == self.p_copy.flags & ~Castling.flags_mask[f] & 0xFFFFFFFFFFFFFFFF
            self.p_copy.flags &= ~Castling.flags_mask[f]
            
            assert self.p_copy.flags & ~Castling.flags_mask[t] == self.p_copy.flags & ~Castling.flags_mask[t] & 0xFFFFFFFFFFFFFFFF
            self.p_copy.flags &= ~Castling.flags_mask[t]
    
            # Update en-passant square
            self.p_copy.ep_sq = Square.NONE
    
            if pc == Piece.PAWN and abs(t - f) == Square.DOUBLE_PAWN_DELTA:
                sq = (f + t) // 2
                if self.pawn_is_attacked(sq, xd):
                    self.p_copy.ep_sq = sq
    
            # Reset move counter if capture or promotion occurred
            if cp != Piece.NONE or pc == Piece.PAWN:
                self.p_copy.moves = 0
    
            # Set recapture square if capture or promotion occurred
            if cp != Piece.NONE or pp != Piece.NONE:
                self.p_copy.recap = t
    
            self.update()
    
        def undo(self):
            """Undo the last move."""
            assert self.p_sp > 0, "No moves to undo"
    
            self.p_sp -= 1
            undo = self.p_stack[self.p_sp]
    
            mv = undo.move
            f = Move.from_sq(mv)
            t = Move.to_sq(mv)
    
            pc = Move.piece(mv)
            cp = Move.cap(mv)
            pp = Move.prom(mv)
    
            xd = self.p_turn
            sd = Side.opposit(xd)
    
            assert self.p_square[t] in (pc, pp), "Undo move inconsistency"
            assert self.square_side(t) == sd, "Undo move side inconsistency"
    
            # Handle castling rook movement
            if undo.castling:
                wg = Wing.KING if t > f else Wing.QUEEN
                index = Castling.index(sd, wg)
    
                assert f == Castling.info[index].kf, "Undo castling king from-square mismatch"
                assert t == Castling.info[index].kt, "Undo castling king to-square mismatch"
    
                self.move_square(Piece.ROOK, sd, Castling.info[index].rt, Castling.info[index].rf, False)
    
            # Handle promotion
            if pp != Piece.NONE:
                assert pc == Piece.PAWN, "Undo promotion for non-pawn piece"
                self.clear_square(pp, sd, t, False)
                self.set_square(Piece.PAWN, sd, f, False)
            else:
                self.move_square(pc, sd, t, f, False)
    
            # Handle captures
            if cp != Piece.NONE:
                self.set_square(cp, xd, undo.cap_sq, False)
    
            # Switch turn back
            self.flip_turn()
            self.p_copy = copy.deepcopy(undo.copy)
    
            self.update()
    
        def move_null(self):
            """
            Make a null move (pass).
            """
            assert self.p_sp < len(self.p_stack), "Move stack overflow"
    
            undo = self.p_stack[self.p_sp]
            self.p_sp += 1
            
            undo.move = Move.NULL_
            undo.copy = copy.deepcopy(self.p_copy)
            
            self.flip_turn()
            self.p_copy.ep_sq = Square.NONE
            self.p_copy.moves = 0
            self.p_copy.recap = Square.NONE
    
            self.update()
    
        def undo_null(self):
            """Undo a null move."""
            assert self.p_sp > 0, "No null moves to undo"
    
            self.p_sp -= 1
            undo = self.p_stack[self.p_sp]
    
            assert undo.move == Move.NULL_, "Last move was not a null move"
    
            self.flip_turn()
            self.p_copy = copy.deepcopy(undo.copy)
    
            self.update()


class Attack:
    @dataclass
    class Attacks:
        size: int = 0
        square: List[int] = field(default_factory=lambda: [0, 0])  # Up to 2 attack squares
        avoid: int = 0
        pinned: int = 0

    # Define constants
    Pawn_Move = [1, -1]  # [WHITE, BLACK]
    Pawn_Attack = [
        [-15, +17],  # WHITE
        [-17, +15],  # BLACK
    ]
    
    Knight_Inc = [-33, -31, -18, -14, +14, +18, +31, +33, 0]
    Bishop_Inc = [-17, -15, +15, +17, 0]
    Rook_Inc = [-16, -1, +1, +16, 0]
    Queen_Inc = [-17, -16, -15, -1, +1, +15, +16, +17, 0]
    
    # Piece_Inc is a list of lists, indexed by piece type
    Piece_Inc = [
        None,         # PAWN has no increments here
        Knight_Inc,   # KNIGHT
        Bishop_Inc,   # BISHOP
        Rook_Inc,     # ROOK
        Queen_Inc,    # QUEEN
        Queen_Inc,    # KING (using Queen increments for consistency)
        None           # NONE
    ]
    
    # Initialize bitboards as class variables
    Pawn_Moves: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Side.SIZE)]
    Pawn_Attacks: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Side.SIZE)]
    Piece_Attacks: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Piece.SIZE)]
    Blockers: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Piece.SIZE)]
    
    Between: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Square.SIZE)]
    Behind: List[List[int]] = [[0 for _ in range(Square.SIZE)] for _ in range(Square.SIZE)]
    
    @staticmethod
    def line_is_empty(f: int, t: int, bd: 'Board.BOARD') -> bool:
        """Check if the line between square f and square t is empty."""
        return (bd.all_pieces() & Attack.Between[f][t]) == 0
    
    @staticmethod
    def ray(f: int, t: int) -> int:
        """Return the bitboard representing the ray between f and t, including t."""
        return Attack.Between[f][t] | Attack.Behind[f][t]  # t should be included
    
    @staticmethod
    def pawn_move(sd: int, f: int, t: int, bd: 'Board.BOARD') -> bool:
        """Check if a pawn can move from f to t."""
        assert sd < Side.SIZE, "Invalid side"
        return Bit.is_set(Attack.Pawn_Moves[sd][f], t) and Attack.line_is_empty(f, t, bd)
    
    @staticmethod
    def pawn_attack(sd: int, f: int, t: int) -> bool:
        """Check if a pawn can attack square t from square f."""
        assert sd < Side.SIZE, "Invalid side"
        return Bit.is_set(Attack.Pawn_Attacks[sd][f], t)
    
    @staticmethod
    def piece_attack(pc: int, f: int, t: int, bd: 'Board.BOARD') -> bool:
        """Check if a non-pawn piece can attack square t from square f."""
        assert pc != Piece.PAWN, "Pawn does not use piece_attack"
        return Bit.is_set(Attack.Piece_Attacks[pc][f], t) and Attack.line_is_empty(f, t, bd)
    
    @staticmethod
    def attack(pc: int, sd: int, f: int, t: int, bd: 'Board.BOARD') -> bool:
        """
        General attack checker.
        pc: Piece type
        sd: Side of the attacking piece
        f: From square
        t: To square
        bd: Board state
        """
        assert sd < Side.SIZE, "Invalid side"
        if pc == Piece.PAWN:
            return Attack.pawn_attack(sd, f, t)
        else:
            return Attack.piece_attack(pc, f, t, bd)
    
    @staticmethod
    def pawn_moves_from(sd: int, bd: 'Board.BOARD') -> int:
        """Generate all squares to which pawns of side sd can move forward."""
        assert sd < Side.SIZE, "Invalid side"
        fs = bd.piece(Piece.PAWN, sd)
        if sd == Side.WHITE:
            return fs << 1
        else:
            return fs >> 1
    
    @staticmethod
    def pawn_moves_to(sd: int, ts: int, bd: 'Board.BOARD') -> int:
        """
        Given target squares ts, compute pawn moves that reach ts.
        sd: Side of the pawns
        ts: Bitboard of target squares
        bd: Board state
        """
        assert sd < Side.SIZE, "Invalid side"
        assert (bd.all_pieces() & ts) == 0, "Target squares must be empty"
        pawns = bd.piece(Piece.PAWN, sd)
        empty = bd.empty()
        fs = 0
        if sd == Side.WHITE:
            fs |= (ts >> 1)
            fs |= (ts >> 2) & (empty >> 1) & Bit.rank(Square.RANK_2)
        else:
            fs |= (ts << 1)
            fs |= (ts << 2) & (empty << 1) & Bit.rank(Square.RANK_7)
        return pawns & fs & ((1 << 64) - 1)
    
    @staticmethod
    def pawn_attacks_from(sd: int, bd: 'Board.BOARD') -> int:
        """Generate all squares attacked by pawns of side sd."""
        assert sd < Side.SIZE, "Invalid side"
        fs = bd.piece(Piece.PAWN, sd)
        if sd == Side.WHITE:
            return ((fs >> 7) | (fs << 9)) & 0xFFFFFFFFFFFFFFFF
        else:
            return ((fs >> 9) | (fs << 7)) & 0xFFFFFFFFFFFFFFFF
    
    @staticmethod
    def pawn_attacks_tos(sd: int, ts: int) -> int:
        """
        Given target squares ts, compute squares from which pawns of side sd can attack ts.
        sd: Side of the pawns
        ts: Bitboard of target squares
        """
        assert sd < Side.SIZE, "Invalid side"
        if sd == Side.WHITE:
            return ((ts >> 9) | (ts << 7)) & 0xFFFFFFFFFFFFFFFF
        else:
            return ((ts >> 7) | (ts << 9)) & 0xFFFFFFFFFFFFFFFF
    
    @staticmethod
    def pawn_attacks_from_single(sd: int, f: int) -> int:
        """
        Given a side and a from square, return the pawn attacks bitboard.
        sd: Side of the pawn
        f: From square index
        """
        assert sd < Side.SIZE, "Invalid side"
        return Attack.Pawn_Attacks[sd][f]
    
    @staticmethod
    def pawn_attacks_to(sd: int, t: int) -> int:
        """
        Given a side and a to square, return the pawn attacks bitboard.
        sd: Side of the pawn
        t: To square index
        """
        assert sd < Side.SIZE, "Invalid side"
        return Attack.pawn_attacks_from_single(Side.opposit(sd), t)
    
    @staticmethod
    def piece_attacks_from(pc: int, f: int, bd: 'Board.BOARD') -> int:
        """
        Generate all squares a non-pawn piece can attack from square f.
        pc: Piece type
        f: From square index
        bd: Board state
        """
        assert pc != Piece.PAWN, "Pawn does not use piece_attacks_from"
        
        ts = Attack.Piece_Attacks[pc][f]
        
        b = bd.all_pieces() & Attack.Blockers[pc][f]
        while b != 0:
            sq = Bit.first(b)
            assert ts & ~Attack.Behind[f][sq] == ts & ~Attack.Behind[f][sq] & 0xFFFFFFFFFFFFFFFF
            ts &= ~Attack.Behind[f][sq]
            b = Bit.rest(b)
            
        return ts
    
    @staticmethod
    def piece_attacks_to(pc: int, t: int, bd: 'Board.BOARD') -> int:
        """
        Generate all squares a non-pawn piece can attack to square t.
        pc: Piece type
        t: To square index
        bd: Board state
        """
        assert pc != Piece.PAWN, "Pawn does not use piece_attacks_to"
        return Attack.piece_attacks_from(pc, t, bd)
    
    @staticmethod
    def piece_moves_from(pc: int, sd: int, f: int, bd: 'Board.BOARD') -> int:
        """
        Generate all squares a piece can move from square f.
        pc: Piece type
        sd: Side of the piece
        f: From square index
        bd: Board state
        """
        if pc == Piece.PAWN:
            # raise NotImplementedError("Pawn moves are handled separately")
            assert False # TODO: blockers
            return Attack.Pawn_Attacks[sd][f]
        else:
            return Attack.piece_attacks_from(pc, f, bd)
    
    @staticmethod
    def attacks_from(pc: int, sd: int, f: int, bd: 'Board.BOARD') -> int:
        """
        Generate all squares a piece can attack from square f.
        pc: Piece type
        sd: Side of the piece
        f: From square index
        bd: Board state
        """
        if pc == Piece.PAWN:
            return Attack.Pawn_Attacks[sd][f]
        else:
            return Attack.piece_attacks_from(pc, f, bd)
    
    @staticmethod
    def attacks_to(pc: int, sd: int, t: int, bd: 'Board.BOARD') -> int:
        """
        Generate all squares a piece can attack to square t.
        pc: Piece type
        sd: Side of the piece
        t: To square index
        bd: Board state
        """
        return Attack.attacks_from(pc, Side.opposit(sd), t, bd)  # HACK for pawns
    
    @staticmethod
    def pseudo_attacks_from(pc: int, sd: int, f: int) -> int:
        """
        Generate pseudo-attacks from square f by a piece.
        pc: Piece type
        sd: Side of the piece
        f: From square index
        """
        if pc == Piece.PAWN:
            return Attack.Pawn_Attacks[sd][f]
        else:
            return Attack.Piece_Attacks[pc][f]
    
    @staticmethod
    def pseudo_attacks_to(pc: int, sd: int, t: int) -> int:
        """
        Generate pseudo-attacks to square t by a piece.
        pc: Piece type
        sd: Side of the piece
        t: To square index
        """
        return Attack.pseudo_attacks_from(pc, Side.opposit(sd), t)  # HACK for pawns
    
    @staticmethod
    def slider_pseudo_attacks_to(sd: int, t: int, bd: 'Board.BOARD') -> int:
        """
        Generate pseudo-attacks by sliders (bishop, rook, queen) to square t.
        sd: Side of the attacking pieces
        t: To square index
        bd: Board state
        """
        assert sd < Side.SIZE, "Invalid side"
        b = 0
        b |= bd.piece(Piece.BISHOP, sd) & Attack.Piece_Attacks[Piece.BISHOP][t]
        b |= bd.piece(Piece.ROOK, sd) & Attack.Piece_Attacks[Piece.ROOK][t]
        b |= bd.piece(Piece.QUEEN, sd) & Attack.Piece_Attacks[Piece.QUEEN][t]
        return b
    
    @staticmethod
    def attack_behind(f: int, t: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if there is an attack behind square t from square f by side sd.
        f: From square index
        t: To square index
        sd: Side of the attacking pieces
        bd: Board state
        """
        assert bd.square(t) != Piece.NONE, "Target square must have a piece"
        
        behind = Attack.Behind[f][t]
        if behind == 0:
            return False
        
        b = Attack.slider_pseudo_attacks_to(sd, t, bd) & behind
        while b != 0:
            sq = Bit.first(b)
            if Bit.single(bd.all_pieces() & Attack.Between[sq][f]):
                return True
            b = Bit.rest(b)
    
        return False
    
    @staticmethod
    def is_attacked(t: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if square t is attacked by side sd.
        t: Target square index
        sd: Side of the attacking pieces
        bd: Board state
        """
        # Non-sliders
        if (bd.piece(Piece.PAWN, sd) & Attack.Pawn_Attacks[Side.opposit(sd)][t]) != 0:
            return True
        if (bd.piece(Piece.KNIGHT, sd) & Attack.Piece_Attacks[Piece.KNIGHT][t]) != 0:
            return True
        if (bd.piece(Piece.KING, sd) & Attack.Piece_Attacks[Piece.KING][t]) != 0:
            return True
        
        # Sliders
        b = Attack.slider_pseudo_attacks_to(sd, t, bd)
        while b != 0:
            f = Bit.first(b)
            if (bd.all_pieces() & Attack.Between[f][t]) == 0:
                return True
            b = Bit.rest(b)

        return False
    
    @staticmethod
    def pinned_by(t: int, sd: int, bd: 'Board.BOARD') -> int:
        """
        Determine which pieces are pinning the piece on square t.
        t: Target square index
        sd: Side of the defending piece
        bd: Board state
        """
        pinned = 0
        
        b = Attack.slider_pseudo_attacks_to(sd, t, bd)
        while b != 0:
            f = Bit.first(b)
            
            bb = bd.all_pieces() & Attack.Between[f][t]
            if bb != 0 and Bit.single(bb):
                pinned |= bb
            b = Bit.rest(b)
            
        return pinned
    
    @staticmethod
    def init_attacks(attacks: 'Attack.Attacks', sd: int, bd: 'Board.BOARD'):
        """
        Initialize attack information for strictly-legal moves.
        attacks: Attacks dataclass instance to populate
        sd: Side of the defending piece
        bd: Board state
        """
        atk = Side.opposit(sd)
        defn = sd
        
        t = bd.king(defn)
        
        attacks.size = 0
        attacks.avoid = 0
        attacks.pinned = 0
        
        # Non-sliders
        b = 0
        b |= bd.piece(Piece.PAWN, atk) & Attack.Pawn_Attacks[defn][t]  # HACK
        b |= bd.piece(Piece.KNIGHT, atk) & Attack.Piece_Attacks[Piece.KNIGHT][t]
        
        if b != 0:
            assert Bit.single(b), "Multiple non-slider attackers detected"
            assert attacks.size < 2, "Too many attack squares"
            attacks.square[attacks.size] = Bit.first(b)
            attacks.size += 1
        
        # Sliders
        b = Attack.slider_pseudo_attacks_to(atk, t, bd)
        while b != 0:
            f = Bit.first(b)
            
            bb = bd.all_pieces() & Attack.Between[f][t]

            if bb == 0:
                assert attacks.size < 2
                attacks.square[attacks.size] = f
                attacks.size += 1
                attacks.avoid |= Attack.ray(f, t)
            elif Bit.single(bb):
                attacks.pinned |= bb

            b = Bit.rest(b)
    
    @staticmethod
    def is_legal(bd: 'Board.BOARD') -> bool:
        """
        Determine if the current board state is legal (i.e., not in check).
        bd: Board state
        """
        atk = bd.turn()
        defn = Side.opposit(atk)
        return not Attack.is_attacked(bd.king(defn), atk, bd)
    
    @staticmethod
    def is_in_check(bd: 'Board.BOARD') -> bool:
        """
        Determine if the current player is in check.
        bd: Board state
        """
        atk = bd.turn()
        defn = Side.opposit(atk)
        return Attack.is_attacked(bd.king(atk), defn, bd)
    
    @staticmethod
    def pawn_moves_debug(sd: int, sq: int) -> int:
        """
        Debug function to compute pawn moves from square sq for side sd.
        sd: Side of the pawn
        sq: From square index
        """
        assert sd < Side.SIZE, "Invalid side"
        b = 0
        
        f = Square.to_88(sq)
        inc = Attack.Pawn_Move[sd]
        
        t = f + inc
        
        if Square.is_valid_88(t):
            b = Bit.set_bit(b, Square.from_88(t))
            
        if Square.rank(sq, sd) == Square.RANK_2:
            t += inc
            assert Square.is_valid_88(t), "Invalid target square in pawn_moves_debug"
            b = Bit.set_bit(b, Square.from_88(t))
            
        return b
    
    @staticmethod
    def pawn_attacks_debug(sd: int, sq: int) -> int:
        """
        Debug function to compute pawn attacks from square sq for side sd.
        sd: Side of the pawn
        sq: From square index
        """
        assert sd < Side.SIZE, "Invalid side"
        b = 0
        
        f = Square.to_88(sq)
        
        for direction in range(2):
            t = f + Attack.Pawn_Attack[sd][direction]
            if Square.is_valid_88(t):
                b = Bit.set_bit(b, Square.from_88(t))
                
        return b
    
    @staticmethod
    def piece_attacks_debug(pc: int, sq: int) -> int:
        """
        Debug function to compute piece attacks from square sq for piece pc.
        pc: Piece type
        sq: From square index
        """
        assert pc != Piece.PAWN, "Pawn does not use piece_attacks_debug"
        b = 0
        
        f = Square.to_88(sq)
        
        direction = 0
        while True:
            inc = Attack.Piece_Inc[pc][direction]
            if inc == 0:
                break
        
            if Piece.is_slider(pc):
                t = f + inc
                while Square.is_valid_88(t):
                    b = Bit.set_bit(b, Square.from_88(t))
                    t += inc
            else:
                t = f + inc
                if Square.is_valid_88(t):
                    b = Bit.set_bit(b, Square.from_88(t))
            direction += 1
            
        return b
    
    @staticmethod
    def delta_inc(f: int, t: int) -> int:
        """
        Determine the increment direction from square f to square t.
        Returns the increment value if a straight line exists, otherwise 0.
        """
        for inc in Attack.Queen_Inc[:-1]:  # Exclude the terminating 0
            sq = f + inc
            while Square.is_valid_88(sq):
                if sq == t:
                    return inc
                sq += inc
        return 0
    
    @staticmethod
    def between_debug(f: int, t: int) -> int:
        """
        Compute the bitboard representing squares between f and t.
        """
        f = Square.to_88(f)
        t = Square.to_88(t)
        
        b = 0
        
        inc = Attack.delta_inc(f, t)
        
        if inc != 0:
            sq = f + inc
            while sq != t:
                b = Bit.set_bit(b, Square.from_88(sq))
                sq += inc
                
        return b
    
    @staticmethod
    def behind_debug(f: int, t: int) -> int:
        """
        Compute the bitboard representing squares behind t from f.
        """
        f = Square.to_88(f)
        t = Square.to_88(t)
        
        b = 0
        
        inc = Attack.delta_inc(f, t)
        
        if inc != 0:
            sq = t + inc
            while Square.is_valid_88(sq):
                b = Bit.set_bit(b, Square.from_88(sq))
                sq += inc
                
        return b
    
    @staticmethod
    def blockers_debug(pc: int, f: int) -> int:
        """
        Compute the blockers for a piece pc from square f.
        """
        assert pc != Piece.PAWN, "Pawn does not use blockers_debug"
        b = 0
        
        attacks = Attack.piece_attacks_debug(pc, f)
        
        bb = attacks
        while bb != 0:
            sq = Bit.first(bb)
            if (attacks & Attack.behind_debug(f, sq)) != 0:
                b = Bit.set_bit(b, sq)
            bb = Bit.rest(bb)

        return b
    
    @staticmethod
    def init():
        """Initialize all attack tables."""
        # Initialize Pawn_Moves and Pawn_Attacks
        for sd in range(Side.SIZE):
            for sq in range(Square.SIZE):
                Attack.Pawn_Moves[sd][sq] = Attack.pawn_moves_debug(sd, sq)
                Attack.Pawn_Attacks[sd][sq] = Attack.pawn_attacks_debug(sd, sq)
        
        # Initialize Piece_Attacks and Blockers for non-pawn pieces
        for pc in range(Piece.KNIGHT, Piece.KING + 1):  # KNIGHT, BISHOP, ROOK, QUEEN, KING
            for sq in range(Square.SIZE):
                Attack.Piece_Attacks[pc][sq] = Attack.piece_attacks_debug(pc, sq)
                Attack.Blockers[pc][sq] = Attack.blockers_debug(pc, sq)
        
        # Initialize Between and Behind bitboards
        for f in range(Square.SIZE):
            for t in range(Square.SIZE):
                Attack.Between[f][t] = Attack.between_debug(f, t)
                Attack.Behind[f][t] = Attack.behind_debug(f, t)


class Gen:
    class List:
        SIZE = 256

        def __init__(self):
            self.p_size = 0
            self.p_pair = [0] * Gen.List.SIZE
            # self.clear()

        def move_to(self, pf: int, pt: int):
            """
            Move a move from position `pf` to position `pt` within the list.
            Shifts elements to make space.
            """
            assert pt <= pf and pf < self.p_size, "Invalid indices in move_to"
            p = self.p_pair[pf]
            for i in range(pf, pt, -1):
                self.p_pair[i] = self.p_pair[i - 1]
            self.p_pair[pt] = p

        def add_pair(self, p: int):
            """
            Add a move-score pair to the list.
            """
            assert self.p_size < Gen.List.SIZE, "List capacity exceeded"
            self.p_pair[self.p_size] = p
            self.p_size += 1

        def pair(self, pos: int) -> int:
            """
            Retrieve the move-score pair at position `pos`.
            """
            assert pos < self.p_size, "Index out of range in pair"
            return self.p_pair[pos]

        def assign(self, ml: 'Gen.List'):
            """
            Copy all move-score pairs from another List `ml` into this list.
            """
            self.clear()
            for pos in range(ml.size()):
                p = ml.pair(pos)
                self.add_pair(p)

        def clear(self):
            """
            Clear the list by resetting its size.
            """
            self.p_size = 0
            # p_pair remains as list

        def add(self, mv: int, sc: int = 0):
            """
            Add a move `mv` with an optional score `sc` to the list.
            Ensures the move is within valid range and not already present.
            """
            assert 0 <= mv < Move.SIZE, f"Move {mv} out of range"
            assert 0 <= sc < Move.SCORE_SIZE, f"Score {sc} out of range"
            assert not self.contain(mv), f"Move {mv} already in list"
            self.add_pair((sc << Move.BITS) | mv)

        def set_score(self, pos: int, sc: int):
            """
            Set the score for the move at position `pos`.
            """
            assert pos < self.p_size, "Position out of range in set_score"
            assert 0 <= sc < Move.SCORE_SIZE, "Score out of range in set_score"
            self.p_pair[pos] = (sc << Move.BITS) | self.move(pos)
            assert self.score(pos) == sc, "Score not set correctly"

        def move_to_front(self, pos: int):
            """
            Move the move at position `pos` to the front of the list.
            """
            self.move_to(pos, 0)

        def sort(self):
            """
            Sort the list using insertion sort in descending order based on move-score pairs.
            """
            for i in range(1, self.p_size):
                p = self.p_pair[i]
                j = i
                while j > 0 and self.p_pair[j - 1] < p:
                    self.p_pair[j] = self.p_pair[j - 1]
                    j -= 1
                self.p_pair[j] = p

            for i in range(self.p_size - 1):
                assert self.p_pair[i] >= self.p_pair[i + 1], "List not sorted correctly"

        def size(self) -> int:
            """
            Return the current size of the list.
            """
            return self.p_size

        def move(self, pos: int) -> int:
            """
            Retrieve the move at position `pos`.
            """
            return self.pair(pos) & Move.MASK

        def score(self, pos: int) -> int:
            """
            Retrieve the score of the move at position `pos`.
            """
            return self.pair(pos) >> Move.BITS

        def contain(self, mv: int) -> bool:
            """
            Check if the move `mv` is already present in the list.
            """
            for pos in range(self.size()):
                if self.move(pos) == mv:
                    return True
            return False

    @staticmethod
    def add_pawn_move(ml: 'Gen.List', f: int, t: int, bd: 'Board.BOARD'):
        """
        Add a pawn move from square `f` to square `t` to the move list `ml`.
        Handles promotions by adding multiple moves with different promotion pieces.
        """
        assert bd.square(f) == Piece.PAWN, "Source square does not have a pawn"

        pc = bd.square(f)
        cp = bd.square(t)

        if Square.is_promotion(t):
            ml.add(Move.make(f, t, pc, cp, Piece.QUEEN))
            ml.add(Move.make(f, t, pc, cp, Piece.KNIGHT))
            ml.add(Move.make(f, t, pc, cp, Piece.ROOK))
            ml.add(Move.make(f, t, pc, cp, Piece.BISHOP))
        else:
            ml.add(Move.make(f, t, pc, cp))

    @staticmethod
    def add_piece_move(ml: 'Gen.List', f: int, t: int, bd: 'Board.BOARD'):
        """
        Add a non-pawn piece move from square `f` to square `t` to the move list `ml`.
        """
        assert bd.square(f) != Piece.PAWN, "Source square has a pawn"
        ml.add(Move.make(f, t, bd.square(f), bd.square(t)))

    @staticmethod
    def add_move(ml: 'Gen.List', f: int, t: int, bd: 'Board.BOARD'):
        """
        Add a move from square `f` to square `t` to the move list `ml`.
        Determines if the move is a pawn move or a piece move.
        """
        if bd.square(f) == Piece.PAWN:
            Gen.add_pawn_move(ml, f, t, bd)
        else:
            Gen.add_piece_move(ml, f, t, bd)

    @staticmethod
    def add_piece_moves_from(ml: 'Gen.List', f: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all possible piece moves from square `f` to target squares `ts` to the move list `ml`.
        """
        pc = bd.square(f)
        attacks = Attack.piece_attacks_from(pc, f, bd) & ts
        while attacks != 0:
            t = Bit.first(attacks)
            Gen.add_piece_move(ml, f, t, bd)
            attacks = Bit.rest(attacks)

    @staticmethod
    def add_captures_to(ml: 'Gen.List', sd: int, t: int, bd: 'Board.BOARD'):
        """
        Add all capture moves to square `t` for side `sd` to the move list `ml`.
        """
        for pc in range(Piece.PAWN, Piece.KING + 1):
            b = bd.piece(pc, sd) & Attack.attacks_to(pc, sd, t, bd)
            while b != 0:
                f = Bit.first(b)
                Gen.add_move(ml, f, t, bd)
                b = Bit.rest(b)

    @staticmethod
    def add_captures_to_no_king(ml: 'Gen.List', sd: int, t: int, bd: 'Board.BOARD'):
        """
        Add all capture moves to square `t` for side `sd`, excluding king captures, to the move list `ml`.
        Used for generating evasions.
        """
        for pc in range(Piece.PAWN, Piece.QUEEN + 1):  # skip king
            b = bd.piece(pc, sd) & Attack.attacks_to(pc, sd, t, bd)
            while b != 0:
                f = Bit.first(b)
                Gen.add_move(ml, f, t, bd)
                b = Bit.rest(b)

    @staticmethod
    def add_pawn_captures(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all pawn capture moves for side `sd` to target squares `ts` to the move list `ml`.
        Handles both left and right captures.
        """
        pawns = bd.piece(Piece.PAWN, sd)
        ts &= bd.side(Side.opposit(sd))  # Intersection with opponent's pieces

        if sd == Side.WHITE:
            # Capture to the left (from white's perspective)
            captures = (ts << 7) & pawns & ((1 << 64) - 1)
            while captures != 0:
                f = Bit.first(captures)
                t = f - 7
                Gen.add_pawn_move(ml, f, t, bd)
                captures = Bit.rest(captures)

            # Capture to the right (from white's perspective)
            captures = (ts >> 9) & pawns
            while captures != 0:
                f = Bit.first(captures)
                t = f + 9
                Gen.add_pawn_move(ml, f, t, bd)
                captures = Bit.rest(captures)
        else:
            # Capture to the left (from black's perspective)
            captures = (ts << 9) & pawns & ((1 << 64) - 1)
            while captures != 0:
                f = Bit.first(captures)
                t = f - 9
                Gen.add_pawn_move(ml, f, t, bd)
                captures = Bit.rest(captures)

            # Capture to the right (from black's perspective)
            captures = (ts >> 7) & pawns
            while captures != 0:
                f = Bit.first(captures)
                t = f + 7
                Gen.add_pawn_move(ml, f, t, bd)
                captures = Bit.rest(captures)

    @staticmethod
    def add_promotions(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all pawn promotion moves for side `sd` to target squares `ts` to the move list `ml`.
        """
        pawns = bd.piece(Piece.PAWN, sd)

        if sd == Side.WHITE:
            # White promotion: pawns on rank 7 move to rank 8
            promotions = pawns & (ts >> 1) & Bit.rank(Square.RANK_7)
            while promotions != 0:
                f = Bit.first(promotions)
                t = f + 1
                assert bd.square(t) == Piece.NONE, "Promotion target not empty"
                assert Square.is_promotion(t), "Target square is not a promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                promotions = Bit.rest(promotions)
        else:
            # Black promotion: pawns on rank 2 move to rank 1
            promotions = pawns & (ts << 1) & Bit.rank(Square.RANK_2)
            while promotions != 0:
                f = Bit.first(promotions)
                t = f - 1
                assert bd.square(t) == Piece.NONE, "Promotion target not empty"
                assert Square.is_promotion(t), "Target square is not a promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                promotions = Bit.rest(promotions)

    # @staticmethod
    # def add_promotions_overload(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
    #     """
    #     Overloaded function to add promotions without specifying target squares.
    #     """
    #     Gen.add_promotions(ml, sd, bd.empty(), bd)

    @staticmethod
    def add_pawn_quiets(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all quiet (non-capturing) pawn moves for side `sd` to target squares `ts` to the move list `ml`.
        Handles single and double pushes.
        """
        pawns = bd.piece(Piece.PAWN, sd)
        empty = bd.empty()

        if sd == Side.WHITE:
            # Single push
            pushes = (pawns & (ts >> 1) & ~Bit.rank(Square.RANK_7))
            assert pushes == pushes & 0xFFFFFFFFFFFFFFFF
            
            while pushes != 0:
                f = Bit.first(pushes)
                t = f + 1
                assert bd.square(t) == Piece.NONE, "Single push target not empty"
                assert not Square.is_promotion(t), "Single push target is promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                pushes = Bit.rest(pushes)

            # Double push
            pushes = (pawns & (ts >> 2) & (empty >> 1) & Bit.rank(Square.RANK_2))
            while pushes != 0:
                f = Bit.first(pushes)
                t = f + 2
                assert bd.square(t) == Piece.NONE, "Double push target not empty"
                assert not Square.is_promotion(t), "Double push target is promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                pushes = Bit.rest(pushes)
        else:
            # Black side
            # Single push
            pushes = (pawns & (ts << 1) & ~Bit.rank(Square.RANK_2))
            assert pushes == pushes & 0xFFFFFFFFFFFFFFFF
            
            while pushes != 0:
                f = Bit.first(pushes)
                t = f - 1
                assert bd.square(t) == Piece.NONE, "Single push target not empty"
                assert not Square.is_promotion(t), "Single push target is promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                pushes = Bit.rest(pushes)

            # Double push
            pushes = (pawns & (ts << 2) & (empty << 1) & Bit.rank(Square.RANK_7))
            while pushes != 0:
                f = Bit.first(pushes)
                t = f - 2
                assert bd.square(t) == Piece.NONE, "Double push target not empty"
                assert not Square.is_promotion(t), "Double push target is promotion square"
                Gen.add_pawn_move(ml, f, t, bd)
                pushes = Bit.rest(pushes)

    @staticmethod
    def add_pawn_pushes(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all pawn quiet moves (single and double pushes) for side `sd` to the move list `ml`.
        """
        ts = 0

        if sd == Side.WHITE:
            ts |= Bit.rank(Square.RANK_7)
            ts |= Bit.rank(Square.RANK_6) & ~Attack.pawn_attacks_from(Side.BLACK, bd) & (~bd.piece(Piece.PAWN) >> 1)
            assert ts == ts & 0xFFFFFFFFFFFFFFFF
            
        else:
            ts |= Bit.rank(Square.RANK_2)
            ts |= Bit.rank(Square.RANK_3) & ~Attack.pawn_attacks_from(Side.WHITE, bd) & (~bd.piece(Piece.PAWN) << 1)
            assert ts == ts & 0xFFFFFFFFFFFFFFFF

        Gen.add_pawn_quiets(ml, sd, ts & bd.empty(), bd)

    @staticmethod
    def add_en_passant(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all en passant capture moves for side `sd` to the move list `ml`.
        """
        t = bd.ep_sq()
        if t != Square.NONE:
            # Find pawns of side sd that can capture en passant to t
            # Attack.Pawn_Attacks[Side.opposit(sd)][t] gives a bitboard of attacking pawns
            fs = bd.piece(Piece.PAWN, sd) & Attack.Pawn_Attacks[Side.opposit(sd)][t]
            while fs != 0:
                f = Bit.first(fs)
                mv = Move.make(f, t, Piece.PAWN, Piece.PAWN)
                ml.add(mv)
                fs = Bit.rest(fs)

    @staticmethod
    def can_castle(sd: int, wg: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if side `sd` can perform castling of wing `wg` on board `bd`.
        """
        index = Castling.index(sd, wg)

        if Castling.flag(bd.flags(), index):
            kf = Castling.info[index].kf
            # kt = Castling.info[index].kt
            rf = Castling.info[index].rf
            rt = Castling.info[index].rt

            assert bd.square_is(kf, Piece.KING, sd), "King square incorrect for castling"
            assert bd.square_is(rf, Piece.ROOK, sd), "Rook square incorrect for castling"

            if not Attack.line_is_empty(kf, rf, bd):
                return False
            if Attack.is_attacked(rt, Side.opposit(sd), bd):
                return False
            return True
        return False

    @staticmethod
    def add_castling(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all possible castling moves for side `sd` to the move list `ml`.
        """
        for wg in range(Wing.SIZE):
            if Gen.can_castle(sd, wg, bd):
                index = Castling.index(sd, wg)
                # Add the castling move: King from kf to kt
                Gen.add_piece_move(ml, Castling.info[index].kf, Castling.info[index].kt, bd)

    @staticmethod
    def add_piece_moves(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all piece moves for side `sd` to target squares `ts` to the move list `ml`.
        """
        assert ts != 0, "Target squares set is empty"

        for pc in range(Piece.KNIGHT, Piece.KING + 1):
            pieces = bd.piece(pc, sd)
            while pieces != 0:
                f = Bit.first(pieces)
                Gen.add_piece_moves_from(ml, f, ts, bd)
                pieces = Bit.rest(pieces)

    @staticmethod
    def add_piece_moves_no_king(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all piece moves (excluding king moves) for side `sd` to target squares `ts` to the move list `ml`.
        Used for generating evasions.
        """
        assert ts != 0, "Target squares set is empty"

        for pc in range(Piece.KNIGHT, Piece.QUEEN + 1):  # skip king
            pieces = bd.piece(pc, sd)
            while pieces != 0:
                f = Bit.first(pieces)
                Gen.add_piece_moves_from(ml, f, ts, bd)
                pieces = Bit.rest(pieces)

    @staticmethod
    def add_piece_moves_rare(ml: 'Gen.List', sd: int, ts: int, bd: 'Board.BOARD'):
        """
        Add all rare piece moves (typically captures) for side `sd` to target squares `ts` to the move list `ml`.
        """
        assert ts != 0, "Target squares set is empty"

        for pc in range(Piece.KNIGHT, Piece.KING + 1):
            pieces = bd.piece(pc, sd)
            while pieces != 0:
                f = Bit.first(pieces)
                attacks = Attack.pseudo_attacks_from(pc, sd, f) & ts
                while attacks != 0:
                    t = Bit.first(attacks)
                    if Attack.line_is_empty(f, t, bd):
                        Gen.add_piece_move(ml, f, t, bd)
                    attacks = Bit.rest(attacks)
                pieces = Bit.rest(pieces)

    @staticmethod
    def add_captures(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all capture moves for side `sd` to the move list `ml`.
        """
        ts = bd.side(Side.opposit(sd))
        Gen.add_pawn_captures(ml, sd, ts, bd)
        Gen.add_piece_moves_rare(ml, sd, ts, bd)
        Gen.add_en_passant(ml, sd, bd)

    @staticmethod
    def add_captures_mvv_lva(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add captures sorted by Most Valuable Victim / Least Valuable Aggressor (MVV/LVA).
        Note: This function is marked as unused in the original C++ code.
        """
        for pc in range(Piece.QUEEN, Piece.PAWN - 1, -1):
            pieces = bd.piece(pc, Side.opposit(sd))
            while pieces != 0:
                f = Bit.first(pieces)
                Gen.add_captures_to(ml, sd, f, bd)
                pieces = Bit.rest(pieces)
        Gen.add_en_passant(ml, sd, bd)

    @staticmethod
    def is_move(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a move `mv` is legal on board `bd`.
        Used for transposition table move legality.
        """
        sd = bd.turn()

        f = Move.from_sq(mv)
        t = Move.to_sq(mv)

        pc = Move.piece(mv)
        cp = Move.cap(mv)

        if not (bd.square(f) == pc and bd.square_side(f) == sd):
            return False

        if bd.square(t) != Piece.NONE and bd.square_side(t) == sd:
            return False

        if pc == Piece.PAWN and t == bd.ep_sq():
            if cp != Piece.PAWN:
                return False
        elif bd.square(t) != cp:
            return False

        if cp == Piece.KING:
            return False

        if pc == Piece.PAWN:
            # TODO: Implement pawn-specific legality checks
            return True
        else:
            # TODO: Implement castling and piece-specific legality checks
            # Example: return Attack.piece_attack(pc, f, t, bd)
            return True

        assert False, "Unreachable in is_move"

    @staticmethod
    def is_quiet_move(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a move `mv` is a quiet move (non-capturing) on board `bd`.
        Used for killer move legality.
        """
        sd = bd.turn()

        f = Move.from_sq(mv)
        t = Move.to_sq(mv)

        pc = Move.piece(mv)
        cp = Move.cap(mv)

        assert cp == Piece.NONE, "Captured piece must be NONE for quiet move"
        assert Move.prom(mv) == Piece.NONE, "Promotion piece must be NONE for quiet move"

        if not (bd.square(f) == pc and bd.square_side(f) == sd):
            return False

        if bd.square(t) != Piece.NONE:
            return False

        if pc == Piece.PAWN:
            inc = Square.pawn_inc(sd)

            if t - f == inc and not Square.is_promotion(t):
                return True
            elif t - f == inc * 2 and Square.rank(f, sd) == Square.RANK_2:
                if bd.square(f + inc) == Piece.NONE:
                    return True
                else:
                    return False
            else:
                return False
        else:
            # TODO: Implement castling legality checks
            return Attack.piece_attack(pc, f, t, bd)

        assert False, "Unreachable in is_quiet_move"

    @staticmethod
    def add_quiets(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all quiet (non-capturing) moves for side `sd` to the move list `ml`.
        """
        Gen.add_castling(ml, sd, bd)
        Gen.add_piece_moves(ml, sd, bd.empty(), bd)
        Gen.add_pawn_quiets(ml, sd, bd.empty(), bd)

    @staticmethod
    def add_evasions(ml: 'Gen.List', sd: int, bd: 'Board.BOARD', attacks: 'Attack.Attacks'):
        """
        Add all evasion moves for side `sd` when in check to the move list `ml`.
        """
        assert attacks.size > 0, "Attacks size must be greater than 0"

        king = bd.king(sd)

        Gen.add_piece_moves_from(ml, king, ~bd.side(sd) & ~attacks.avoid & 0xFFFFFFFFFFFFFFFF, bd)

        if attacks.size == 1:
            t = attacks.square[0]
            Gen.add_captures_to_no_king(ml, sd, t, bd)
            Gen.add_en_passant(ml, sd, bd)

            ts = Attack.Between[king][t]
            assert Attack.line_is_empty(king, t, bd)

            if ts != 0:
                Gen.add_pawn_quiets(ml, sd, ts, bd)
                Gen.add_promotions(ml, sd, ts, bd)
                Gen.add_piece_moves_no_king(ml, sd, ts, bd)

    @staticmethod
    def add_evasions_overload(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Overloaded function to add evasions without explicitly passing attacks.
        """
        attacks = Attack.Attacks()
        Attack.init_attacks(attacks, sd, bd)
        Gen.add_evasions(ml, sd, bd, attacks)

    @staticmethod
    def add_checks(ml: 'Gen.List', sd: int, bd: 'Board.BOARD'):
        """
        Add all check-inducing moves for side `sd` to the move list `ml`.
        """
        atk = sd
        defn = Side.opposit(sd)

        king = bd.king(defn)
        pinned = Attack.pinned_by(king, atk, bd)
        empty = bd.empty()
        empty &= ~Attack.pawn_attacks_from(Side.opposit(sd), bd)  # pawn-safe
        assert empty == empty & 0xFFFFFFFFFFFFFFFF

        # Discovered checks
        fs = bd.pieces(atk) & pinned
        while fs != 0:
            f = Bit.first(fs)
            ts = empty & ~Attack.ray(king, f)
            assert ts == ts & 0xFFFFFFFFFFFFFFFF
            
            Gen.add_piece_moves_from(ml, f, ts, bd)
            fs = Bit.rest(fs)

        # Direct checks, pawns
        ts = Attack.pseudo_attacks_to(Piece.PAWN, sd, king) & empty
        Gen.add_pawn_quiets(ml, sd, ts, bd)

        # Direct checks, knights
        pc = Piece.KNIGHT
        attacks = Attack.pseudo_attacks_to(pc, sd, king) & empty
        pieces = bd.piece(pc, sd) & ~pinned
        assert pieces == pieces & 0xFFFFFFFFFFFFFFFF
        
        while pieces != 0:
            f = Bit.first(pieces)
            moves = Attack.pseudo_attacks_from(pc, sd, f)
            pseudo_attacks = moves & attacks
            while pseudo_attacks != 0:
                t = Bit.first(pseudo_attacks)
                Gen.add_piece_move(ml, f, t, bd)
                pseudo_attacks = Bit.rest(pseudo_attacks)
            pieces = Bit.rest(pieces)

        # Direct checks, sliders
        for pc in range(Piece.BISHOP, Piece.QUEEN + 1):
            attacks = Attack.pseudo_attacks_to(pc, sd, king) & empty
            pieces = bd.piece(pc, sd) & ~pinned
            assert pieces == pieces & 0xFFFFFFFFFFFFFFFF
            
            while pieces != 0:
                f = Bit.first(pieces)
                moves = Attack.pseudo_attacks_from(pc, sd, f) & attacks
                while moves != 0:
                    t = Bit.first(moves)
                    if Attack.line_is_empty(f, t, bd) and Attack.line_is_empty(t, king, bd):
                        Gen.add_piece_move(ml, f, t, bd)
                    moves = Bit.rest(moves)
                pieces = Bit.rest(pieces)

    @staticmethod
    def is_legal_debug(mv: int, bd: 'Board.BOARD') -> bool:
        """
        Debug function to check if move `mv` is legal on board `bd`.
        Temporarily makes the move, checks legality, and undoes it.
        """
        bd.move(mv)
        b = Attack.is_legal(bd)
        bd.undo()
        return b

    @staticmethod
    def gen_moves_debug(ml: 'Gen.List', bd: 'Board.BOARD'):
        """
        Generate all pseudo-legal moves and add them to the move list `ml`.
        If the king is in check, generate evasion moves instead.
        """
        ml.clear()
        sd = bd.turn()

        if Attack.is_in_check(bd):
            Gen.add_evasions_overload(ml, sd, bd)
        else:
            Gen.add_captures(ml, sd, bd)
            Gen.add_promotions(ml, sd, bd.empty(), bd)
            Gen.add_quiets(ml, sd, bd)

    @staticmethod
    def filter_legals(dst: 'Gen.List', src: 'Gen.List', bd: 'Board.BOARD'):
        """
        Filter pseudo-legal moves in `src` by checking their legality and add legal moves to `dst`.
        """
        dst.clear()

        for pos in range(src.size()):
            mv = src.move(pos)
            if Gen.is_legal_debug(mv, bd):
                dst.add(mv)

    @staticmethod
    def gen_legals(ml: 'Gen.List', bd: 'Board.BOARD'):
        """
        Generate all legal moves for the current position on board `bd` and add them to `ml`.
        """
        pseudos = Gen.List()
        Gen.gen_moves_debug(pseudos, bd)
        Gen.filter_legals(ml, pseudos, bd)

    @staticmethod
    def gen_legal_evasions(ml: 'Gen.List', bd: 'Board.BOARD'):
        """
        Generate all legal evasion moves for side `sd` when in check and add them to `ml`.
        """
        sd = bd.turn()

        attacks = Attack.Attacks()
        Attack.init_attacks(attacks, sd, bd)

        if attacks.size == 0:
            ml.clear()
            return

        pseudos = Gen.List()
        Gen.add_evasions(ml, sd, bd, attacks)

        Gen.filter_legals(ml, pseudos, bd)


class Score:
    # Constants
    NONE = -10000
    MIN = -9999
    EVAL_MIN = -8999
    EVAL_MAX = +8999
    MAX = +9999
    MATE = +10000

    # Flags
    FLAGS_NONE = 0
    FLAGS_LOWER = 1 << 0
    FLAGS_UPPER = 1 << 1
    FLAGS_EXACT = FLAGS_LOWER | FLAGS_UPPER

    @staticmethod
    def is_mate(sc: int) -> bool:
        """
        Check if the score represents a mate situation.
        """
        return sc < Score.EVAL_MIN or sc > Score.EVAL_MAX

    @staticmethod
    def signed_mate(sc: int) -> int:
        """
        Convert a mate score to a signed mate distance.
        """
        if sc < Score.EVAL_MIN:  # -MATE
            return -(Score.MATE + sc) // 2
        elif sc > Score.EVAL_MAX:  # +MATE
            return (Score.MATE - sc + 1) // 2
        else:
            assert False, "signed_mate called with non-mate score"
            return 0

    @staticmethod
    def side_score(sc: int, sd: int) -> int:
        """
        Adjust the score based on the side.
        """
        return +sc if sd == Side.WHITE else -sc

    @staticmethod
    def from_trans(sc: int, ply: int) -> int:
        """
        Adjust the score when storing it in the transposition table.
        """
        if sc < Score.EVAL_MIN:
            return sc + ply
        elif sc > Score.EVAL_MAX:
            return sc - ply
        else:
            return sc

    @staticmethod
    def to_trans(sc: int, ply: int) -> int:
        """
        Adjust the score when retrieving it from the transposition table.
        """
        if sc < Score.EVAL_MIN:
            return sc - ply
        elif sc > Score.EVAL_MAX:
            return sc + ply
        else:
            return sc

    @staticmethod
    def flags(sc: int, alpha: int, beta: int) -> int:
        """
        Determine the flags based on the score and alpha-beta bounds.
        """
        flags = Score.FLAGS_NONE
        if sc > alpha:
            flags |= Score.FLAGS_LOWER
        if sc < beta:
            flags |= Score.FLAGS_UPPER
        return flags


class Trans:
    class Entry:
        """
        Packed representation of a transposition table entry.
        Bit layout:
        [103-72] lock (unsigned 32 bits)
        [71-40]  move (unsigned 32 bits)
        [39-24]  score (signed 16 bits)
        [23-16]  date (unsigned 8 bits)
        [15-8]   depth (signed 8 bits)
        [7-0]    flags (unsigned 8 bits)
        """

        # Define bit shifts for each field
        LOCK_SHIFT = 72
        MOVE_SHIFT = 40
        SCORE_SHIFT = 24
        DATE_SHIFT = 16
        DEPTH_SHIFT = 8
        FLAGS_SHIFT = 0

        # Define bit masks for each field
        LOCK_MASK = 0xFFFFFFFF << LOCK_SHIFT
        MOVE_MASK = 0xFFFFFFFF << MOVE_SHIFT
        SCORE_MASK = 0xFFFF << SCORE_SHIFT
        DATE_MASK = 0xFF << DATE_SHIFT
        DEPTH_MASK = 0xFF << DEPTH_SHIFT
        FLAGS_MASK = 0xFF  << FLAGS_SHIFT  # same as 0xFF

        def __init__(self, value=0):
            self.value = value

        @classmethod
        def create(cls, lock=0, move=0, score=0, date=0, depth=-1, flags=0):
            """
            Create a new Entry with the specified values.
            Expects:
            - lock: unsigned 32-bit int
            - move: unsigned 32-bit int (e.g. Move.NONE)
            - score: signed 16-bit int
            - date: unsigned 8-bit int
            - depth: signed 8-bit int (default -1)
            - flags: unsigned 8-bit int (e.g. Score.FLAGS_NONE)
            """
            # Ensure values are within range
            lock  &= 0xFFFFFFFF   # 32 bits
            move  &= 0xFFFFFFFF   # 32 bits
            score &= 0xFFFF       # 16 bits (stored in two's complement)
            date  &= 0xFF         # 8 bits
            depth &= 0xFF         # 8 bits (two's complement for signed)
            flags &= 0xFF         # 8 bits

            # Pack values into one integer
            value = (lock  << cls.LOCK_SHIFT) | \
                    (move  << cls.MOVE_SHIFT) | \
                    (score << cls.SCORE_SHIFT) | \
                    (date  << cls.DATE_SHIFT) | \
                    (depth << cls.DEPTH_SHIFT) | \
                    (flags << cls.FLAGS_SHIFT)
            return cls(value)

        @property
        def lock(self):
            return (self.value & self.LOCK_MASK) >> self.LOCK_SHIFT

        @lock.setter
        def lock(self, lock):
            self.value = (self.value & ~self.LOCK_MASK) | ((lock & 0xFFFFFFFF) << self.LOCK_SHIFT)

        @property
        def move(self):
            return (self.value & self.MOVE_MASK) >> self.MOVE_SHIFT

        @move.setter
        def move(self, move):
            self.value = (self.value & ~self.MOVE_MASK) | ((move & 0xFFFFFFFF) << self.MOVE_SHIFT)

        @property
        def score(self):
            # Extract score and perform sign extension for a 16-bit signed integer.
            score = (self.value & self.SCORE_MASK) >> self.SCORE_SHIFT
            if score & 0x8000:  # If sign bit is set
                score = score - 0x10000
            return score

        @score.setter
        def score(self, score):
            score &= 0xFFFF
            self.value = (self.value & ~self.SCORE_MASK) | ((score & 0xFFFF) << self.SCORE_SHIFT)

        @property
        def date(self):
            return (self.value & self.DATE_MASK) >> self.DATE_SHIFT

        @date.setter
        def date(self, date):
            self.value = (self.value & ~self.DATE_MASK) | ((date & 0xFF) << self.DATE_SHIFT)

        @property
        def depth(self):
            # Extract depth and perform sign extension for an 8-bit signed integer.
            depth = (self.value & self.DEPTH_MASK) >> self.DEPTH_SHIFT
            if depth & 0x80:  # Check if the sign bit is set
                depth = depth - 0x100
            return depth

        @depth.setter
        def depth(self, depth):
            depth &= 0xFF
            self.value = (self.value & ~self.DEPTH_MASK) | ((depth & 0xFF) << self.DEPTH_SHIFT)

        @property
        def flags(self):
            return (self.value & self.FLAGS_MASK) >> self.FLAGS_SHIFT

        @flags.setter
        def flags(self, flags):
            flags &= 0xFF
            self.value = (self.value & ~self.FLAGS_MASK) | ((flags & 0xFF) << self.FLAGS_SHIFT)

    class Table:
        def __init__(self):
            self.p_table: Optional[List[int]] = None
            self.p_bits: int = 0
            self.p_size: int = 1
            self.p_mask: int = 0

            self.p_date: int = 0
            self.p_used: int = 0

        def size_to_bits(self, size: int) -> int:
            """
            Calculate the number of bits needed based on the desired table size.
            """
            bits = 0
            entries = (size << 20) // 16 # assuming 16 bytes per entry
            while entries > 1:
                bits += 1
                entries //= 2
            return bits

        def set_size(self, size: int):
            """
            Set the size of the transposition table.
            """
            bits = self.size_to_bits(size)
            if bits == self.p_bits:
                return
            self.p_bits = bits
            self.p_size = 1 << bits
            self.p_mask = self.p_size - 1
            if self.p_table is not None:
                self.free()
                self.alloc()

        def alloc(self):
            """
            Allocate the transposition table.
            """
            assert self.p_table is None, "Transposition table already allocated."
            entry = Trans.Entry.create(lock=0, move=Move.NONE, score=0, date=0, depth=-1, flags=Score.FLAGS_NONE)
            self.p_table = [entry.value] * self.p_size
            self.p_date = 1
            self.p_used = 0

        def free(self):
            """
            Free the transposition table.
            """
            assert self.p_table is not None, "Transposition table is not allocated."
            self.p_table = None

        def clear(self):
            """
            Clear all entries in the transposition table.
            """
            assert self.p_table is not None, "Transposition table is not allocated."
            entry = Trans.Entry.create(lock=0, move=Move.NONE, score=0, date=0, depth=-1, flags=Score.FLAGS_NONE)
            self.p_table = [entry.value] * self.p_size
            self.p_date = 1
            self.p_used = 0

        def inc_date(self):
            """
            Increment the current date to invalidate old entries.
            """
            self.p_date = (self.p_date + 1) % 256
            self.p_used = 0

        def store(self, key: int, depth: int, ply: int, move: int, score: int, flags: int):
            """
            Store an entry in the transposition table.
            """
            assert 0 <= depth < 100, "Depth out of range."
            assert move != Move.NULL_, "Cannot store NULL_ move."
            assert Score.MIN <= score <= Score.MAX, "Score out of range."

            score = Score.to_trans(score, ply)

            index = Hash.index(key) & self.p_mask
            lock = Hash.lock(key)

            best_index = None
            best_score = -1

            for i in range(4):
                idx = (index + i) & self.p_mask
                assert idx < self.p_size, "Index out of bounds."
                entry = Trans.Entry(self.p_table[idx])

                if entry.lock == lock:
                    if entry.date != self.p_date:
                        entry.date = self.p_date
                        self.p_used += 1
                        # Update the entry in the table
                        self.p_table[idx] = entry.value

                    if depth >= entry.depth:
                        if move != Move.NONE:
                            entry.move = move
                        entry.depth = depth
                        entry.score = score
                        entry.flags = flags
                        # Update the entry in the table
                        self.p_table[idx] = entry.value

                    elif entry.move == Move.NONE:
                        entry.move = move
                        # Update the entry in the table
                        self.p_table[idx] = entry.value
                    return

                sc = 99 - entry.depth  # entry.depth can be -1
                if entry.date != self.p_date:
                    sc += 101
                assert 0 <= sc < 202, "Score calculation out of range."

                if sc > best_score:
                    best_index = idx
                    best_score = sc

            assert best_index is not None, "No suitable entry found for replacement."

            # Create a new entry and store it
            best_entry = Trans.Entry.create(
                lock=lock,
                move=move,
                score=score,
                date=self.p_date,
                depth=depth,
                flags=flags
            )

            old_entry = Trans.Entry(self.p_table[best_index])
            if old_entry.date != self.p_date:
                self.p_used += 1

            self.p_table[best_index] = best_entry.value

        def retrieve(self, key: int, depth: int, ply: int) -> tuple:
            """
            Retrieve an entry from the transposition table.
            Returns a tuple (found, move, score, flags).
            """
            assert 0 <= depth < 100, "Depth out of range."

            index = Hash.index(key) & self.p_mask
            lock = Hash.lock(key)

            for i in range(4):
                idx = (index + i) & self.p_mask
                assert idx < self.p_size, "Index out of bounds."
                entry = Trans.Entry(self.p_table[idx])

                if entry.lock == lock:
                    if entry.date != self.p_date:
                        entry.date = self.p_date
                        self.p_used += 1
                        # Update the entry in the table
                        self.p_table[idx] = entry.value

                    move = entry.move
                    score = Score.from_trans(entry.score, ply)
                    flags = entry.flags

                    if entry.depth >= depth:
                        return True, move, score, flags
                    elif Score.is_mate(score):
                        if score < 0:
                            flags &= ~Score.FLAGS_LOWER
                        else:
                            flags &= ~Score.FLAGS_UPPER
                        return True, move, score, flags

                    return False, move, score, flags

            return False, Move.NONE, 0, Score.FLAGS_NONE

        def used(self) -> int:
            """
            Get the usage percentage of the transposition table.
            """
            return round(self.p_used * 1000 / self.p_size)


class Engine:
    @dataclass
    class ENGINE:
        hash_size: int = 64
        ponder: bool = False
        threads: int = 1
        log: bool = False
     
    engine = ENGINE()
    
    @classmethod
    def init(cls):
        """
        Initialize the engine configuration to default values.
        """
        cls.engine.hash_size = 64
        cls.engine.ponder = False
        cls.engine.threads = 1
        cls.engine.log = False


class Pawn:
    @dataclass
    class Info:
        open_file: List[List[int]] = field(default_factory=lambda: [[0 for _ in range(Side.SIZE)] for _ in range(Square.FILE_SIZE)])
        shelter: List[List[int]] = field(default_factory=lambda: [[0 for _ in range(Side.SIZE)] for _ in range(Square.FILE_SIZE)])
        passed: int = 0
        target: List[int] = field(default_factory=lambda: [0 for _ in range(Side.SIZE)])
        safe: int = 0
        lock: int = 0
        mg: int = 0
        eg: int = 0
        left_file: int = Square.FILE_A
        right_file: int = Square.FILE_H
    
    class Table:
        BITS = 12
        SIZE = 1 << BITS
        MASK = SIZE - 1

        def __init__(self):
            self.p_table: List['Pawn.Info'] = [Pawn.Info() for _ in range(Pawn.Table.SIZE)]

        def clear(self):
            """
            Clear all pawn info entries.
            """
            for info in self.p_table:
                Pawn.clear_info(info)

        def clear_fast(self):
            """
            Quickly clear the table by setting the lock.
            """
            for info in self.p_table:
                info.lock = 1  # Board without pawns has key 0

        def info(self, bd: 'Board.BOARD') -> 'Pawn.Info':
            """
            Retrieve pawn information for the current board state.
            """
            key = bd.pawn_key()
            index = Hash.index(key) & Pawn.Table.MASK
            lock = Hash.lock(key)

            entry = self.p_table[index]

            if entry.lock != lock:
                entry.lock = lock
                Pawn.comp_info(entry, bd)

            return entry
        
    # Initialize passed_me and passed_opp as class variables
    passed_me: List[List[int]] = [[0 for _ in range(Side.SIZE)] for _ in range(Square.SIZE)]
    passed_opp: List[List[int]] = [[0 for _ in range(Side.SIZE)] for _ in range(Square.SIZE)]
    
    # Precomputed duo bitboards
    duo = [0 for _ in range(Square.SIZE)]

    @staticmethod
    def is_passed(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if a pawn is passed.
        """
        return (bd.piece(Piece.PAWN, Side.opposit(sd)) & Pawn.passed_opp[sq][sd]) == 0 and \
               (bd.piece(Piece.PAWN, sd) & Pawn.passed_me[sq][sd]) == 0

    @staticmethod
    def square_distance(ks: int, ps: int, sd: int) -> int:
        """
        Calculate the distance of a pawn from its promotion square relative to the king's position.
        """
        prom = Square.promotion(ps, sd)
        return Square.distance(ks, prom) - Square.distance(ps, prom)

    @staticmethod
    def clear_info(info: 'Pawn.Info'):
        """
        Reset the pawn information to default values.
        """
        info.passed = 0
        info.safe = 0
        info.lock = 1  # Board without pawns has key 0
        info.mg = 0
        info.eg = 0
        info.left_file = Square.FILE_A
        info.right_file = Square.FILE_H

        for sd in range(Side.SIZE):
            info.target[sd] = 0

        for fl in range(Square.FILE_SIZE):
            for sd in range(Side.SIZE):
                info.open_file[fl][sd] = 0
                info.shelter[fl][sd] = 0
    
    @staticmethod
    def is_empty(sq: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a square is empty of pawns.
        """
        return bd.square(sq) != Piece.PAWN

    @staticmethod
    def is_attacked(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a square is attacked by pawns of a given side.
        """
        return (bd.piece(Piece.PAWN, sd) & Attack.pawn_attacks_to(sd, sq)) != 0

    @staticmethod
    def is_controlled(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if a square is controlled more by attackers than defenders.
        """
        attackers = bd.piece(Piece.PAWN, sd) & Attack.pawn_attacks_to(sd, sq)
        defenders = bd.piece(Piece.PAWN, Side.opposit(sd)) & Attack.pawn_attacks_to(Side.opposit(sd), sq)
        return Bit.count(attackers) > Bit.count(defenders)

    @staticmethod
    def is_safe(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Determine if a square is safe for a pawn.
        """
        return Pawn.is_empty(sq, bd) and not Pawn.is_controlled(sq, Side.opposit(sd), bd)

    @staticmethod
    def potential_attacks(sq: int, sd: int, bd: 'Board.BOARD') -> int:
        """
        Compute potential attacks from a square in front of a pawn.
        """
        inc = Square.pawn_inc(sd)
        attacks = Attack.pawn_attacks_from_single(sd, sq)

        while not Square.is_promotion(sq + inc) and Pawn.is_safe(sq + inc, sd, bd):
            sq += inc
            attacks |= Attack.pawn_attacks_from_single(sd, sq)

        return attacks

    @staticmethod
    def is_duo(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a pawn is part of a duo (double pawns).
        """
        return (bd.piece(Piece.PAWN, sd) & Pawn.duo[sq]) != 0

    @staticmethod
    def is_isolated(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a pawn is isolated.
        """
        fl = Square.file(sq)
        files = Bit.files(fl) & ~Bit.file(fl)
        assert files == files & 0xFFFFFFFFFFFFFFFF
        
        return (bd.piece(Piece.PAWN, sd) & files) == 0

    @staticmethod
    def is_weak(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a pawn is weak.
        """
        fl = Square.file(sq)
        rk = Square.rank(sq, sd)

        pawns = bd.piece(Piece.PAWN, sd)
        inc = Square.pawn_inc(sd)

        # Already fine?
        if (pawns & Pawn.duo[sq]) != 0:
            return False
        if Pawn.is_attacked(sq, sd, bd):
            return False

        # Can advance next to another pawn in one move?
        s1 = sq + inc
        s2 = s1 + inc

        if (pawns & Pawn.duo[s1]) != 0 and Pawn.is_safe(s1, sd, bd):
            return False
        if rk == Square.RANK_2 and (pawns & Pawn.duo[s2]) != 0 and Pawn.is_safe(s1, sd, bd) and Pawn.is_safe(s2, sd, bd):
            return False

        # Can be defended in one move?
        if fl != Square.FILE_A:
            s0 = sq + Square.INC_LEFT
            s1 = s0 - inc
            s2 = s1 - inc
            s3 = s2 - inc

            if bd.square_is(s2, Piece.PAWN, sd) and Pawn.is_safe(s1, sd, bd):
                return False

            if rk == Square.RANK_5 and bd.square_is(s3, Piece.PAWN, sd) and Pawn.is_safe(s2, sd, bd) and Pawn.is_safe(s1, sd, bd):
                return False

        if fl != Square.FILE_H:
            s0 = sq + Square.INC_RIGHT
            s1 = s0 - inc
            s2 = s1 - inc
            s3 = s2 - inc

            if bd.square_is(s2, Piece.PAWN, sd) and Pawn.is_safe(s1, sd, bd):
                return False

            if rk == Square.RANK_5 and bd.square_is(s3, Piece.PAWN, sd) and Pawn.is_safe(s2, sd, bd) and Pawn.is_safe(s1, sd, bd):
                return False

        return True

    @staticmethod
    def is_doubled(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a pawn is doubled.
        """
        fl = Square.file(sq)
        return (bd.piece(Piece.PAWN, sd) & Bit.file(fl) & Bit.rear_side(sq, sd)) != 0

    @staticmethod
    def is_blocked(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        """
        Check if a pawn is blocked.
        """
        return not Pawn.is_safe(Square.stop(sq, sd), sd, bd) and not Pawn.is_attacked(sq, Side.opposit(sd), bd)

    @staticmethod
    def shelter_file(fl: int, sd: int, bd: 'Board.BOARD') -> int:
        """
        Calculate shelter value for a specific file.
        """
        assert 0 <= fl < 8, "File index out of range."

        if bd.square_is(Square.make(fl, Square.RANK_2, sd), Piece.PAWN, sd):
            return 2
        elif bd.square_is(Square.make(fl, Square.RANK_3, sd), Piece.PAWN, sd):
            return 1
        else:
            return 0

    @staticmethod
    def shelter_files(fl: int, sd: int, bd: 'Board.BOARD') -> int:
        """
        Calculate shelter score for adjacent files.
        """
        fl_left = fl + 1 if fl == Square.FILE_A else fl - 1
        fl_right = fl - 1 if fl == Square.FILE_H else fl + 1

        sc = Pawn.shelter_file(fl, sd, bd) * 2 + \
             Pawn.shelter_file(fl_left, sd, bd) + \
             Pawn.shelter_file(fl_right, sd, bd)
        assert 0 <= sc <= 8, "Shelter score out of range."

        return sc

    @staticmethod
    def comp_info(info: 'Pawn.Info', bd: 'Board.BOARD'):
        """
        Compute pawn-related information based on the current board state.
        """
        # Pawn.clear_info(info)
        info.passed = 0
        info.safe = 0

        info.mg = 0
        info.eg = 0
        
        info.left_file = Square.FILE_H + 1
        info.right_file = Square.FILE_A - 1

        for sd in range(Side.SIZE):
            info.target[sd] = 0

        weak = 0
        strong = 0
        safe = [(1 << 64) - 1 for _ in range(Side.SIZE)]  # Initialize all bits to 1

        for sd in range(Side.SIZE):
            p12 = Piece.make(Piece.PAWN, sd)

            # Defended pawns
            strong |= bd.piece(Piece.PAWN, sd) & Attack.pawn_attacks_from(sd, bd)

            # Material evaluation
            n = bd.count(Piece.PAWN, sd)
            info.mg += n * Material.score(Piece.PAWN, Stage.MG)
            info.eg += n * Material.score(Piece.PAWN, Stage.EG)

            # Iterate through all pawns of the side
            b = bd.piece(Piece.PAWN, sd)
            while b != 0:
                sq = Bit.first(b)

                fl = Square.file(sq)
                rk = Square.rank(sq, sd)

                if fl < info.left_file:
                    info.left_file = fl
                if fl > info.right_file:
                    info.right_file = fl

                # Add positional scores
                info.mg += PST.score(p12, sq, Stage.MG)
                info.eg += PST.score(p12, sq, Stage.EG)

                # Check pawn structure
                if Pawn.is_isolated(sq, sd, bd):
                    weak = Bit.set_bit(weak, sq)
                    info.mg -= 10
                    info.eg -= 20
                elif Pawn.is_weak(sq, sd, bd):
                    weak = Bit.set_bit(weak, sq)
                    info.mg -= 5
                    info.eg -= 10

                if Pawn.is_doubled(sq, sd, bd):
                    info.mg -= 5
                    info.eg -= 10

                if Pawn.is_passed(sq, sd, bd):
                    info.passed = Bit.set_bit(info.passed, sq)
                    info.mg += 10
                    info.eg += 20
                    if rk >= Square.RANK_5:
                        stop = Square.stop(sq, sd)
                        if Pawn.is_duo(sq, sd, bd) and rk <= Square.RANK_6:
                            stop += Square.pawn_inc(sd)
                        info.target[Side.opposit(sd)] = Bit.set_bit(info.target[Side.opposit(sd)], stop)

                safe[Side.opposit(sd)] &= ~Pawn.potential_attacks(sq, sd, bd)
                assert safe[Side.opposit(sd)] == safe[Side.opposit(sd)] & 0xFFFFFFFFFFFFFFFF

                # Remove the processed pawn
                b = Bit.rest(b)

            # Compute shelter for each file
            for fl in range(Square.FILE_SIZE):
                info.shelter[fl][sd] = Pawn.shelter_files(fl, sd, bd) * 4

            # Negate material scores
            info.mg = -info.mg
            info.eg = -info.eg

        weak &= ~strong  # Defended doubled pawns are not weak
        assert weak == weak & 0xFFFFFFFFFFFFFFFF
        
        assert (weak & strong) == 0, "Weak and strong pawns overlap."

        info.target[Side.WHITE] |= bd.piece(Piece.PAWN, Side.BLACK) & weak
        info.target[Side.BLACK] |= bd.piece(Piece.PAWN, Side.WHITE) & weak

        info.safe = (safe[Side.WHITE] & Bit.front(Square.RANK_4)) | \
                    (safe[Side.BLACK] & Bit.rear(Square.RANK_5))

        if info.left_file > info.right_file:  # No pawns
            info.left_file = Square.FILE_A
            info.right_file = Square.FILE_H

        assert info.left_file <= info.right_file, "Invalid file range."

        # File openness
        for sd in range(Side.SIZE):
            for fl in range(Square.FILE_SIZE):
                file_bb = Bit.file(fl)
                if (bd.piece(Piece.PAWN, sd) & file_bb) != 0:
                    open_val = 0
                elif (bd.piece(Piece.PAWN, Side.opposit(sd)) & file_bb) == 0:
                    open_val = 4
                elif (strong & file_bb) != 0:
                    open_val = 1
                elif (weak & file_bb) != 0:
                    open_val = 3
                else:
                    open_val = 2
                info.open_file[fl][sd] = open_val * 5

    @staticmethod
    def init():
        """
        Initialize precomputed bitboards for pawn evaluations.
        """
        for sq in range(Square.SIZE):
            fl = Square.file(sq)
            rk = Square.rank(sq)

            Pawn.passed_me[sq][Side.WHITE] = Bit.file(fl) & Bit.front(rk)
            Pawn.passed_me[sq][Side.BLACK] = Bit.file(fl) & Bit.rear(rk)

            Pawn.passed_opp[sq][Side.WHITE] = Bit.files(fl) & Bit.front(rk)
            Pawn.passed_opp[sq][Side.BLACK] = Bit.files(fl) & Bit.rear(rk)

            b = 0
            if fl != Square.FILE_A:
                b = Bit.set_bit(b, sq + Square.INC_LEFT)
            if fl != Square.FILE_H:
                b = Bit.set_bit(b, sq + Square.INC_RIGHT)
            Pawn.duo[sq] = b


class Sort:
    @dataclass
    class KillerList:
        k0: int = Move.NONE
        k1: int = Move.NONE

    class Killer:
        PLY = 100

        def __init__(self):
            self.p_killer: List[Sort.KillerList] = [Sort.KillerList() for _ in range(Sort.Killer.PLY)]

        def clear(self):
            """
            Clear all killer moves for each ply.
            """
            for killer in self.p_killer:
                killer.k0 = Move.NONE
                killer.k1 = Move.NONE

        def add(self, mv: int, ply: int):
            """
            Add a killer move for a specific ply.
            """
            assert 0 <= ply < Sort.Killer.PLY, "Ply out of range."
            if self.p_killer[ply].k0 != mv:
                self.p_killer[ply].k1 = self.p_killer[ply].k0
                self.p_killer[ply].k0 = mv

        def killer_0(self, ply: int) -> int:
            """
            Get the primary killer move for a specific ply.
            """
            assert 0 <= ply < Sort.Killer.PLY, "Ply out of range."
            return self.p_killer[ply].k0

        def killer_1(self, ply: int) -> int:
            """
            Get the secondary killer move for a specific ply.
            """
            assert 0 <= ply < Sort.Killer.PLY, "Ply out of range."
            return self.p_killer[ply].k1
        
    class History:
        PROB_BIT = 11
        PROB_ONE = 1 << PROB_BIT
        PROB_HALF = 1 << (PROB_BIT - 1)
        PROB_SHIFT = 5

        def __init__(self):
            # Initialize p_prob with PROB_HALF for all possible move indices
            self.p_prob: List[int] = [Sort.History.PROB_HALF] * (Piece.SIDE_SIZE * Square.SIZE)

        @staticmethod
        def index(mv: int, bd: 'Board.BOARD') -> int:
            """
            Calculate the index for the history table based on the move and board state.
            """
            assert not Move.is_tactical(mv), "Move should not be tactical."

            sd = bd.square_side(Move.from_sq(mv))
            p12 = Piece.make(Move.piece(mv), sd)

            idx = p12 * Square.SIZE + Move.to_sq(mv)
            assert idx < Piece.SIDE_SIZE * Square.SIZE, "History index out of range."

            return idx

        def update_good(self, mv: int, bd: 'Board.BOARD'):
            """
            Update the history table for a good move.
            """
            if not Move.is_tactical(mv):
                idx = Sort.History.index(mv, bd)
                self.p_prob[idx] += (Sort.History.PROB_ONE - self.p_prob[idx]) >> Sort.History.PROB_SHIFT

        def update_bad(self, mv: int, bd: 'Board.BOARD'):
            """
            Update the history table for a bad move.
            """
            if not Move.is_tactical(mv):
                idx = Sort.History.index(mv, bd)
                self.p_prob[idx] -= self.p_prob[idx] >> Sort.History.PROB_SHIFT

        def clear(self):
            """
            Reset the history table to initial probabilities.
            """
            for idx in range(Piece.SIDE_SIZE * Square.SIZE):
                self.p_prob[idx] = Sort.History.PROB_HALF

        def add(self, bm: int, searched: 'Gen.List', bd: 'Board.BOARD'):
            """
            Add moves to the history table based on search results.
            """
            assert bm != Move.NONE, "Best move cannot be NONE."

            self.update_good(bm, bd)

            for pos in range(searched.size()):
                mv = searched.move(pos)
                if mv != bm:
                    self.update_bad(mv, bd)

        def score(self, mv: int, bd: 'Board.BOARD') -> int:
            """
            Retrieve the history score for a move.
            """
            idx = Sort.History.index(mv, bd)
            return self.p_prob[idx]

    @staticmethod
    def capture_score_debug(pc: int, cp: int) -> int:
        """
        Calculate debug capture score based on piece types.
        """
        sc = Piece.score(cp) * 6 + (5 - Piece.score(pc))
        assert 0 <= sc < 36, "Capture score out of valid range."
        return sc

    @staticmethod
    def promotion_score_debug(pp: int) -> int:
        """
        Calculate debug promotion score based on promoted piece type.
        """
        if pp == Piece.QUEEN:
            return 3
        elif pp == Piece.KNIGHT:
            return 2
        elif pp == Piece.ROOK:
            return 1
        elif pp == Piece.BISHOP:
            return 0
        else:
            assert False, "Invalid promotion piece type."
            return 0

    @staticmethod
    def tactical_score_debug(pc: int, cp: int, pp: int) -> int:
        """
        Calculate debug tactical score based on move characteristics.
        """
        if cp != Piece.NONE:
            sc = Sort.capture_score_debug(pc, cp) + 4
        else:
            sc = Sort.promotion_score_debug(pp)
        assert 0 <= sc < 40, "Tactical score out of valid range."
        return sc

    @staticmethod
    def capture_score(mv: int) -> int:
        """
        Calculate capture score for a move.
        """
        assert Move.is_capture(mv), "Move is not a capture."
        return Sort.capture_score_debug(Move.piece(mv), Move.cap(mv))

    @staticmethod
    def promotion_score(mv: int) -> int:
        """
        Calculate promotion score for a move.
        """
        assert Move.is_promotion(mv) and not Move.is_capture(mv), "Move is not a promotion or is a capture."
        return Sort.promotion_score_debug(Move.prom(mv))

    @staticmethod
    def tactical_score(mv: int) -> int:
        """
        Calculate tactical score for a move.
        """
        assert Move.is_tactical(mv), "Move is not tactical."
        return Sort.tactical_score_debug(Move.piece(mv), Move.cap(mv), Move.prom(mv))

    @staticmethod
    def evasion_score(mv: int, trans_move: int) -> int:
        """
        Calculate evasion score for a move.
        """
        if mv == trans_move:
            sc = Move.SCORE_MASK
        elif Move.is_tactical(mv):
            sc = Sort.tactical_score(mv) + 1
            assert 1 <= sc < 41, "Evasion score out of valid range."
        else:
            sc = 0
        return sc

    @staticmethod
    def sort_tacticals(ml: 'Gen.List'):
        """
        Sort tactical moves based on their tactical scores.
        """
        for pos in range(ml.size()):
            mv = ml.move(pos)
            sc = Sort.tactical_score(mv)
            ml.set_score(pos, sc)
        ml.sort()

    @staticmethod
    def sort_history(ml: 'Gen.List', bd: 'Board.BOARD', history: 'Sort.History'):
        """
        Sort moves based on history scores.
        """
        for pos in range(ml.size()):
            mv = ml.move(pos)
            sc = history.score(mv, bd)
            ml.set_score(pos, sc)
        ml.sort()

    @staticmethod
    def sort_evasions(ml: 'Gen.List', trans_move: int):
        """
        Sort evasion moves based on their evasion scores.
        """
        for pos in range(ml.size()):
            mv = ml.move(pos)
            sc = Sort.evasion_score(mv, trans_move)
            ml.set_score(pos, sc)
        ml.sort()


class GenSort:
    class Inst(Enum):
        GEN_EVASION = 0
        GEN_TRANS = 1
        GEN_TACTICAL = 2
        GEN_KILLER = 3
        GEN_CHECK = 4
        GEN_PAWN = 5
        GEN_QUIET = 6
        GEN_BAD = 7
        GEN_END = 8
        POST_MOVE = 9
        POST_MOVE_SEE = 10
        POST_KILLER = 11
        POST_KILLER_SEE = 12
        POST_BAD = 13

    # Move generation programs
    Prog_Main = [
        Inst.GEN_TRANS,
        Inst.POST_KILLER,
        Inst.GEN_TACTICAL,
        Inst.POST_MOVE_SEE,
        Inst.GEN_KILLER,
        Inst.POST_KILLER_SEE,
        Inst.GEN_QUIET,
        Inst.POST_MOVE_SEE,
        Inst.GEN_BAD,
        Inst.POST_BAD,
        Inst.GEN_END
    ]

    Prog_QS_Root = [
        Inst.GEN_TRANS,
        Inst.POST_KILLER,
        Inst.GEN_TACTICAL,
        Inst.POST_MOVE,
        Inst.GEN_CHECK,
        Inst.POST_KILLER,
        Inst.GEN_PAWN,
        Inst.POST_MOVE,
        Inst.GEN_END
    ]

    Prog_QS = [
        Inst.GEN_TRANS,
        Inst.POST_KILLER,
        Inst.GEN_TACTICAL,
        Inst.POST_MOVE,
        Inst.GEN_END
    ]

    Prog_Evasion = [
        Inst.GEN_EVASION,
        Inst.POST_MOVE_SEE,
        Inst.GEN_BAD,
        Inst.POST_BAD,
        Inst.GEN_END
    ]

    class List:
        def __init__(self):
            self.p_board: Optional['Board.BOARD'] = None
            self.p_attacks: Optional['Attack.Attacks'] = None
            self.p_killer: Optional['Sort.Killer'] = None
            self.p_hist: Optional['Sort.History'] = None
            self.p_trans_move: int = Move.NONE

            self.p_ip: 'GenSort.Inst' = []
            self.p_gen: 'GenSort.Inst' = GenSort.Inst.GEN_END
            self.p_post: 'GenSort.Inst' = GenSort.Inst.GEN_END

            self.p_todo: 'Gen.List' = Gen.List()
            self.p_done: 'Gen.List' = Gen.List()
            self.p_bad: 'Gen.List' = Gen.List()

            self.p_pos: int = 0
            self.p_candidate: bool = False

        def gen(self) -> bool:
            """
            Generate the next set of moves based on the current instruction pointer.
            """
            self.p_todo.clear()
            self.p_pos = 0

            if not self.p_ip:
                return False  # No more instructions

            # Fetch the next generation and post-generation instructions
            self.p_gen = self.p_ip.pop(0)
            self.p_post = self.p_ip.pop(0) if self.p_ip else GenSort.Inst.GEN_END

            if self.p_gen == GenSort.Inst.GEN_EVASION:
                Gen.add_evasions(self.p_todo, self.p_board.turn(), self.p_board, self.p_attacks)
                Sort.sort_evasions(self.p_todo, self.p_trans_move)
            elif self.p_gen == GenSort.Inst.GEN_TRANS:
                mv = self.p_trans_move
                if mv != Move.NONE and Gen.is_move(mv, self.p_board):
                    self.p_todo.add(mv)
                self.p_candidate = True
            elif self.p_gen == GenSort.Inst.GEN_TACTICAL:
                Gen.add_captures(self.p_todo, self.p_board.turn(), self.p_board)
                Gen.add_promotions(self.p_todo, self.p_board.turn(), self.p_board.empty(), self.p_board)
                Sort.sort_tacticals(self.p_todo)
                self.p_candidate = True
            elif self.p_gen == GenSort.Inst.GEN_KILLER:
                k0 = self.p_killer.killer_0(self.p_board.ply())
                if k0 != Move.NONE and Gen.is_quiet_move(k0, self.p_board):
                    self.p_todo.add(k0)
                k1 = self.p_killer.killer_1(self.p_board.ply())
                if k1 != Move.NONE and Gen.is_quiet_move(k1, self.p_board):
                    self.p_todo.add(k1)
                self.p_candidate = True
            elif self.p_gen == GenSort.Inst.GEN_CHECK:
                Gen.add_checks(self.p_todo, self.p_board.turn(), self.p_board)
                self.p_candidate = True  # Not needed yet
            elif self.p_gen == GenSort.Inst.GEN_PAWN:
                Gen.add_castling(self.p_todo, self.p_board.turn(), self.p_board)
                Gen.add_pawn_pushes(self.p_todo, self.p_board.turn(), self.p_board)
                self.p_candidate = True  # Not needed yet
            elif self.p_gen == GenSort.Inst.GEN_QUIET:
                Gen.add_quiets(self.p_todo, self.p_board.turn(), self.p_board)
                Sort.sort_history(self.p_todo, self.p_board, self.p_hist)
                self.p_candidate = False
            elif self.p_gen == GenSort.Inst.GEN_BAD:
                self.p_todo = self.p_bad
                self.p_candidate = False
            elif self.p_gen == GenSort.Inst.GEN_END:
                return False
            else:
                assert False, "Invalid generation instruction."

            return True

        def post(self, mv: int) -> bool:
            """
            Post-processing after generating a move.
            """
            assert mv != Move.NONE, "Post move cannot be NONE."

            if self.p_post == GenSort.Inst.POST_MOVE:
                if self.p_done.contain(mv):
                    return False
                if not Move.is_legal(mv, self.p_board, self.p_attacks):
                    return False
            elif self.p_post == GenSort.Inst.POST_MOVE_SEE:
                if self.p_done.contain(mv):
                    return False
                if not Move.is_legal(mv, self.p_board, self.p_attacks):
                    return False
                if not Move.is_safe(mv, self.p_board):
                    self.p_bad.add(mv)
                    return False
            elif self.p_post == GenSort.Inst.POST_KILLER:
                if self.p_done.contain(mv):
                    return False
                if not Move.is_legal(mv, self.p_board, self.p_attacks):
                    return False
                self.p_done.add(mv)
            elif self.p_post == GenSort.Inst.POST_KILLER_SEE:
                if self.p_done.contain(mv):
                    return False
                if not Move.is_legal(mv, self.p_board, self.p_attacks):
                    return False
                self.p_done.add(mv)
                if not Move.is_safe(mv, self.p_board):
                    self.p_bad.add(mv)
                    return False
            elif self.p_post == GenSort.Inst.POST_BAD:
                assert Move.is_legal(mv, self.p_board, self.p_attacks), "Move should be legal."
            else:
                assert False, "Invalid post-generation instruction."

            return True
        
        def init(self, depth: int, bd: 'Board.BOARD', attacks: 'Attack.Attacks', trans_move: int, killer: 'Sort.Killer', history: 'Sort.History', use_fp: bool = False):
            """
            Initialize the move generator with the current search context.
            """
            self.p_board = bd
            self.p_attacks = attacks
            self.p_killer = killer
            self.p_hist = history
            self.p_trans_move = trans_move

            # Determine the move generation program based on the current context
            if attacks.size != 0:  # In check
                self.p_ip = GenSort.Prog_Evasion.copy()
            elif depth < 0:
                self.p_ip = GenSort.Prog_QS.copy()
            elif depth == 0:
                self.p_ip = GenSort.Prog_QS_Root.copy()
            elif use_fp:
                self.p_ip = GenSort.Prog_QS_Root.copy()
            else:
                self.p_ip = GenSort.Prog_Main.copy()

            self.p_todo.clear()
            self.p_done.clear()
            self.p_bad.clear()

            self.p_pos = 0
            self.p_candidate = False
            
        def next_move(self) -> int:
            """
            Retrieve the next move from the generator.
            """
            while True:
                while self.p_pos >= self.p_todo.size():
                    if not self.gen():
                        return Move.NONE
                mv = self.p_todo.move(self.p_pos)
                self.p_pos += 1
                if self.post(mv):
                    return mv

        def is_candidate(self) -> bool:
            """
            Check if the current move is a candidate move.
            """
            return self.p_candidate


class PST:
    # Line tables for positional evaluations
    Knight_Line = [-4, -2,  0, +1, +1,  0, -2, -4]
    King_Line = [-3, -1,  0, +1, +1,  0, -1, -3]
    King_File = [+1, +2,  0, -2, -2,  0, +2, +1]
    King_Rank = [+1,  0, -2, -4, -6, -8, -10, -12]
    Advance_Rank = [-3, -2, -1,  0, +1, +2, +1,  0]

    # Piece-Square Tables
    p_score = [[[0 for _ in range(Stage.SIZE)] for _ in range(Square.SIZE)] for _ in range(Piece.SIDE_SIZE)]

    @staticmethod
    def score(p12: int, sq: int, stage: int) -> int:
        """
        Retrieve the positional score for a piece on a specific square and game stage.
        """
        assert 0 <= p12 < Piece.SIDE_SIZE, "Side-Piece type out of range."
        assert 0 <= sq < Square.SIZE, "Square index out of range."
        assert 0 <= stage < Stage.SIZE, "Stage out of range."
        return PST.p_score[p12][sq][stage]

    @staticmethod
    def init():
        """
        Initialize the piece-square tables with predefined values.
        """
        for p12 in range(Piece.SIDE_SIZE):
            for sq in range(Square.SIZE):
                PST.p_score[p12][sq][Stage.MG] = 0
                PST.p_score[p12][sq][Stage.EG] = 0

        for sq in range(Square.SIZE):
            fl = Square.file(sq)
            rk = Square.rank(sq)
            
            # White Pawn
            PST.p_score[Piece.WHITE_PAWN][sq][Stage.MG] = 0
            PST.p_score[Piece.WHITE_PAWN][sq][Stage.EG] = 0

            # White Knight
            PST.p_score[Piece.WHITE_KNIGHT][sq][Stage.MG] = (PST.Knight_Line[fl] + PST.Knight_Line[rk] + PST.Advance_Rank[rk]) * 4
            PST.p_score[Piece.WHITE_KNIGHT][sq][Stage.EG] = (PST.Knight_Line[fl] + PST.Knight_Line[rk] + PST.Advance_Rank[rk]) * 4

            # White Bishop
            PST.p_score[Piece.WHITE_BISHOP][sq][Stage.MG] = (PST.King_Line[fl] + PST.King_Line[rk]) * 2
            PST.p_score[Piece.WHITE_BISHOP][sq][Stage.EG] = (PST.King_Line[fl] + PST.King_Line[rk]) * 2

            # White Rook
            PST.p_score[Piece.WHITE_ROOK][sq][Stage.MG] = PST.King_Line[fl] * 5
            PST.p_score[Piece.WHITE_ROOK][sq][Stage.EG] = 0

            # White Queen
            PST.p_score[Piece.WHITE_QUEEN][sq][Stage.MG] = (PST.King_Line[fl] + PST.King_Line[rk]) * 1
            PST.p_score[Piece.WHITE_QUEEN][sq][Stage.EG] = (PST.King_Line[fl] + PST.King_Line[rk]) * 1

            # White King
            PST.p_score[Piece.WHITE_KING][sq][Stage.MG] = (PST.King_File[fl] + PST.King_Rank[rk]) * 8
            PST.p_score[Piece.WHITE_KING][sq][Stage.EG] = (PST.King_Line[fl] + PST.King_Line[rk] + PST.Advance_Rank[rk]) * 8

        # Mirror scores for Black pieces
        for pc in range(Piece.PAWN, Piece.KING + 1):
            wp = Piece.make(pc, Side.WHITE)
            bp = Piece.make(pc, Side.BLACK)
            for sq in range(Square.SIZE):
                ws = Square.opposit_rank(sq)
                PST.p_score[bp][sq][Stage.MG] = PST.p_score[wp][ws][Stage.MG]
                PST.p_score[bp][sq][Stage.EG] = PST.p_score[wp][ws][Stage.EG]


class Eval:
    @dataclass
    class Entry:
        lock: int = 0
        _eval: int = 0

    @dataclass
    class AttackInfo:
        piece_attacks: List[int] = field(default_factory=lambda: [0 for _ in range(Square.SIZE)])
        all_attacks: List[int] = field(default_factory=lambda: [0 for _ in range(Side.SIZE)])
        multiple_attacks: List[int] = field(default_factory=lambda: [0 for _ in range(Side.SIZE)])
        ge_pieces: List[List[int]] = field(default_factory=lambda: [[0 for _ in range(Piece.SIZE)] for _ in range(Side.SIZE)])
        lt_attacks: List[List[int]] = field(default_factory=lambda: [[0 for _ in range(Piece.SIZE)] for _ in range(Side.SIZE)])
        le_attacks: List[List[int]] = field(default_factory=lambda: [[0 for _ in range(Piece.SIZE)] for _ in range(Side.SIZE)])
        king_evasions: List[int] = field(default_factory=lambda: [0 for _ in range(Side.SIZE)])
        pinned: int = 0

    class Table:
        BITS = 16
        SIZE = 1 << BITS
        MASK = SIZE - 1
    
        def __init__(self):
            self.p_table: List['Eval.Entry'] = [Eval.Entry() for _ in range(Eval.Table.SIZE)]
    
        def clear(self):
            for entry in self.p_table:
                entry.lock = 0
                entry._eval = 0
    
        def evaluate(self, bd: 'Board.BOARD', pawn_table: 'Pawn.Table') -> int:
            key = bd.eval_key()
    
            index = Hash.index(key) & Eval.Table.MASK
            lock = Hash.lock(key)
    
            entry = self.p_table[index]
    
            if entry.lock == lock:
                return entry._eval
    
            eval_val = Eval.comp_eval(bd, pawn_table)
    
            entry.lock = lock
            entry._eval = eval_val
    
            return eval_val

    # Evaluation Weights
    attack_weight = [0, 4, 4, 2, 1, 4, 0]
    attacked_weight = [0, 1, 1, 2, 4, 8, 0]

    mob_weight = [0 for _ in range(32)]
    dist_weight = [0 for _ in range(8)]  # for king-passer distance

    # Center Bitboards
    small_centre = 0
    medium_centre = 0
    large_centre = 0
    centre_0 = 0
    centre_1 = 0

    # Area Bitboards
    side_area = [0 for _ in range(Side.SIZE)]
    king_area = [[0 for _ in range(Square.SIZE)] for _ in range(Side.SIZE)]

    @staticmethod
    def comp_attacks(ai: Eval.AttackInfo, bd: 'Board.BOARD'):
        # Prepare for adding defended opponent pieces
        for sd in range(Side.SIZE):
            b = 0
            for pc in range(Piece.KING, Piece.KNIGHT - 1, -1):
                b |= bd.piece(pc, sd)
                ai.ge_pieces[sd][pc] = b
            ai.ge_pieces[sd][Piece.BISHOP] = ai.ge_pieces[sd][Piece.KNIGHT]  # Minors are equal

        # Pawn attacks
        pc = Piece.PAWN
        for sd in range(Side.SIZE):
            b = Attack.pawn_attacks_from(sd, bd)
            ai.lt_attacks[sd][pc] = 0  # Not needed
            ai.le_attacks[sd][pc] = b
            ai.all_attacks[sd] = b

        # Piece attacks
        ai.multiple_attacks[Side.WHITE] = 0
        ai.multiple_attacks[Side.BLACK] = 0

        for pc in range(Piece.KNIGHT, Piece.KING + 1):
            lower_piece = Piece.PAWN if pc == Piece.BISHOP else pc - 1
            assert lower_piece >= Piece.PAWN and lower_piece < pc, "lower_piece out of range"

            for sd in range(Side.SIZE):
                ai.lt_attacks[sd][pc] = ai.le_attacks[sd][lower_piece]

            for sd in range(Side.SIZE):
                fs = bd.piece(pc, sd)
                while fs != 0:
                    sq = Bit.first(fs)
                    ts = Attack.piece_attacks_from(pc, sq, bd)
                    ai.piece_attacks[sq] = ts

                    ai.multiple_attacks[sd] |= ts & ai.all_attacks[sd]
                    ai.all_attacks[sd] |= ts
                    fs = Bit.rest(fs)

                ai.le_attacks[sd][pc] = ai.all_attacks[sd]
                assert (ai.le_attacks[sd][pc] & ai.lt_attacks[sd][pc]) == ai.lt_attacks[sd][pc], "Bitwise AND failed"

                if pc == Piece.BISHOP:
                    ai.le_attacks[sd][Piece.KNIGHT] = ai.le_attacks[sd][Piece.BISHOP]

        # King evasions
        for sd in range(Side.SIZE):
            king = bd.king(sd)
            ts = Attack.pseudo_attacks_from(Piece.KING, sd, king)
            ai.king_evasions[sd] = ts & ~bd.side(sd) & ~ai.all_attacks[Side.opposit(sd)]
            assert ai.king_evasions[sd] == ai.king_evasions[sd] & 0xFFFFFFFFFFFFFFFF

        # Pinned pieces
        ai.pinned = 0
        for sd in range(Side.SIZE):
            king_sq = bd.king(sd)
            pinned_bits = Attack.pinned_by(king_sq, Side.opposit(sd), bd)
            ai.pinned |= bd.side(sd) & pinned_bits

    @staticmethod
    def mul_shift(a: int, b: int, c: int) -> int:
        bias = 1 << (c - 1)
        return (a * b + bias) >> c

    @staticmethod
    def passed_score(sc: int, rk: int) -> int:
        passed_weight = [0, 0, 0, 2, 6, 12, 20, 0]
        return Eval.mul_shift(sc, passed_weight[rk], 4)

    @staticmethod
    def mobility_score(pc: int, ts: int) -> int:
        mob = Bit.count(ts)
        return Eval.mul_shift(20, Eval.mob_weight[mob], 8)

    @staticmethod
    def attack_mg_score(pc: int, sd: int, ts: int) -> int:
        assert pc < Piece.SIZE

        c0 = Bit.count(ts & Eval.centre_0)
        c1 = Bit.count(ts & Eval.centre_1)
        sc = c1 * 2 + c0

        sc += Bit.count(ts & Eval.side_area[Side.opposit(sd)])
        
        if (sc - 4) * Eval.attack_weight[pc] < 0:
            return int((sc - 4) * Eval.attack_weight[pc] / 2)
        else:
            return (sc - 4) * Eval.attack_weight[pc] // 2

    @staticmethod
    def attack_eg_score(pc: int, sd: int, ts: int, pi: 'Pawn.Info') -> int:
        assert pc < Piece.SIZE
        return Bit.count(ts & pi.target[sd]) * Eval.attack_weight[pc] * 4

    @staticmethod
    def capture_score(pc: int, sd: int, ts: int, bd: 'Board.BOARD', ai: AttackInfo) -> int:
        assert pc < Piece.SIZE

        sc = 0
        b = ts & bd.pieces(Side.opposit(sd))
        while b != 0:
            t = Bit.first(b)
            cp = bd.square(t)
            sc += Eval.attacked_weight[cp]
            if Bit.is_set(ai.pinned, t):
                sc += Eval.attacked_weight[cp] * 2
            b = Bit.rest(b)

        return Eval.attack_weight[pc] * sc * 4

    @staticmethod
    def shelter_score(sq: int, sd: int, bd: 'Board.BOARD', pi: 'Pawn.Info') -> int:
        if Square.rank(sq, sd) > Square.RANK_2:
            return 0

        s0 = pi.shelter[Square.file(sq)][sd]
        s1 = 0

        for wg in range(Wing.SIZE):
            index = Castling.index(sd, wg)
            if Castling.flag(bd.flags(), index):
                fl = Wing.shelter_file[wg]
                s1 = max(s1, pi.shelter[fl][sd])

        if s1 > s0:
            assert (s0 + s1) >= 0
            return (s0 + s1) // 2
        else:
            return s0

    @staticmethod
    def king_score(sc: int, n: int) -> int:
        weight = 256 - (256 >> n)
        return Eval.mul_shift(sc, weight, 8)

    @staticmethod
    def eval_outpost(sq: int, sd: int, bd: 'Board.BOARD', pi: 'Pawn.Info') -> int:
        assert Square.rank(sq, sd) >= Square.RANK_5

        xd = Side.opposit(sd)

        weight = 0

        if Bit.is_set(pi.safe, sq):
            weight += 2

        if bd.square_is(Square.stop(sq, sd), Piece.PAWN, xd):
            weight += 1

        if Pawn.is_attacked(sq, sd, bd):
            weight += 1

        return weight - 2

    @staticmethod
    def passer_is_unstoppable(sq: int, sd: int, bd: 'Board.BOARD') -> bool:
        if not Material.lone_king(Side.opposit(sd), bd):
            return False

        fl = Square.file(sq)
        front = Bit.file(fl) & Bit.front_side(sq, sd)

        if (bd.all_pieces() & front) != 0:
            return False

        if Pawn.square_distance(bd.king(Side.opposit(sd)), sq, sd) >= 2:
            return True

        king_attacks = Attack.pseudo_attacks_from(Piece.KING, sd, bd.king(sd))
        
        assert (front & ~king_attacks) == (front & ~king_attacks) & 0xFFFFFFFFFFFFFFFF
        if (front & ~king_attacks) == 0:
            return True

        return False

    @staticmethod
    def eval_passed(sq: int, sd: int, bd: 'Board.BOARD', ai: AttackInfo) -> int:
        fl = Square.file(sq)
        xd = Side.opposit(sd)

        weight = 4

        # Blocker
        stop_sq = Square.stop(sq, sd)
        if bd.square(stop_sq) != Piece.NONE:
            weight -= 1

        # Free Path
        front = Bit.file(fl) & Bit.front_side(sq, sd)
        rear = Bit.file(fl) & Bit.rear_side(sq, sd)

        if (bd.all_pieces() & front) == 0:
            major_behind = False
            majors = bd.piece(Piece.ROOK, xd) | bd.piece(Piece.QUEEN, xd)

            b = majors & rear
            while b != 0:
                f = Bit.first(b)
                if Attack.line_is_empty(f, sq, bd):
                    major_behind = True
                b = Bit.rest(b)

            if not major_behind and (ai.all_attacks[xd] & front) == 0:
                weight += 2

        return weight

    @staticmethod
    def eval_pawn_cap(sd: int, bd: 'Board.BOARD', ai: AttackInfo) -> int:
        ts = Attack.pawn_attacks_from(sd, bd)
        sc = 0

        b = ts & bd.pieces(Side.opposit(sd))
        while b != 0:
            t = Bit.first(b)
            cp = bd.square(t)
            if cp == Piece.KING:
                b = Bit.rest(b)
                continue

            sc += Piece.value(cp) - 50
            if Bit.is_set(ai.pinned, t):
                sc += (Piece.value(cp) - 50) * 2

            b = Bit.rest(b)
            
        assert sc >= 0
        return sc // 10

    @staticmethod
    def eval_pattern(bd: 'Board.BOARD') -> int:
        eval_val = 0

        # Fianchetto Patterns
        if (bd.square_is(Square.B2, Piece.BISHOP, Side.WHITE) and
            bd.square_is(Square.B3, Piece.PAWN, Side.WHITE) and
            bd.square_is(Square.C2, Piece.PAWN, Side.WHITE)):
            eval_val += 20

        if (bd.square_is(Square.G2, Piece.BISHOP, Side.WHITE) and
            bd.square_is(Square.G3, Piece.PAWN, Side.WHITE) and
            bd.square_is(Square.F2, Piece.PAWN, Side.WHITE)):
            eval_val += 20

        if (bd.square_is(Square.B7, Piece.BISHOP, Side.BLACK) and
            bd.square_is(Square.B6, Piece.PAWN, Side.BLACK) and
            bd.square_is(Square.C7, Piece.PAWN, Side.BLACK)):
            eval_val -= 20

        if (bd.square_is(Square.G7, Piece.BISHOP, Side.BLACK) and
            bd.square_is(Square.G6, Piece.PAWN, Side.BLACK) and
            bd.square_is(Square.F7, Piece.PAWN, Side.BLACK)):
            eval_val -= 20

        return eval_val

    @staticmethod
    def has_minor(sd: int, bd: 'Board.BOARD') -> bool:
        return (bd.count(Piece.KNIGHT, sd) + bd.count(Piece.BISHOP, sd)) != 0

    @staticmethod
    def draw_mul(sd: int, bd: 'Board.BOARD', pi: 'Pawn.Info') -> int:
        xd = Side.opposit(sd)

        pawn_counts = [bd.count(Piece.PAWN, Side.WHITE), bd.count(Piece.PAWN, Side.BLACK)]

        force = Material.force(sd, bd) - Material.force(xd, bd)

        # Rook-file Pawns
        if Material.lone_king_or_bishop(sd, bd) and pawn_counts[sd] != 0:
            b = bd.piece(Piece.BISHOP, sd)

            # Queenside
            assert (bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_A)) == (bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_A)) & 0xFFFFFFFFFFFFFFFF
            if ((bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_A)) == 0 and
                (bd.piece(Piece.PAWN, xd) & Bit.file(Square.FILE_B)) == 0):
                prom = Square.A8 if sd == Side.WHITE else Square.A1
                if Square.distance(bd.king(xd), prom) <= 1:
                    if b == 0 or not Square.same_colour(Bit.first(b), prom):
                        return 1

            # Kingside
            assert (bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_H)) == (bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_H)) & 0xFFFFFFFFFFFFFFFF
            if ((bd.piece(Piece.PAWN, sd) & ~Bit.file(Square.FILE_H)) == 0 and
                (bd.piece(Piece.PAWN, xd) & Bit.file(Square.FILE_G)) == 0):
                prom = Square.H8 if sd == Side.WHITE else Square.H1
                if Square.distance(bd.king(xd), prom) <= 1:
                    if b == 0 or not Square.same_colour(Bit.first(b), prom):
                        return 1

        # Lone King or Minor
        if pawn_counts[sd] == 0 and Material.lone_king_or_minor(sd, bd):
            return 1

        # Two Knights
        if pawn_counts[sd] == 0 and Material.two_knights(sd, bd):
            return 2

        # Force <= 1
        if pawn_counts[sd] == 0 and force <= 1:
            return 2

        # One Pawn and No Force with Opponent Minor
        if pawn_counts[sd] == 1 and force == 0 and Eval.has_minor(xd, bd):
            return 4

        # One Pawn and No Force
        if pawn_counts[sd] == 1 and force == 0:
            king = bd.king(xd)
            pawn = Bit.first(bd.piece(Piece.PAWN, sd))
            stop = Square.stop(pawn, sd)

            if king == stop or (Square.rank(pawn, sd) <= Square.RANK_6 and king == Square.stop(stop, sd)):
                return 4

        # Two Pawns with Opponent's Pawn and Minor
        if (pawn_counts[sd] == 2 and pawn_counts[xd] >= 1 and force == 0 and
            Eval.has_minor(xd, bd) and (bd.piece(Piece.PAWN, sd) & pi.passed) == 0):
            return 8

        # Opposite-color Bishops
        if (Material.lone_bishop(Side.WHITE, bd) and
            Material.lone_bishop(Side.BLACK, bd) and
            abs(pawn_counts[Side.WHITE] - pawn_counts[Side.BLACK]) <= 2):
            wb = Bit.first(bd.piece(Piece.BISHOP, Side.WHITE))
            bb = Bit.first(bd.piece(Piece.BISHOP, Side.BLACK))
            if not Square.same_colour(wb, bb):
                return 8

        return 16

    @staticmethod
    def my_distance(f: int, t: int, weight: int) -> int:
        dist = Square.distance(f, t)
        return Eval.mul_shift(Eval.dist_weight[dist], weight, 8)

    @staticmethod
    def check_number(pc: int, sd: int, ts: int, king: int, bd: 'Board.BOARD') -> int:
        assert pc != Piece.KING

        xd = Side.opposit(sd)
        checks = ts & ~bd.side(sd) & Attack.pseudo_attacks_to(pc, sd, king)
        assert checks == checks & 0xFFFFFFFFFFFFFFFF

        if not Piece.is_slider(pc):
            return Bit.count(checks)

        n = 0

        # Contact Checks
        b = checks & Attack.pseudo_attacks_to(Piece.KING, xd, king)
        n += Bit.count(b) * 2
        checks &= ~b
        assert checks == checks & 0xFFFFFFFFFFFFFFFF

        # Sliding Checks
        while checks != 0:
            t = Bit.first(checks)
            if Attack.line_is_empty(t, king, bd):
                n += 1
            checks = Bit.rest(checks)

        return n

    @staticmethod
    def comp_eval(bd: 'Board.BOARD', pawn_table: 'Pawn.Table') -> int:
        # NOTE: score for white
        ai = Eval.AttackInfo()
        Eval.comp_attacks(ai, bd)

        pi = pawn_table.info(bd)

        eval_val = 0
        mg = 0
        eg = 0

        shelter = [0 for _ in range(Side.SIZE)]

        # Compute shelter scores
        for sd in range(Side.SIZE):
            shelter[sd] = Eval.shelter_score(bd.king(sd), sd, bd, pi)

        # Evaluate each side
        for sd in range(Side.SIZE):
            xd = Side.opposit(sd)

            my_king = bd.king(sd)
            op_king = bd.king(xd)

            target = ~(bd.piece(Piece.PAWN, sd) | Attack.pawn_attacks_from(xd, bd)) & 0xFFFFFFFFFFFFFFFF

            king_n = 0
            king_power = 0

            # Pawns Evaluation
            fs = bd.piece(Piece.PAWN, sd)
            front = Bit.front(Square.RANK_3) if sd == Side.WHITE else Bit.rear(Square.RANK_6)
            b = fs & pi.passed & front

            while b != 0:
                sq = Bit.first(b)
                rk = Square.rank(sq, sd)

                if Eval.passer_is_unstoppable(sq, sd, bd):
                    weight = max(rk - Square.RANK_3, 0)
                    assert 0 <= weight < 5, "Weight out of range"
                    eg += (Piece.QUEEN_VALUE - Piece.PAWN_VALUE) * weight // 5
                else:
                    sc = Eval.eval_passed(sq, sd, bd, ai)

                    sc_mg = sc * 20
                    sc_eg = sc * 25

                    stop = Square.stop(sq, sd)
                    sc_eg -= Eval.my_distance(my_king, stop, 10)
                    sc_eg += Eval.my_distance(op_king, stop, 20)

                    mg += Eval.passed_score(sc_mg, rk)
                    eg += Eval.passed_score(sc_eg, rk)

                b = Bit.rest(b)

            # Pawn Moves and Captures
            pawn_moves = Attack.pawn_moves_from(sd, bd) & bd.empty()
            eval_val += Bit.count(pawn_moves) * 4 - bd.count(Piece.PAWN, sd) * 2
            eval_val += Eval.eval_pawn_cap(sd, bd, ai)

            # Pieces Evaluation
            for pc in range(Piece.KNIGHT, Piece.KING + 1):
                p12 = Piece.make(pc, sd)  # For PST

                n_pieces = bd.count(pc, sd)
                mg += Material.score(pc, Stage.MG) * n_pieces
                eg += Material.score(pc, Stage.EG) * n_pieces

                b = bd.piece(pc, sd)
                while b != 0:
                    sq = Bit.first(b)
                    fl = Square.file(sq)
                    rk = Square.rank(sq, sd)

                    # Compute safe attacks
                    ts_all = ai.piece_attacks[sq]
                    ts_pawn_safe = ts_all & target

                    safe = (~ai.all_attacks[xd] | ai.multiple_attacks[sd]) & 0xFFFFFFFFFFFFFFFF

                    if Piece.is_slider(pc):
                        # Battery support
                        bishops = bd.piece(Piece.BISHOP, sd) | bd.piece(Piece.QUEEN, sd)
                        rooks = bd.piece(Piece.ROOK, sd) | bd.piece(Piece.QUEEN, sd)

                        support = 0
                        support |= bishops & Attack.pseudo_attacks_to(Piece.BISHOP, sd, sq)
                        support |= rooks & Attack.pseudo_attacks_to(Piece.ROOK, sd, sq)

                        inner_b = ts_all & support
                        while inner_b != 0:
                            f = Bit.first(inner_b)
                            assert Attack.line_is_empty(f, sq, bd), "Line not empty in battery support"
                            safe |= Attack.Behind[f][sq]
                            inner_b = Bit.rest(inner_b)

                    ts_safe = ts_pawn_safe & ~ai.lt_attacks[xd][pc] & safe
                    assert ts_safe == ts_safe & 0xFFFFFFFFFFFFFFFF

                    mg += PST.score(p12, sq, Stage.MG)
                    eg += PST.score(p12, sq, Stage.EG)

                    if pc == Piece.KING:
                        eg += Eval.mobility_score(pc, ts_safe)
                    else:
                        eval_val += Eval.mobility_score(pc, ts_safe)

                    if pc != Piece.KING:
                        mg += Eval.attack_mg_score(pc, sd, ts_pawn_safe)

                    eg += Eval.attack_eg_score(pc, sd, ts_pawn_safe, pi)

                    capture_ts = ts_all & (ai.ge_pieces[xd][pc] | target)
                    eval_val += Eval.capture_score(pc, sd, capture_ts, bd, ai)

                    if pc != Piece.KING:
                        check_num = Eval.check_number(pc, sd, ts_safe, op_king, bd)
                        eval_val += check_num * Material.power(pc) * 6

                    if pc != Piece.KING and (ts_safe & Eval.king_area[xd][op_king]) != 0:
                        # King attack
                        king_n += 1
                        king_power += Material.power(pc)

                    if Piece.is_minor(pc) and Square.RANK_5 <= rk <= Square.RANK_6 and Square.FILE_C <= fl <= Square.FILE_F:
                        # Outpost
                        eval_val += Eval.eval_outpost(sq, sd, bd, pi) * 5

                    if Piece.is_minor(pc) and Square.RANK_5 <= rk and not Bit.is_set(ai.all_attacks[sd], sq):
                        # Loose minor
                        mg -= 10

                    if Piece.is_minor(pc) and Square.RANK_3 <= rk <= Square.RANK_4 and \
                       bd.square_is(Square.stop(sq, sd), Piece.PAWN, sd):
                        # Shielded minor
                        mg += 10

                    if pc == Piece.ROOK:
                        # Open file
                        sc = pi.open_file[fl][sd]

                        minors = bd.piece(Piece.KNIGHT, xd) | bd.piece(Piece.BISHOP, xd)
                        if sc >= 10 and (minors & Bit.file(fl) & ~target) != 0:
                            # Blocked by minor
                            sc = 5

                        eval_val += sc - 10

                        if sc >= 10 and abs(Square.file(op_king) - fl) <= 1:
                            # Open file on king
                            weight = 2 if Square.file(op_king) == fl else 1
                            assert sc * weight >= 0
                            mg += sc * weight // 2

                    if pc == Piece.ROOK and Square.rank(sq, sd) == Square.RANK_7:
                        # 7th rank
                        pawns = bd.piece(Piece.PAWN, xd) & Bit.rank(Square.rank(sq))
                        if Square.rank(op_king, sd) >= Square.RANK_7 or pawns != 0:
                            mg += 10
                            eg += 20

                    if pc == Piece.KING:
                        # King out
                        dl = (pi.left_file - 1) - fl
                        if dl > 0:
                            eg -= dl * 20
                        dr = fl - (pi.right_file + 1)
                        if dr > 0:
                            eg -= dr * 20

                    b = Bit.rest(b)

            if bd.count(Piece.BISHOP, sd) >= 2:
                mg += 30
                eg += 50

            mg += shelter[sd]
            mg += Eval.mul_shift(Eval.king_score(king_power * 30, king_n), 32 - shelter[xd], 5)

            # Flip perspective to the opponent
            eval_val = -eval_val
            mg = -mg
            eg = -eg

        mg += pi.mg
        eg += pi.eg

        eval_val += Eval.eval_pattern(bd)

        eval_val += Material.interpolation(mg, eg, bd)

        if eval_val != 0:
            winner = Side.WHITE if eval_val > 0 else Side.BLACK
            eval_val = Eval.mul_shift(eval_val, Eval.draw_mul(winner, bd, pi), 4)

        assert Score.EVAL_MIN <= eval_val <= Score.EVAL_MAX, "Eval out of range"

        return eval_val

    @staticmethod
    def evaluate(bd: 'Board.BOARD', table: 'Eval.Table', pawn_table: 'Pawn.Table') -> int:
        return Score.side_score(table.evaluate(bd, pawn_table), bd.turn())

    @staticmethod
    def init():
        # init_centres
        Eval.small_centre = 0
        Eval.medium_centre = 0
        Eval.large_centre = 0

        for sq in range(Square.SIZE):
            fl = Square.file(sq)
            rk = Square.rank(sq)
            if Square.FILE_D <= fl <= Square.FILE_E and Square.RANK_4 <= rk <= Square.RANK_5:
                Eval.small_centre = Bit.set_bit(Eval.small_centre, sq)
            if Square.FILE_C <= fl <= Square.FILE_F and Square.RANK_3 <= rk <= Square.RANK_6:
                Eval.medium_centre = Bit.set_bit(Eval.medium_centre, sq)
            if Square.FILE_B <= fl <= Square.FILE_G and Square.RANK_2 <= rk <= Square.RANK_7:
                Eval.large_centre = Bit.set_bit(Eval.large_centre, sq)

        Eval.large_centre &= ~Eval.medium_centre
        Eval.medium_centre &= ~Eval.small_centre

        Eval.centre_0 = Eval.small_centre | Eval.large_centre
        Eval.centre_1 = Eval.small_centre | Eval.medium_centre

        # init_side_area
        Eval.side_area[Side.WHITE] = 0
        Eval.side_area[Side.BLACK] = 0

        for sq in range(Square.SIZE):
            rk = Square.rank(sq)
            if rk <= Square.RANK_4:
                Eval.side_area[Side.WHITE] = Bit.set_bit(Eval.side_area[Side.WHITE], sq)
            else:
                Eval.side_area[Side.BLACK] = Bit.set_bit(Eval.side_area[Side.BLACK], sq)

        # init_king_area
        for ks in range(Square.SIZE):
            Eval.king_area[Side.WHITE][ks] = 0
            Eval.king_area[Side.BLACK][ks] = 0
            
            for asq in range(Square.SIZE):
                df = Square.file(asq) - Square.file(ks)
                dr = Square.rank(asq) - Square.rank(ks)
                if abs(df) <= 1 and -1 <= dr <= 2:
                    Eval.king_area[Side.WHITE][ks] = Bit.set_bit(Eval.king_area[Side.WHITE][ks], asq)
                if abs(df) <= 1 and -2 <= dr <= 1:
                    Eval.king_area[Side.BLACK][ks] = Bit.set_bit(Eval.king_area[Side.BLACK][ks], asq)
        
        # init_weights
        for i in range(32):
            x = float(i) * 0.5
            y = 1.0 - math.exp(-x)
            Eval.mob_weight[i] = Util.round(y * 512.0) - 256

        for i in range(8):
            x = float(i) - 3.0
            y = 1.0 / (1.0 + math.exp(-x))
            Eval.dist_weight[i] = Util.round(y * 7.0 * 256.0)


class Search:
    # Constants
    MAX_DEPTH = 100
    MAX_PLY = 100
    NODE_PERIOD = 1024
    MAX_THREADS = 16

    # Exception Class
    class Abort(Exception):
        """SP fail-high exception"""
        pass
    
    # PV Class
    class PV:
        def __init__(self):
            self.SIZE: int = Search.MAX_PLY
            self.p_move: List[int] = [Move.NONE] * self.SIZE
            self.p_size: int = 0
            # self.clear()

        def assign(self, pv: 'Search.PV'):
            """Assign another PV to this PV."""
            self.clear()
            self.add_pv(pv)

        def clear(self):
            self.p_size = 0
            
        def add_move(self, mv: int):
            if self.p_size < self.SIZE:
                self.p_move[self.p_size] = mv
                self.p_size += 1

        def add_pv(self, pv: 'Search.PV'):
            for pos in range(pv.size()):
                mv = pv.move(pos)
                self.add_move(mv)

        def cat(self, mv: int, pv: 'Search.PV'):
            self.clear()
            self.add_move(mv)
            self.add_pv(pv)

        def size(self) -> int:
            return self.p_size

        def move(self, pos: int) -> int:
            return self.p_move[pos]

        def to_can(self) -> str:
            s = ""
            for pos in range(self.size()):
                mv = self.move(pos)
                if pos != 0:
                    s += " "
                s += Move.to_can(mv)
            return s
    
    # Data Classes
    @dataclass
    class Time:
        depth_limited: bool = False
        node_limited: bool = False
        time_limited: bool = False
        depth_limit: int = 0
        node_limit: int = 0
        time_limit: int = 0
        smart: bool = False
        ponder: bool = False
        flag: bool = False
        limit_0: int = 0
        limit_1: int = 0
        limit_2: int = 0
        last_score: int = 0
        drop: bool = False
        timer: 'Util.Timer' = field(default_factory=Util.Timer)

    @dataclass
    class Current:
        depth: int = 0
        max_ply: int = 0
        node: int = 0
        time: int = 0
        speed: int = 0

        move: int = Move.NONE
        pos: int = 0
        size: int = 0
        fail_high: bool = False

        last_time: int = 0
    
    @dataclass
    class Best:
        depth: int = 0
        move: int = Move.NONE
        score: int = Score.NONE
        flags: int = Score.FLAGS_NONE
        pv: 'Search.PV' = field(default_factory=lambda: Search.PV())

    # Search_Global Class
    class SearchGlobal(Util.Lockable):
        def __init__(self):
            super().__init__()
            self.trans: 'Trans.Table' = Trans.Table()
            self.history: 'Sort.History' = Sort.History()

    # SMP Class
    class SMP(Util.Lockable):
        def __init__(self):
            super().__init__()

    # Split_Point Class
    class SplitPoint(Util.Lockable):
        def __init__(self):
            super().__init__()
            self.p_master: Optional['Search.SearchLocal'] = None
            self.p_parent: Optional['Search.SplitPoint'] = None

            self.p_board: Board.BOARD = Board.BOARD()
            self.p_depth: int = 0
            self.p_old_alpha: int = 0
            self.p_alpha: int = 0
            self.p_beta: int = 0

            self.p_todo: Gen.List = Gen.List()
            self.p_done: Gen.List = Gen.List()

            self.p_workers: int = 0
            self.p_sent: int = 0
            self.p_received: int = 0

            self.p_bs: int = Score.NONE
            self.p_bm: int = Move.NONE
            self.p_pv: Search.PV = Search.PV()

        def init_root(self, master: 'Search.SearchLocal'):
            self.lock()
            try:
                self.p_master = master
                self.p_parent = None

                self.p_bs = Score.NONE
                self.p_beta = Score.MAX
                self.p_todo.clear()

                self.p_workers = 1
                self.p_received = -1  # HACK
            finally:
                self.unlock()

        def init(self, master: 'Search.SearchLocal', parent: 'Search.SplitPoint', bd: 'Board.BOARD', depth: int, old_alpha: int, alpha: int, beta: int,
                       todo: 'GenSort.List', done: 'Gen.List', bs: int, bm: int, pv: 'Search.PV'):
            assert depth > 4
            assert old_alpha <= alpha
            assert alpha < beta
            assert done.size() != 0
            assert bs != Score.NONE

            self.p_master = master
            self.p_parent = parent

            self.p_board.assign(bd)
            self.p_depth = depth
            self.p_old_alpha = old_alpha
            self.p_alpha = alpha
            self.p_beta = beta

            self.p_todo.clear()

            mv = todo.next_move()
            while mv != Move.NONE:
                self.p_todo.add(mv)
                mv = todo.next_move()

            self.p_done.assign(done)

            self.p_workers = 0
            self.p_sent = 0
            self.p_received = 0

            self.p_bs = bs
            self.p_bm = bm
            self.p_pv.assign(pv)

        def enter(self):
            self.lock()
            try:
                self.p_workers += 1
            finally:
                self.unlock()

        def leave(self):
            self.lock()
            try:
                assert self.p_workers > 0
                self.p_workers -= 1
                if self.p_workers == 0:
                    Search.sl_signal(self.p_master)
            finally:
                self.unlock()

        def next_move(self) -> int:
            # Not locking as per C++ comment
            mv = Move.NONE
            if self.p_bs < self.p_beta and self.p_sent < self.p_todo.size():
                mv = self.p_todo.move(self.p_sent)
                self.p_sent += 1
            return mv

        def update_root(self):
            self.lock()
            try:
                self.p_received = 0
                self.p_workers = 0
            finally:
                self.unlock()

        def update(self, mv: int, sc: int, pv: 'Search.PV'):
            self.lock()
            try:
                self.p_done.add(mv)
                assert self.p_received < self.p_todo.size()
                self.p_received += 1

                if sc > self.p_bs:
                    self.p_bs = sc
                    self.p_pv.cat(mv, pv)

                    if sc > self.p_alpha:
                        self.p_bm = mv
                        self.p_alpha = sc
            finally:
                self.unlock()

        def board(self) -> 'Board.BOARD':
            return self.p_board

        def parent(self) -> Optional['Search.SplitPoint']:
            return self.p_parent

        def depth(self) -> int:
            return self.p_depth

        def alpha(self) -> int:
            return self.p_alpha

        def beta(self) -> int:
            return self.p_beta

        def old_alpha(self) -> int:
            return self.p_old_alpha

        def bs(self) -> int:
            return self.p_bs

        def bm(self) -> int:
            return self.p_bm

        def solved(self) -> bool:
            return self.p_bs >= self.p_beta or self.p_received == self.p_todo.size()

        def free(self) -> bool:
            return self.p_workers == 0

        def searched(self) -> 'Gen.List':
            return self.p_done

        def searched_size(self) -> int:
            return self.p_done.size()

        def result(self, pv: 'Search.PV') -> int:
            pv.assign(self.p_pv)
            return self.p_bs

    # Search_Local Class
    class SearchLocal(Util.Waitable):
        def __init__(self):
            super().__init__()
            self.id = 0
            self.thread: Optional[threading.Thread] = None

            self.todo = False
            self.todo_sp: Optional[Search.SplitPoint] = None

            self.board = Board.BOARD()
            self.killer = Sort.Killer()
            self.pawn_table = Pawn.Table()
            self.eval_table = Eval.Table()

            self.node = 0
            self.max_ply = 0

            self.msp_stack: List[Search.SplitPoint] = [Search.SplitPoint() for _ in range(16)]
            self.msp_stack_size = 0

            self.ssp_stack: List[Search.SplitPoint] = []
            self.ssp_stack_size = 0

    # Helper Functions
    @staticmethod
    def new_search():
        Search.p_time.depth_limited = True
        Search.p_time.node_limited = False
        Search.p_time.time_limited = False

        Search.p_time.depth_limit = Search.MAX_DEPTH - 1

        Search.p_time.smart = False
        Search.p_time.ponder = False

    @staticmethod
    def set_depth_limit(depth: int):
        Search.p_time.depth_limited = True
        Search.p_time.depth_limit = depth

    @staticmethod
    def set_node_limit(node: int):
        Search.p_time.node_limited = True
        Search.p_time.node_limit = node

    @staticmethod
    def set_time_limit(time_limit: int):
        Search.p_time.time_limited = True
        Search.p_time.time_limit = time_limit

    @staticmethod
    def set_ponder():
        Search.p_time.ponder = True

    @staticmethod
    def clear():
        Search.p_time.flag = False
        Search.p_time.timer.reset()
        Search.p_time.timer.start()

        Search.current.depth = 0
        Search.current.max_ply = 0
        Search.current.node = 0
        Search.current.time = 0
        Search.current.speed = 0

        Search.current.move = Move.NONE
        Search.current.pos = 0
        Search.current.size = 0
        Search.current.fail_high = False

        Search.current.last_time = 0

        Search.best.depth = 0
        Search.best.move = Move.NONE
        Search.best.score = Score.NONE
        Search.best.flags = Score.FLAGS_NONE
        Search.best.pv.clear()

    @staticmethod
    def update_current():
        node = 0
        max_ply = 0

        for tid in range(Engine.engine.threads):
            sl = Search.p_sl[tid]
            node += sl.node
            if sl.max_ply > max_ply:
                max_ply = sl.max_ply

        Search.current.node = node
        Search.current.max_ply = max_ply

        Search.current.time = Search.p_time.timer.elapsed()
        Search.current.speed = 0 if Search.current.time < 10 else int(Search.current.node * 1000 / Search.current.time)

    @staticmethod
    def write_pv(best: 'Search.Best'):
        Search.sg.lock()
        try:
            print("info", end="")
            print(f" depth {best.depth}", end="")
            print(f" seldepth {Search.current.max_ply}", end="")
            print(f" nodes {Search.current.node}", end="")
            print(f" time {Search.current.time}", end="")

            if Score.is_mate(best.score):
                print(f" score mate {Score.signed_mate(best.score)}", end="")
            else:
                print(f" score cp {best.score}", end="")
            if best.flags == Score.FLAGS_LOWER:
                print(" lowerbound", end="")
            if best.flags == Score.FLAGS_UPPER:
                print(" upperbound", end="")

            print(f" pv {best.pv.to_can()}")
            sys.stdout.flush()
        finally:
            Search.sg.unlock()

    @staticmethod
    def write_info():
        Search.sg.lock()
        try:
            print("info", end="")
            print(f" depth {Search.current.depth}", end="")
            print(f" seldepth {Search.current.max_ply}", end="")
            print(f" currmove {Move.to_can(Search.current.move)}", end="")
            print(f" currmovenumber {Search.current.pos + 1}", end="")
            print(f" nodes {Search.current.node}", end="")
            print(f" time {Search.current.time}", end="")
            if Search.current.speed != 0:
                print(f" nps {Search.current.speed}", end="")
            print(f" hashfull {Search.sg.trans.used()}")
            sys.stdout.flush()
        finally:
            Search.sg.unlock()

    @staticmethod
    def write_info_opt():
        time_elapsed = Search.current.time

        if time_elapsed >= Search.current.last_time + 1000:
            Search.write_info()
            Search.current.last_time = time_elapsed - (time_elapsed % 1000)

    @staticmethod
    def depth_start(depth: int):
        Search.current.depth = depth

    @staticmethod
    def depth_end():
        pass

    @staticmethod
    def move_start(mv: int, pos: int, size: int):
        assert size > 0
        assert pos < size

        Search.current.move = mv
        Search.current.pos = pos
        Search.current.size = size

        Search.current.fail_high = False

    @staticmethod
    def move_fail_high():
        Search.current.fail_high = True
        Search.p_time.flag = False

    @staticmethod
    def move_end():
        Search.current.fail_high = False

    @staticmethod
    def update_best(best: 'Search.Best', sc: int, flags: int, pv: 'Search.PV'):
        assert sc != Score.NONE
        assert pv.size() != 0

        Search.p_time.drop = flags == Score.FLAGS_UPPER or (sc <= Search.p_time.last_score - 30 and Search.current.size > 1)

        if pv.move(0) != best.move or Search.p_time.drop:
            Search.p_time.flag = False

        best.depth = Search.current.depth
        best.move = pv.move(0)
        best.score = sc
        best.flags = flags
        best.pv.assign(pv)

    @staticmethod
    def search_end():
        Search.p_time.timer.stop()
        Search.update_current()
        Search.write_info()

    @staticmethod
    def idle_loop(sl: 'Search.SearchLocal', wait_sp: 'Search.SplitPoint'):
        Search.sl_push(sl, wait_sp)

        while True:
            sl.lock()
            try:
                assert sl.todo
                assert sl.todo_sp is None

                sl.todo = False
                sl.todo_sp = None

                while not wait_sp.free() and sl.todo_sp is None:
                    sl.wait()

                sp = sl.todo_sp
                sl.todo = True
                sl.todo_sp = None
            finally:
                sl.unlock()

            if sp is None:
                break

            Search.sl_push(sl, sp)

            try:
                Search.search_split_point(sl, sp)
            except Search.Abort:
                pass

            Search.sl_pop(sl)

            sp.leave()

        Search.sl_pop(sl)

    @staticmethod
    def helper_program(sl: 'Search.SearchLocal'):
        Search.sl_init_late(sl)
        Search.idle_loop(sl, Search.root_sp)

    @staticmethod
    def can_split(master: 'Search.SearchLocal', parent: 'Search.SplitPoint') -> bool:
        if Engine.engine.threads == 1:
            return False
        if master.msp_stack_size >= 16:
            return False
        if Search.sl_stop(master):
            return False

        for tid in range(Engine.engine.threads):
            worker = Search.p_sl[tid]
            if worker is not master and Search.sl_idle(worker, parent):
                return True

        return False

    @staticmethod
    def send_work(worker: 'Search.SearchLocal', sp: 'Search.SplitPoint'):
        worker.lock()
        try:
            if Search.sl_idle(worker, sp.parent()):
                sp.enter()
                worker.todo = True
                worker.todo_sp = sp
                worker.signal()
        finally:
            worker.unlock()

    @staticmethod
    def init_sg():
        Search.sg.history.clear()

    @staticmethod
    def search_root(sl: "Search.SearchLocal", ml: "Gen.List", depth: int, alpha: int, beta: int):
        assert 0 < depth < Search.MAX_DEPTH, "depth out of bounds"
        assert alpha < beta, "alpha must be less than beta"
        
        bd = sl.board
        assert Attack.is_legal(bd), "Board is not legal"
        
        pv_node = True

        bs = Score.NONE
        bm = Move.NONE
        old_alpha = alpha

        # Transposition table key
        key = 0
        if depth >= 0:
            key = bd.key()
        
        # Determine if the side to move is in check.
        in_check = Attack.is_in_check(bd)
        
        searched_size = 0

        # Loop over moves in the generated move list
        for pos in range(ml.size()):
            mv = ml.move(pos)
            
            # Determine if the move is "dangerous"
            dangerous = (in_check or 
                        Move.is_tactical(mv) or 
                        Move.is_check(mv, bd) or 
                        Move.is_castling(mv) or 
                        Move.is_pawn_push(mv, bd))
            
            ext = Search.extension(sl, mv, depth, pv_node)
            red = Search.reduction(sl, mv, depth, pv_node, in_check, searched_size, dangerous)
            if ext != 0:
                red = 0
            assert ext == 0 or red == 0, "Extension and reduction conflict"
            
            sc = 0

            # Create a new PV for this move’s search
            npv = Search.PV()
            
            # Start the move search.
            Search.move_start(mv, pos, ml.size())
            Search.move(sl, mv)
            
            # Search the move.
            if (pv_node and searched_size != 0) or red != 0:
                sc = -Search.search(sl, depth + ext - red - 1, -alpha - 1, -alpha, npv)
                if sc > alpha:
                    Search.move_fail_high()
                    sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)
            else:
                sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)
            
            # Undo the move and finish the move's processing.
            Search.undo(sl)
            Search.move_end()
            
            searched_size += 1
            
            if sc > bs:
                bs = sc
                
                # Create a PV for this best move.
                pv = Search.PV()
                pv.cat(mv, npv)
                
                Search.update_best(Search.best, sc, Score.flags(sc, alpha, beta), pv)
                Search.update_current()
                Search.write_pv(Search.best)
                
                if sc > alpha:
                    bm = mv
                    alpha = sc
                    # Optionally update the move ordering
                    ml.move_to_front(pos)
                    
                    if depth >= 0:
                        Search.sg.trans.store(key, depth, bd.ply(), mv, sc, Score.FLAGS_LOWER)
                    
                    if sc >= beta:
                        # Fail-hard cutoff.
                        return
                            
        # Ensure a valid score was found.
        assert bs != Score.NONE, "No score computed, should be a terminal node"
        assert bs < beta, "Best score should be less than beta"
        
        if depth >= 0:
            flags = Score.flags(bs, old_alpha, beta)
            Search.sg.trans.store(key, depth, bd.ply(), bm, bs, flags)

    @staticmethod
    def search(sl: 'Search.SearchLocal', depth: int, alpha: int, beta: int, pv: 'Search.PV') -> int:
        assert depth < Search.MAX_DEPTH
        assert alpha < beta

        bd = sl.board
        assert Attack.is_legal(bd)

        pv.clear()

        pv_node = depth > 0 and beta != alpha + 1

        # Mate-distance pruning
        sc = Score.from_trans(Score.MATE - 1, bd.ply())

        if sc < beta:
            beta = sc
            if sc <= alpha:
                return sc

        # Transposition table
        if bd.is_draw():
            return 0

        attacks = Attack.Attacks()
        Attack.init_attacks(attacks, bd.turn(), bd)
        in_check = attacks.size != 0

        use_trans = depth >= 0
        trans_depth = depth

        if depth < 0 and in_check:
            use_trans = True
            trans_depth = 0

        key = 0
        trans_move = Move.NONE

        if use_trans:
            key = bd.key()
            found, trans_move, trans_score, trans_flags = Search.sg.trans.retrieve(key, trans_depth, bd.ply())

            if found and not pv_node:
                if trans_flags == Score.FLAGS_LOWER and trans_score >= beta:
                    return trans_score
                if trans_flags == Score.FLAGS_UPPER and trans_score <= alpha:
                    return trans_score
                if trans_flags == Score.FLAGS_EXACT:
                    return trans_score

        # Ply limit
        if bd.ply() >= Search.MAX_PLY:
            return Search.eval(sl)

        # Beta pruning
        if not pv_node and depth > 0 and depth <= 3 and not Score.is_mate(beta) and not in_check:
            sc = Search.eval(sl) - depth * 50
            if sc >= beta:
                return sc

        # Null-move pruning
        if (not pv_node and depth > 0 and not Score.is_mate(beta) and not in_check and 
            not Material.lone_king(bd.turn(), bd) and Search.eval(sl) >= beta):
            
            bd.move_null()
            sc = Score.MIN

            if depth <= 3:
                sc = -Search.qs_static(sl, -beta + 1, 100)
            else:
                npv = Search.PV()
                sc = -Search.search(sl, depth - 4, -beta, -beta + 1, npv)

            bd.undo_null()

            if sc >= beta:
                if use_trans:
                    Search.sg.trans.store(key, trans_depth, bd.ply(), Move.NONE, sc, Score.FLAGS_LOWER)
                return sc

        # Stand pat
        bs = Score.NONE
        bm = Move.NONE
        old_alpha = alpha
        val = Score.NONE  # for delta pruning

        if depth <= 0 and not in_check:
            bs = Search.eval(sl)
            val = bs + 100  # QS-DP margin

            if bs > alpha:
                alpha = bs
                if bs >= beta:
                    return bs

        # Futility-pruning condition
        use_fp = False

        if depth > 0 and depth <= 8 and not Score.is_mate(alpha) and not in_check:
            sc = Search.eval(sl) + depth * 40
            val = sc + 50  # FP-DP margin, extra 50 for captures

            if sc <= alpha:
                bs = sc
                use_fp = True

        if depth <= 0 and not in_check:
            use_fp = True  # unify FP and QS

        # IID (Internal iterative deepening)
        if pv_node and depth >= 3 and trans_move == Move.NONE:
            npv = Search.PV()
            sc = Search.search(sl, depth - 2, alpha, beta, npv)
            if sc > alpha and npv.size() != 0:
                trans_move = npv.move(0)

        # Move loop
        ml_sorted = GenSort.List()
        ml_sorted.init(depth, bd, attacks, trans_move, sl.killer, Search.sg.history, use_fp)

        searched = Gen.List()

        while True:
            mv = ml_sorted.next_move()
            if mv == Move.NONE:
                break

            dangerous = (in_check or 
                         Move.is_tactical(mv) or 
                         Move.is_check(mv, bd) or 
                         Move.is_castling(mv) or 
                         Move.is_pawn_push(mv, bd) or 
                         ml_sorted.is_candidate())

            if (use_fp and Move.is_tactical(mv) and 
                not Move.is_check(mv, bd) and 
                val + Move.see_max(mv) <= alpha):
                continue  # delta pruning

            if (use_fp and not Move.is_safe(mv, bd)):
                continue  # SEE pruning

            if (not pv_node and depth > 0 and depth <= 3 and not Score.is_mate(bs) and 
                searched.size() >= depth * 4 and not dangerous):
                continue  # late-move pruning

            ext = Search.extension(sl, mv, depth, pv_node)
            red = Search.reduction(sl, mv, depth, pv_node, in_check, searched.size(), dangerous)

            if ext != 0:
                red = 0
            assert ext == 0 or red == 0

            sc = 0
            npv = Search.PV()

            Search.move(sl, mv)

            if (pv_node and searched.size() != 0) or red != 0:
                sc = -Search.search(sl, depth + ext - red - 1, -alpha - 1, -alpha, npv)
                if sc > alpha:
                    sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)
            else:
                sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)

            Search.undo(sl)

            searched.add(mv)

            if sc > bs:
                bs = sc
                pv.cat(mv, npv)

                if sc > alpha:
                    bm = mv
                    alpha = sc

                    if use_trans:
                        Search.sg.trans.store(key, trans_depth, bd.ply(), mv, sc, Score.FLAGS_LOWER)

                    if sc >= beta:
                        if depth > 0 and not in_check and not Move.is_tactical(mv):
                            sl.killer.add(mv, bd.ply())
                            Search.sg.history.add(mv, searched, bd)
                        return sc

            if depth >= 6 and not in_check and not use_fp and Search.can_split(sl, Search.sl_top(sl)):
                return Search.split(sl, depth, old_alpha, alpha, beta, pv, ml_sorted, searched, bs, bm)

        if bs == Score.NONE:
            assert depth > 0 or in_check
            return -Score.MATE + bd.ply() if in_check else 0

        assert bs < beta

        if use_trans:
            flags = Score.flags(bs, old_alpha, beta)
            Search.sg.trans.store(key, trans_depth, bd.ply(), bm, bs, flags)

        return bs

    @staticmethod
    def split(master: 'Search.SearchLocal', depth: int, old_alpha: int, alpha: int, beta: int, pv: 'Search.PV',
              todo: 'GenSort.List', done: 'Gen.List', bs: int, bm: int) -> int:
        Search.smp.lock()
        try:
            assert master.msp_stack_size < 16
            sp = master.msp_stack[master.msp_stack_size]
            master.msp_stack_size += 1

            parent = Search.sl_top(master)

            sp.init(master, parent, master.board, depth, old_alpha, alpha, beta, todo, done, bs, bm, pv)

            for tid in range(Engine.engine.threads):
                worker = Search.p_sl[tid]
                if worker is not master and Search.sl_idle(worker, parent):
                    Search.send_work(worker, sp)
        finally:
            Search.smp.unlock()

        try:
            Search.master_split_point(master, sp)
        except Search.Abort:
            pass

        assert master.msp_stack_size > 0
        assert master.msp_stack[master.msp_stack_size - 1] == sp
        master.msp_stack_size -= 1

        return sp.result(pv)

    @staticmethod
    def master_split_point(sl: 'Search.SearchLocal', sp: 'Search.SplitPoint'):
        sp.enter()

        Search.sl_push(sl, sp)

        try:
            Search.search_split_point(sl, sp)
        except Search.Abort:
            pass

        Search.sl_pop(sl)

        sp.leave()

        Search.idle_loop(sl, sp)
        sl.board.assign(sp.board())

        assert sp.free()

        # Update move-ordering tables
        bd = sl.board
        depth = sp.depth()
        ply = bd.ply()

        bs = sp.bs()
        bm = sp.bm()

        assert bs != Score.NONE

        if bs >= sp.beta() and depth > 0 and not Attack.is_in_check(bd) and not Move.is_tactical(bm):
            sl.killer.add(bm, ply)
            Search.sg.history.add(bm, sp.searched(), bd)

        if depth >= 0:
            flags = Score.flags(bs, sp.old_alpha(), sp.beta())
            Search.sg.trans.store(bd.key(), depth, ply, bm, bs, flags)

    @staticmethod
    def search_split_point(sl: 'Search.SearchLocal', sp: 'Search.SplitPoint'):
        bd = sl.board
        bd.assign(sp.board())

        depth = sp.depth()
        old_alpha = sp.old_alpha()
        beta = sp.beta()

        pv_node = depth > 0 and beta != old_alpha + 1

        in_check = Attack.is_in_check(bd)

        while True:
            sp.lock()
            try:
                mv = sp.next_move()
                alpha = sp.alpha()
                searched_size = sp.searched_size()
            finally:
                sp.unlock()

            if mv == Move.NONE:
                break

            assert alpha < beta

            dangerous = (in_check or 
                         Move.is_tactical(mv) or 
                         Move.is_check(mv, bd) or 
                         Move.is_castling(mv) or 
                         Move.is_pawn_push(mv, bd))

            ext = Search.extension(sl, mv, depth, pv_node)
            red = Search.reduction(sl, mv, depth, pv_node, in_check, searched_size, dangerous)

            if ext != 0:
                red = 0
            assert ext == 0 or red == 0

            sc = 0
            npv = Search.PV()

            Search.move(sl, mv)

            if (pv_node and searched_size != 0) or red != 0:
                sc = -Search.search(sl, depth + ext - red - 1, -alpha - 1, -alpha, npv)
                if sc > alpha:
                    sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)
            else:
                sc = -Search.search(sl, depth + ext - 1, -beta, -alpha, npv)

            Search.undo(sl)

            sp.update(mv, sc, npv)

    @staticmethod
    def qs_static(sl: 'Search.SearchLocal', beta: int, gain: int) -> int:
        bd = sl.board

        assert Attack.is_legal(bd)
        # assert not Attack.is_in_check(bd)  # Uncomment if necessary

        # Stand pat
        bs = Search.eval(sl)
        val = bs + gain

        if bs >= beta:
            return bs

        # Move loop
        attacks = Attack.Attacks()
        Attack.init_attacks(attacks, bd.turn(), bd)

        ml = GenSort.List()
        ml.init(-1, bd, attacks, Move.NONE, sl.killer, Search.sg.history, False)
        # Gen.gen_legals(ml, bd)

        done = 0  # Using bitmask for processed captures

        while True:
            mv = ml.next_move()
            if mv == Move.NONE:
                break

            if Bit.is_set(done, Move.to_sq(mv)):
                continue

            done = Bit.set_bit(done, Move.to_sq(mv))

            see = Move.see(mv, 0, Score.EVAL_MAX, bd)
            if see <= 0:
                continue

            sc = val + see

            if sc > bs:
                bs = sc
                if sc >= beta:
                    return sc

        assert bs < beta

        return bs

    @staticmethod
    def inc_node(sl: 'Search.SearchLocal'):
        sl.node += 1

        if sl.node % Search.NODE_PERIOD == 0:
            abort = False

            Search.update_current()

            if Search.poll():
                abort = True

            if Search.p_time.node_limited and Search.current.node >= Search.p_time.node_limit:
                abort = True

            if Search.p_time.time_limited and Search.current.time >= Search.p_time.time_limit:
                abort = True

            if Search.p_time.smart and Search.current.depth > 1 and Search.current.time >= Search.p_time.limit_0:
                if Search.current.pos == 0 or Search.current.time >= Search.p_time.limit_1:
                    if (not Search.p_time.drop and not Search.current.fail_high) or Search.current.time >= Search.p_time.limit_2:
                        if Search.p_time.ponder:
                            Search.p_time.flag = True
                        else:
                            abort = True
            
            assert Search.p_time.limit_0 >= 0
            if Search.p_time.smart and Search.current.depth > 1 and Search.current.size == 1 and Search.current.time >= Search.p_time.limit_0 // 8:
                if Search.p_time.ponder:
                    Search.p_time.flag = True
                else:
                    abort = True

            if abort:
                Search.sg_abort()

        if Search.sl_stop(sl):
            raise Search.Abort()

    @staticmethod
    def poll() -> bool:
        Search.write_info_opt()

        Search.sg.lock()
        try:
            if not Input.input_instance.has_input():
                return False

            line_ok, line = Input.input_instance.get_line()
            if Engine.engine.log:
                Util.log(line)

            if not line_ok:
                sys.exit(0)
            elif line == "isready":
                print("readyok")
                sys.stdout.flush()
                return False
            elif line == "stop":
                UCI.infinite = False
                return True
            elif line == "ponderhit":
                UCI.infinite = False
                Search.p_time.ponder = False
                return Search.p_time.flag
            elif line == "quit":
                sys.exit(0)

        finally:
            Search.sg.unlock()

        return False

    @staticmethod
    def move(sl: 'Search.SearchLocal', mv: int):
        bd = sl.board

        Search.inc_node(sl)
        bd.move(mv)

        ply = bd.ply()

        if ply > sl.max_ply:
            assert ply <= Search.MAX_PLY
            sl.max_ply = ply

    @staticmethod
    def undo(sl: 'Search.SearchLocal'):
        bd = sl.board
        bd.undo()

    @staticmethod
    def eval(sl: 'Search.SearchLocal') -> int:
        bd = sl.board
        return Eval.evaluate(bd, sl.eval_table, sl.pawn_table)

    @staticmethod
    def extension(sl: 'Search.SearchLocal', mv: int, depth: int, pv_node: bool) -> int:
        bd = sl.board

        if ((depth <= 4 and Move.is_check(mv, bd)) or
            (depth <= 4 and Move.is_recapture(mv, bd)) or
            (pv_node and Move.is_check(mv, bd)) or
            (pv_node and Move.is_tactical(mv) and Move.is_win(mv, bd)) or
            (pv_node and Move.is_pawn_push(mv, bd))):
            return 1
        else:
            return 0

    @staticmethod
    def reduction(sl: 'Search.SearchLocal', mv: int, depth: int, pv_node: bool, in_check: bool, searched_size: int, dangerous: bool) -> int:
        red = 0

        if depth >= 3 and searched_size >= 3 and not dangerous:
            red = depth // 3 if searched_size >= 6 else 1

        return red

    @staticmethod
    def gen_sort(sl: 'Search.SearchLocal', ml: 'Gen.List'):
        bd = sl.board

        Gen.gen_legals(ml, bd)

        v = Search.eval(sl)

        for pos in range(ml.size()):
            mv = ml.move(pos)
            Search.move(sl, mv)
            sc = -Search.qs_static(sl, Score.MAX, 0)
            Search.undo(sl)
            
            if (sc - v) < 0:
                sc = int((sc - v) / 4) + 1024
            else:
                sc = ((sc - v) // 4) + 1024  # HACK for unsigned 11-bit move-list scores
            assert 0 <= sc < Move.SCORE_SIZE

            ml.set_score(pos, sc)

        ml.sort()

    @staticmethod
    def sl_init_early(sl: 'Search.SearchLocal', id: int):
        sl.id = id
        sl.todo = True
        sl.todo_sp = None

        sl.node = 0
        sl.max_ply = 0

        sl.msp_stack_size = 0
        sl.ssp_stack_size = 0

    @staticmethod
    def sl_init_late(sl: 'Search.SearchLocal'):
        sl.killer.clear()
        sl.pawn_table.clear()
        sl.eval_table.clear()

    @staticmethod
    def sl_set_root(sl: 'Search.SearchLocal', bd: 'Board.BOARD'):
        sl.board.assign(bd)
        sl.board.set_root()

    @staticmethod
    def sl_signal(sl: 'Search.SearchLocal'):
        sl.lock()
        try:
            sl.signal()
        finally:
            sl.unlock()

    @staticmethod
    def sl_stop(sl: 'Search.SearchLocal') -> bool:
        sp = Search.sl_top(sl)
        while sp is not None:
            if sp.solved():
                return True
            sp = sp.parent()
        return False

    @staticmethod
    def sl_idle(worker: 'Search.SearchLocal', sp: 'Search.SplitPoint') -> bool:
        assert sp is not None

        if worker.todo:
            return False
        if worker.todo_sp is not None:
            return False

        wait_sp = Search.sl_top(worker)
        while sp is not None:
            if sp is wait_sp:
                return True
            sp = sp.parent()

        return False

    @staticmethod
    def sl_push(sl: 'Search.SearchLocal', sp: 'Search.SplitPoint'):
        assert sl.ssp_stack_size < 16, "ssp_stack_size exceeded"
        sl.ssp_stack.append(sp)
        sl.ssp_stack_size += 1

    @staticmethod
    def sl_pop(sl: 'Search.SearchLocal'):
        assert sl.ssp_stack_size > 0, "ssp_stack_size underflow"
        sl.ssp_stack.pop()
        sl.ssp_stack_size -= 1

    @staticmethod
    def sl_top(sl: 'Search.SearchLocal') -> 'Search.SplitPoint':
        assert sl.ssp_stack_size > 0, "ssp_stack_size underflow"
        return sl.ssp_stack[-1]

    @staticmethod
    def sg_abort():
        Search.root_sp.update_root()
        for tid in range(Engine.engine.threads):
            Search.sl_signal(Search.p_sl[tid])

    @staticmethod
    def search_asp(ml: 'Gen.List', depth: int):
        sl = Search.p_sl[0]
        assert depth <= 1 or Search.p_time.last_score == Search.best.score

        if depth >= 6 and not Score.is_mate(Search.p_time.last_score):
            margin = 10
            while margin < 500:
                a = Search.p_time.last_score - margin
                b = Search.p_time.last_score + margin
                assert Score.EVAL_MIN <= a < b <= Score.EVAL_MAX

                Search.search_root(sl, ml, depth, a, b)

                if a < Search.best.score < b:
                    return
                elif Score.is_mate(Search.best.score):
                    break

                margin *= 2

        Search.search_root(sl, ml, depth, Score.MIN, Score.MAX)

    @staticmethod
    def search_id(bd: 'Board.BOARD'):
        sl = Search.p_sl[0]
        Search.sl_set_root(sl, bd)

        Search.sl_push(sl, Search.root_sp)

        # Move generation
        ml = Gen.List()
        Search.gen_sort(sl, ml)
        assert ml.size() != 0

        Search.best.move = ml.move(0)
        Search.best.score = 0

        easy = (ml.size() == 1 or (ml.size() > 1 and ml.score(0) - ml.score(1) >= 12))  # 50 / 4 = 12
        easy_move = ml.move(0)

        Search.p_time.last_score = Score.NONE

        # Iterative deepening
        assert Search.p_time.depth_limited

        for depth in range(1, Search.p_time.depth_limit + 1):
            Search.depth_start(depth)
            Search.search_asp(ml, depth)
            Search.depth_end()

            Search.p_time.last_score = Search.best.score

            if Search.best.move != easy_move or Search.p_time.drop:
                easy = False

            if Search.p_time.smart and not Search.p_time.drop:
                abort = False

                Search.update_current()
                
                assert Search.p_time.limit_0 >= 0
                if ml.size() == 1 and Search.current.time >= Search.p_time.limit_0 // 16:
                    abort = True

                if easy and Search.current.time >= Search.p_time.limit_0 // 4:
                    abort = True

                if Search.current.time >= Search.p_time.limit_0 // 2:
                    abort = True

                if abort:
                    if Search.p_time.ponder:
                        Search.p_time.flag = True
                    else:
                        break

        Search.sl_pop(sl)

    @staticmethod
    def search_go(bd: 'Board.BOARD'):
        Search.clear()

        Search.init_sg()
        Search.sg.trans.inc_date()

        for tid in range(Engine.engine.threads):
            Search.sl_init_early(Search.p_sl[tid], tid)

        Search.root_sp.init_root(Search.p_sl[0])

        threads = []
        for tid in range(1, Engine.engine.threads):
            t = threading.Thread(target=Search.helper_program, args=(Search.p_sl[tid],))
            t.start()
            Search.p_sl[tid].thread = t
            threads.append(t)

        Search.sl_init_late(Search.p_sl[0])

        try:
            Search.search_id(bd)
        except Search.Abort:
            pass

        Search.sg_abort()

        for t in threads:
            t.join()

        Search.search_end()

    @staticmethod
    def search_dumb(bd: 'Board.BOARD'):
        Search.p_time.smart = False
        Search.p_time.last_score = Score.NONE
        Search.p_time.drop = False

        Search.search_go(bd)

    @staticmethod
    def search_smart(bd: 'Board.BOARD', moves: int, time_limit: int, inc: int):
        if moves == 0:
            moves = 40
        moves = min(moves, Material.interpolation(35, 15, bd))
        assert moves > 0

        total = time_limit + inc * (moves - 1)
        factor = 140 if Engine.engine.ponder else 120 
        alloc = total // moves * factor // 100
        reserve = total * (moves - 1) // 40
        max_time = min(time_limit, total - reserve)
        max_time = min(max_time - 60, max_time * 95 // 100)  # 60ms for lag

        alloc = max(alloc, 0)
        max_time = max(max_time, 0)

        Search.p_time.smart = True
        Search.p_time.limit_0 = min(alloc, max_time)
        Search.p_time.limit_1 = min(alloc * 4, max_time)
        Search.p_time.limit_2 = max_time
        Search.p_time.last_score = Score.NONE
        Search.p_time.drop = False

        assert 0 <= Search.p_time.limit_0 <= Search.p_time.limit_1 <= Search.p_time.limit_2

        Search.search_go(bd)

    @classmethod
    def init(cls):
        cls.p_time = cls.Time()
        cls.current = cls.Current()
        cls.best = cls.Best()
        cls.sg = cls.SearchGlobal()
        cls.smp = cls.SMP()
        cls.p_sl = [cls.SearchLocal() for _ in range(cls.MAX_THREADS)]
        cls.root_sp = cls.SplitPoint()

        cls.sg.trans.set_size(Engine.engine.hash_size)
        cls.sg.trans.alloc()
    
class UCI:
    bd: 'Board.BOARD' = Board.BOARD()
    infinite: bool = False
    delay: bool = False

    class Scanner:
        def __init__(self, ss: io.StringIO):
            self.p_ss = ss
            self.p_keywords: List[str] = []
            self.p_undo: bool = False
            self.p_word: str = ""
            self.add_keyword("")

        def is_keyword(self, word: str) -> bool:
            return word in self.p_keywords

        def add_keyword(self, keyword: str) -> None:
            self.p_keywords.append(keyword)

        def get_keyword(self) -> str:
            word = self.get_word()
            assert self.is_keyword(word), f"Word '{word}' is not a keyword."
            return word

        def get_args(self) -> str:
            args = []
            while True:
                word = self.get_word()
                if self.is_keyword(word):
                    self.unget_word()
                    break
                args.append(word)
            return ' '.join(args)

        def get_word(self) -> str:
            if self.p_undo:
                self.p_undo = False
                return self.p_word
            else:
                word = self.p_ss.read().split(maxsplit=1)
                if word:
                    self.p_word, remainder = word[0], word[1] if len(word) > 1 else ""
                    self.p_ss = io.StringIO(remainder)
                else:
                    self.p_word = ""
                return self.p_word

        def unget_word(self) -> None:
            assert not self.p_undo, "Cannot undo more than once."
            self.p_undo = True
        
    @staticmethod
    def fen(fen_str: str) -> None:
        UCI.bd.init_fen(fen_str)

    @staticmethod
    def move(move_str: str) -> None:
        mv = Move.from_string(move_str, UCI.bd)
        UCI.bd.move(mv)

    @staticmethod
    def send_bestmove() -> None:
        best_move_can = Move.to_can(Search.best.move)
        output = f"bestmove {best_move_can}"
        if Search.best.pv.size() > 1:
            ponder_move_can = Move.to_can(Search.best.pv.move(1))
            output += f" ponder {ponder_move_can}"
        print(output)
        sys.stdout.flush()
        UCI.delay = False

    @staticmethod
    def command(scan: 'UCI.Scanner') -> None:
        command = scan.get_word()
        # print(f"debug: command is {command}")

        if False:
            pass
        elif command == "uci":
            print("id name Senpai 1.0")
            print("id author Fabien Letouzey")
            print(f"option name Hash type spin default {Engine.engine.hash_size} min 16 max 16384")
            print(f"option name Ponder type check default {Engine.engine.ponder}")
            print(f"option name Threads type spin default {Engine.engine.threads} min 1 max 16")
            print(f"option name Log File type check default {Engine.engine.log}")
            print("uciok")
            sys.stdout.flush()

        elif command == "isready":
            print("readyok")
            sys.stdout.flush()

        elif command == "setoption":
            scan.add_keyword("name")
            scan.add_keyword("value")

            name = ""
            value = ""
            while True:
                part = scan.get_keyword()
                if part == "":
                    break
                if part == "name":
                    name = scan.get_args()
                elif part == "value":
                    value = scan.get_args()

            if Util.string_case_equal(name, "Hash"):
                Engine.engine.hash = int(Util.to_int(value))
                Search.sg.trans.set_size(Engine.engine.hash)
            elif Util.string_case_equal(name, "Ponder"):
                Engine.engine.ponder = Util.to_bool(value)
            elif Util.string_case_equal(name, "Threads") or Util.string_case_equal(name, "Cores"):
                Engine.engine.threads = int(Util.to_int(value))
            elif Util.string_case_equal(name, "Log File"):
                Engine.engine.log = Util.to_bool(value)

        elif command == "ucinewgame":
            Search.sg.trans.clear()

        elif command == "position":
            scan.add_keyword("fen")
            scan.add_keyword("startpos")
            scan.add_keyword("moves")

            while True:
                part = scan.get_keyword()
                if part == "":
                    break
                if part == "fen":
                    fen_str = scan.get_args()
                    UCI.fen(fen_str)
                elif part == "startpos":
                    UCI.fen(Board.start_fen)
                elif part == "moves":
                    while True:
                        arg = scan.get_word()
                        if arg == "":
                            break
                        UCI.move(arg)

        elif command == "go":
            scan.add_keyword("searchmoves")
            scan.add_keyword("ponder")
            scan.add_keyword("wtime")
            scan.add_keyword("btime")
            scan.add_keyword("winc")
            scan.add_keyword("binc")
            scan.add_keyword("movestogo")
            scan.add_keyword("depth")
            scan.add_keyword("nodes")
            scan.add_keyword("mate")
            scan.add_keyword("movetime")
            scan.add_keyword("infinite")

            Search.new_search()

            infinite = False
            UCI.delay = False

            smart = False
            time = 60000
            inc = 0
            movestogo = 0

            while True:
                part = scan.get_keyword()
                if part == "":
                    break
                args = scan.get_args()

                if part == "ponder":
                    infinite = True
                    Search.set_ponder()
                elif part == "wtime":
                    if UCI.bd.turn() == Side.WHITE:
                        smart = True
                        time = int(Util.to_int(args))
                elif part == "btime":
                    if UCI.bd.turn() == Side.BLACK:
                        smart = True
                        time = int(Util.to_int(args))
                elif part == "winc":
                    if UCI.bd.turn() == Side.WHITE:
                        smart = True
                        inc = int(Util.to_int(args))
                elif part == "binc":
                    if UCI.bd.turn() == Side.BLACK:
                        smart = True
                        inc = int(Util.to_int(args))
                elif part == "movestogo":
                    smart = True
                    movestogo = int(Util.to_int(args))
                elif part == "depth":
                    Search.set_depth_limit(int(Util.to_int(args)))
                elif part == "nodes":
                    Search.set_node_limit(Util.to_int(args))
                elif part == "movetime":
                    Search.set_time_limit(int(Util.to_int(args)))
                elif part == "infinite":
                    infinite = True

            if smart:
                Search.search_smart(UCI.bd, movestogo, time, inc)
            else:
                Search.search_dumb(UCI.bd)

            if infinite:  # Implementing the UCI-design mistake
                UCI.delay = True
            else:
                UCI.send_bestmove()

        elif command == "stop":
            if UCI.delay:
                UCI.send_bestmove()

        elif command == "ponderhit":
            if UCI.delay:
                UCI.send_bestmove()

        elif command == "quit":
            sys.exit(0)

    @staticmethod
    def line(line_str: str) -> None:
        if Engine.engine.log:
            Util.log(line_str)
        args = io.StringIO(line_str)
        scan = UCI.Scanner(args)
        UCI.command(scan)

    @staticmethod
    def loop() -> None:
        # print(True)  # Equivalent to std::cout << std::boolalpha in C++
        UCI.delay = False
        UCI.bd.init_fen(Board.start_fen)

        while True:
            try:
                line_ok, line = Input.input_instance.get_line()
                if line is None:
                    break
                UCI.line(line)
            except EOFError:
                break


def main():
    # Initialize all components
    Util.init()
    Input.init()
    Bit.init()
    Hash.init()
    Castling.init()
    Attack.init()
    Engine.init()
    Material.init()
    PST.init()
    Pawn.init()
    Eval.init()
    Search.init()

    # Start the UCI loop
    UCI.loop()


if __name__ == "__main__":
    main()
