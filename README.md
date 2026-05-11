# Assignment3
"""
game_state.py
=============
Manages all game logic — scoring, mistake tracking, round flow —
completely separated from the GUI layer.

Classes:
    RoundResult  - immutable record of one completed round
    GameState    - central state machine for the game session
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from image_processor import DifferenceRegion


# ===========================================================================
# RoundResult — immutable snapshot of a completed round
# ===========================================================================

@dataclass(frozen=True)
class RoundResult:
    """
    Stores the outcome of one completed round.
    frozen=True makes instances immutable (like a record).
    """
    found: int        # how many differences were found
    total: int        # total differences in the round (always 5)
    mistakes: int     # mistakes made this round
    revealed: bool    # True if the player pressed Reveal

    @property
    def score(self) -> int:
        """Points earned: +10 per found difference, -3 per mistake, min 0."""
        return max(0, self.found * 10 - self.mistakes * 3)


# ===========================================================================
# GameState — the central game state machine
# ===========================================================================

class GameState:
    """
    Manages the complete state of a game session across multiple rounds.

    Responsibilities:
        - Track how many differences have been found this round
        - Track mistakes and enforce the 3-mistake limit
        - Calculate and accumulate scores
        - Fire callback hooks so the GUI can react to events
        - Keep a history of completed rounds

    Demonstrates:
        - Encapsulation  : all mutable state is private; exposed via properties
        - Class interaction : processes DifferenceRegion objects from ImageProcessor
    """

    MAX_MISTAKES:    int = 3   # player is locked out after this many mistakes
    NUM_DIFFERENCES: int = 5   # total differences per round
    CLICK_TOLERANCE: int = 55  # pixel grace around a difference centre

    def __init__(self) -> None:
        # Session-level (persist across rounds)
        self._total_score: int = 0
        self._history: List[RoundResult] = []

        # Round-level (reset each new round)
        self._differences: List[DifferenceRegion] = []
        self._mistakes: int = 0
        self._revealed: bool = False
        self._round_active: bool = False

        # --- Callback hooks (set by GameApp in gui.py) ---
        # Called with the DifferenceRegion that was just found
        self.on_found:          Optional[Callable[[DifferenceRegion], None]] = None
        # Called on every wrong click
        self.on_mistake:        Optional[Callable[[], None]] = None
        # Called when the round ends for any reason
        self.on_round_complete: Optional[Callable[[], None]] = None
        # Called specifically when the player reaches MAX_MISTAKES
        self.on_locked_out:     Optional[Callable[[], None]] = None

    # -----------------------------------------------------------------------
    # Read-only properties (encapsulation — GUI reads but cannot write)
    # -----------------------------------------------------------------------

    @property
    def total_score(self) -> int:
        """Cumulative score across all rounds."""
        return self._total_score

    @property
    def current_mistakes(self) -> int:
        """Number of mistakes made in the current round."""
        return self._mistakes

    @property
    def found_count(self) -> int:
        """Number of differences found so far this round."""
        return sum(1 for d in self._differences if d.found)

    @property
    def remaining(self) -> int:
        """Number of differences not yet found this round."""
        return sum(1 for d in self._differences if not d.found)

    @property
    def round_over(self) -> bool:
        """True once the round has ended (win, lockout, or reveal)."""
        return not self._round_active

    @property
    def locked_out(self) -> bool:
        """True if the player has reached the mistake limit."""
        return self._mistakes >= self.MAX_MISTAKES

    @property
    def differences(self) -> List[DifferenceRegion]:
        """All difference regions for the current round."""
        return self._differences

    @property
    def found_regions(self) -> List[DifferenceRegion]:
        """Differences that have been found."""
        return [d for d in self._differences if d.found]

    @property
    def unfound_regions(self) -> List[DifferenceRegion]:
        """Differences that have NOT been found yet."""
        return [d for d in self._differences if not d.found]

    @property
    def history(self) -> List[RoundResult]:
        """Copy of the completed-round history list."""
        return list(self._history)

    @property
    def rounds_played(self) -> int:
        """Total rounds completed so far."""
        return len(self._history)

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def start_round(self, differences: List[DifferenceRegion]) -> None:
        """
        Begin a new round using the given list of DifferenceRegion objects
        (provided by ImageProcessor after loading an image).
        Resets all per-round counters.
        """
        self._differences = differences
        # Reset the found flag on every region
        for d in self._differences:
            d.found = False
        self._mistakes = 0
        self._revealed = False
        self._round_active = True

    def process_click(self, px: int, py: int) -> Tuple[bool, Optional[DifferenceRegion]]:
        """
        Evaluate a click at (px, py) on the modified image.

        Returns a tuple (hit, region):
            hit    : True  if the click matched an unfound difference
            region : the matched DifferenceRegion, or None on a miss

        Side effects:
            - Marks the region as found on a hit
            - Increments mistake counter on a miss
            - Fires the appropriate callback
            - Ends the round if all found or mistakes maxed out
        """
        # Ignore clicks if round is already over or player is locked out
        if not self._round_active or self.locked_out:
            return False, None

        # Check every unfound difference
        for diff in self._differences:
            if not diff.found and diff.contains_point(px, py, self.CLICK_TOLERANCE):
                # HIT — mark as found
                diff.found = True
                if self.on_found:
                    self.on_found(diff)
                # Check if all 5 are now found
                if self.remaining == 0:
                    self._end_round(revealed=False)
                return True, diff

        # MISS
        self._mistakes += 1
        if self.on_mistake:
            self.on_mistake()

        # Check if the player just hit the mistake limit
        if self.locked_out:
            self._end_round(revealed=False)
            if self.on_locked_out:
                self.on_locked_out()

        return False, None

    def reveal_all(self) -> List[DifferenceRegion]:
        """
        Reveal all currently unfound differences and end the round.
        Returns the list of regions that were revealed.
        """
        if not self._round_active:
            return []
        unfound = self.unfound_regions
        self._revealed = True
        self._end_round(revealed=True)
        return unfound

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _end_round(self, revealed: bool) -> None:
        """
        Record the round result, add points to the session total,
        mark the round as inactive, and fire on_round_complete.
        """
        result = RoundResult(
            found=self.found_count,
            total=self.NUM_DIFFERENCES,
            mistakes=self._mistakes,
            revealed=revealed,
        )
        self._history.append(result)
        self._total_score += result.score
        self._round_active = False

        if self.on_round_complete:
            self.on_round_complete()
