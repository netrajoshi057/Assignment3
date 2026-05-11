# Assignment3
#Identifying differences in images
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

# GameState — the central game state machine

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


            """
gui.py
======
Tkinter GUI for the Spot-the-Difference game.

Classes:
    ImagePanel  - reusable Canvas widget that shows an OpenCV image
    StatusBar   - HUD widget showing remaining / mistakes / score
    GameApp     - main application window; wires everything together

Demonstrates:
    - Inheritance    : ImagePanel and StatusBar extend tk.Frame
    - Encapsulation  : each class manages its own widgets and state
    - Polymorphism   : update_image() accepts any BGR ndarray
    - Class interaction: GameApp orchestrates ImageProcessor + GameState
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Callable, List
import cv2
import numpy as np
from PIL import Image, ImageTk

from image_processor import ImageProcessor, DifferenceRegion
from game_state import GameState

# ---------------------------------------------------------------------------
# Colour palette — dark theme with gold accent
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":       "#0f1117",    # main background
    "panel":    "#1a1d27",    # image panel background
    "accent":   "#e8c547",    # gold — used for headings and score
    "accent2":  "#4ecdc4",    # teal — used for remaining count
    "danger":   "#e84545",    # red — used for mistakes
    "success":  "#4caf50",    # green — used for success messages
    "text":     "#f0f0f0",    # primary text
    "subtext":  "#8a8fa8",    # secondary/hint text
    "border":   "#2e3044",    # widget borders
}


# ===========================================================================
# ImagePanel — reusable image display widget (inherits tk.Frame)
# ===========================================================================

class ImagePanel(tk.Frame):
    """
    A Tkinter Frame that contains a Canvas for displaying an OpenCV
    (BGR numpy array) image.

    If clickable=True, left-click events are forwarded to click_callback.

    Inherits from tk.Frame to demonstrate inheritance.
    """

    PLACEHOLDER = "Load an image\nto begin"

    def __init__(
        self,
        parent,
        width: int,
        height: int,
        label: str = "",
        clickable: bool = False,
        click_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        # Call parent constructor (tk.Frame)
        super().__init__(parent, bg=PALETTE["panel"], **kwargs)

        self._width = width
        self._height = height
        self._clickable = clickable
        self._click_callback = click_callback
        self._photo: Optional[ImageTk.PhotoImage] = None  # must keep reference!

        # Optional title label above the canvas
        if label:
            tk.Label(
                self, text=label,
                font=("Courier New", 11, "bold"),
                fg=PALETTE["accent"], bg=PALETTE["panel"], pady=6,
            ).pack(side=tk.TOP)

        # Main canvas for image rendering
        self._canvas = tk.Canvas(
            self,
            width=width, height=height,
            bg=PALETTE["bg"],
            highlightthickness=2,
            highlightbackground=PALETTE["border"],
            cursor="crosshair" if clickable else "arrow",
        )
        self._canvas.pack(padx=8, pady=(0, 8))

        # Always bind — guard with self._clickable flag at runtime
        self._canvas.bind("<Button-1>", self._on_click)

        self._show_placeholder()

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def update_image(self, bgr_image: np.ndarray) -> None:
        """
        Convert an OpenCV BGR ndarray to a Tkinter-compatible photo image
        and render it on the canvas.
        """
        # OpenCV uses BGR; Tkinter/PIL uses RGB — must convert
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Resize to exactly match the fixed canvas size
        pil_img = Image.fromarray(rgb).resize(
            (self._width, self._height), Image.LANCZOS
        )
        # Keep a reference — without this, garbage collector deletes the image
        self._photo = ImageTk.PhotoImage(pil_img)
        # Do NOT resize the canvas — just redraw on the fixed-size canvas
        self._canvas.delete("all")
        self._canvas.create_image(
            0, 0,
            anchor=tk.NW,
            image=self._photo
        )
        # Force Tkinter to update the display immediately
        self._canvas.update_idletasks()

    def set_clickable(self, state: bool) -> None:
        """Enable or disable player click interaction on this panel."""
        self._clickable = state
        self._canvas.config(cursor="crosshair" if state else "arrow")

    def clear(self) -> None:
        """Remove the image and show the placeholder text."""
        self._photo = None
        self._canvas.delete("all")
        self._show_placeholder()

    def flash_border(self, colour: str, duration_ms: int = 300) -> None:
        """Briefly change the canvas border colour then restore it."""
        self._canvas.config(highlightbackground=colour)
        self._canvas.after(
            duration_ms,
            lambda: self._canvas.config(highlightbackground=PALETTE["border"])
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Draw a placeholder message on the blank canvas."""
        self._canvas.create_rectangle(
            0, 0, self._width, self._height,
            fill=PALETTE["bg"], outline="",
        )
        self._canvas.create_text(
            self._width // 2, self._height // 2,
            text=self.PLACEHOLDER,
            fill=PALETTE["subtext"],
            font=("Courier New", 13),
            justify=tk.CENTER,
        )

    def _on_click(self, event: tk.Event) -> None:
        """Forward canvas click coordinates to the registered callback."""
        if self._clickable and self._click_callback:
            self._click_callback(event.x, event.y)


# ===========================================================================
# StatusBar — HUD display widget (inherits tk.Frame)
# ===========================================================================

class StatusBar(tk.Frame):
    """
    A Tkinter Frame that displays three live counters:
      - Remaining differences to find
      - Current mistakes vs maximum allowed
      - Cumulative score

    Inherits from tk.Frame to demonstrate inheritance.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=PALETTE["panel"], **kwargs)

        label_cfg = dict(
            font=("Courier New", 12, "bold"),
            bg=PALETTE["panel"],
            padx=16, pady=8,
        )

        # StringVars allow the labels to update without recreating widgets
        self._remaining_var = tk.StringVar(value="Remaining: –")
        self._mistakes_var  = tk.StringVar(value="Mistakes:  –")
        self._score_var     = tk.StringVar(value="Score: 0")

        tk.Label(self, textvariable=self._remaining_var,
                 fg=PALETTE["accent2"], **label_cfg).pack(side=tk.LEFT)

        self._mistake_label = tk.Label(
            self, textvariable=self._mistakes_var,
            fg=PALETTE["danger"], **label_cfg
        )
        self._mistake_label.pack(side=tk.LEFT)

        tk.Label(self, textvariable=self._score_var,
                 fg=PALETTE["accent"], **label_cfg).pack(side=tk.LEFT)

        # Hint text shown on the right side
        self._hint_label = tk.Label(
            self, text="",
            font=("Courier New", 11),
            fg=PALETTE["text"], bg=PALETTE["panel"], padx=12,
        )
        self._hint_label.pack(side=tk.RIGHT)

    def update(self, remaining: int, mistakes: int,
               max_mistakes: int, score: int, hint: str = "") -> None:
        """Refresh all status bar values."""
        self._remaining_var.set(f"Remaining: {remaining}")
        self._mistakes_var.set(f"Mistakes:  {mistakes}/{max_mistakes}")
        # Turn mistake label orange when getting close, red at max
        if mistakes >= max_mistakes:
            self._mistake_label.config(fg=PALETTE["danger"])
        elif mistakes >= max_mistakes - 1:
            self._mistake_label.config(fg="#e8a020")  # orange warning
        else:
            self._mistake_label.config(fg=PALETTE["accent2"])
        self._score_var.set(f"Score: {score}")
        self._hint_label.config(text=hint)

    def reset(self) -> None:
        """Reset all labels to their initial placeholder state."""
        self._remaining_var.set("Remaining: –")
        self._mistakes_var.set("Mistakes:  –")
        self._score_var.set("Score: 0")
        self._hint_label.config(text="")


# ===========================================================================
# GameApp — root application window
# ===========================================================================

class GameApp(tk.Tk):
    """
    The main Tkinter window.

    Creates and owns:
        - ImageProcessor  (handles OpenCV image work)
        - GameState       (handles all game logic)
        - ImagePanel x2   (original and modified image displays)
        - StatusBar       (HUD)
        - Buttons         (Load Image, Reveal All)

    Wires GameState callbacks to GUI update methods so the GUI reacts
    automatically to game events (hit, miss, round over, lockout).

    Inherits from tk.Tk to demonstrate inheritance.
    """

    IMG_W: int = 550   # display width for each image
    IMG_H: int = 430   # display height for each image

    def __init__(self):
        super().__init__()   # initialise the Tkinter root window

        # --- Instantiate core logic objects ---
        self._processor = ImageProcessor()
        self._state = GameState()

        # --- Wire GameState event callbacks to GUI methods ---
        self._state.on_found          = self._on_difference_found
        self._state.on_mistake        = self._on_mistake
        self._state.on_round_complete = self._on_round_complete
        self._state.on_locked_out     = self._on_locked_out

        # --- Configure the root window ---
        self.title("Spot the Difference  —  HIT137 Assignment 3")
        self.resizable(False, False)
        self.configure(bg=PALETTE["bg"])

        # --- Build all widgets ---
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create and lay out all widgets in the window."""

        # ── Header bar ──────────────────────────────────────────────────────
        header = tk.Frame(self, bg=PALETTE["bg"])
        header.pack(fill=tk.X, padx=20, pady=(16, 4))

        tk.Label(
            header,
            text="◈  SPOT  THE  DIFFERENCE  ◈",
            font=("Courier New", 18, "bold"),
            fg=PALETTE["accent"], bg=PALETTE["bg"],
        ).pack(side=tk.LEFT)

        self._rounds_var = tk.StringVar(value="Rounds played: 0")
        tk.Label(
            header, textvariable=self._rounds_var,
            font=("Courier New", 11),
            fg=PALETTE["subtext"], bg=PALETTE["bg"],
        ).pack(side=tk.RIGHT)

        # ── Button row ──────────────────────────────────────────────────────
        btn_bar = tk.Frame(self, bg=PALETTE["bg"])
        btn_bar.pack(fill=tk.X, padx=20, pady=8)

        btn_style = dict(
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT, bd=0,
            padx=18, pady=7,
            cursor="hand2",
        )

        # Load Image button — always enabled
        self._load_btn = tk.Button(
            btn_bar, text="⬆  Load Image",
            bg=PALETTE["accent"], fg=PALETTE["bg"],
            activebackground="#cfaa30",
            activeforeground=PALETTE["bg"],
            command=self._load_image,
            **btn_style,
        )
        self._load_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Reveal All button — disabled until an image is loaded
        self._reveal_btn = tk.Button(
            btn_bar, text="◎  Reveal All",
            bg=PALETTE["border"], fg=PALETTE["accent2"],
            activebackground="#2e3044",
            activeforeground=PALETTE["accent2"],
            command=self._reveal_all,
            state=tk.DISABLED,
            **btn_style,
        )
        self._reveal_btn.pack(side=tk.LEFT)

        # ── Image panels (side by side) ──────────────────────────────────
        panels_frame = tk.Frame(self, bg=PALETTE["bg"])
        panels_frame.pack(padx=20, pady=4)

        # Left panel — original image, NOT clickable
        self._orig_panel = ImagePanel(
            panels_frame,
            width=self.IMG_W, height=self.IMG_H,
            label="◁  ORIGINAL  (reference only)  ▷",
            clickable=False,
        )
        self._orig_panel.pack(side=tk.LEFT, padx=(0, 8))

        # Right panel — modified image, CLICKABLE
        self._mod_panel = ImagePanel(
            panels_frame,
            width=self.IMG_W, height=self.IMG_H,
            label="◁  FIND THE DIFFERENCES — click here  ▷",
            clickable=False,                        # enabled after image loads
            click_callback=self._on_image_click,
        )
        self._mod_panel.pack(side=tk.LEFT)

        # ── Status bar ──────────────────────────────────────────────────────
        self._status_bar = StatusBar(self)
        self._status_bar.pack(fill=tk.X, padx=20, pady=(4, 2))

        tk.Frame(self, height=1, bg=PALETTE["border"]).pack(fill=tk.X, padx=20)

        # ── Round history row ────────────────────────────────────────────
        history_frame = tk.Frame(self, bg=PALETTE["bg"])
        history_frame.pack(fill=tk.X, padx=20, pady=(6, 14))

        self._history_var = tk.StringVar(value="No rounds completed yet.")
        tk.Label(
            history_frame, textvariable=self._history_var,
            font=("Courier New", 10),
            fg=PALETTE["subtext"], bg=PALETTE["bg"],
            justify=tk.LEFT, anchor="w",
        ).pack(fill=tk.X)

    # -----------------------------------------------------------------------
    # Button command handlers
    # -----------------------------------------------------------------------

    def _load_image(self) -> None:
        """Open a file dialog, load the chosen image, and start a new round."""
        path = filedialog.askopenfilename(
            title="Choose an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return   # user cancelled the dialog

        success = self._processor.load_image(
            path, display_size=(self.IMG_W, self.IMG_H)
        )
        if not success:
            messagebox.showerror(
                "Cannot Load Image",
                "The selected file could not be opened.\n"
                "Please choose a valid JPG, PNG, or BMP file."
            )
            return

        # Start a new game round with the freshly generated differences
        self._state.start_round(self._processor.differences)

        # Enable player interaction
        self._mod_panel.set_clickable(True)
        self._reveal_btn.config(state=tk.NORMAL)

        # Render both panels and force the window to redraw immediately
        self._refresh_images()
        self._refresh_status()
        self._rounds_var.set(f"Rounds played: {self._state.rounds_played}")
        self.update()   # force Tkinter to repaint everything now

    def _reveal_all(self) -> None:
        """Reveal all unfound differences in blue and end the round."""
        if self._state.round_over:
            return

        self._state.reveal_all()   # triggers on_round_complete callback

        # Show blue circles for the revealed differences
        self._refresh_images_with_reveal()
        self._refresh_status(hint="Differences revealed — load a new image to play again.")
        self._mod_panel.set_clickable(False)
        self._reveal_btn.config(state=tk.DISABLED)
        self._update_history()

    # -----------------------------------------------------------------------
    # GameState callback handlers (fired by GameState, executed on GUI thread)
    # -----------------------------------------------------------------------

    def _on_difference_found(self, region: DifferenceRegion) -> None:
        """Called by GameState when the player correctly identifies a difference."""
        self._refresh_images()
        self._refresh_status(hint=f"✔  Found!  {self._state.remaining} remaining.")

    def _on_mistake(self) -> None:
        """Called by GameState on every wrong click."""
        self._refresh_status(hint="✘  Incorrect — try again!")
        # Flash the modified panel border red as visual feedback
        self._mod_panel.flash_border(PALETTE["danger"])

    def _on_locked_out(self) -> None:
        """Called by GameState when the player reaches 3 mistakes."""
        self._mod_panel.set_clickable(False)
        self._reveal_btn.config(state=tk.DISABLED)
        self._refresh_images()   # show found circles only (no blue reveal)
        self._refresh_status(
            hint="❌  Too many mistakes — load a new image to restart."
        )
        self._update_history()
        messagebox.showwarning(
            "Too Many Mistakes!",
            f"You made {GameState.MAX_MISTAKES} mistakes.\n\n"
            f"Differences found:  {self._state.found_count} / "
            f"{GameState.NUM_DIFFERENCES}\n\n"
            "Load a new image to try again.",
        )

    def _on_round_complete(self) -> None:
        """
        Called by GameState when the round ends for any reason.
        If the player found all 5 (not locked out, not revealed), show
        the success dialog.
        """
        if not self._state.locked_out and not self._state.history[-1].revealed:
            # Player won! All 5 found without using Reveal
            self._mod_panel.set_clickable(False)
            self._reveal_btn.config(state=tk.DISABLED)
            self._refresh_images()
            self._refresh_status(hint="🎉  All differences found!")
            self._update_history()
            last = self._state.history[-1]
            messagebox.showinfo(
                "Congratulations! 🎉",
                f"You found all {GameState.NUM_DIFFERENCES} differences!\n\n"
                f"Mistakes this round:   {last.mistakes}\n"
                f"Points earned:         {last.score}\n"
                f"Total score:           {self._state.total_score}\n\n"
                "Load another image to keep playing.",
            )

    # -----------------------------------------------------------------------
    # Click forwarding
    # -----------------------------------------------------------------------

    def _on_image_click(self, x: int, y: int) -> None:
        """
        Receives a click coordinate from ImagePanel and passes it to
        GameState for evaluation.  GameState fires the appropriate callback.
        """
        if not self._state.round_over:
            self._state.process_click(x, y)

    # -----------------------------------------------------------------------
    # Image rendering helpers
    # -----------------------------------------------------------------------

    def _refresh_images(self) -> None:
        """
        Re-render both panels with current state:
          - Red circles on found differences
          - No reveal circles (used during active play)
        """
        found = self._state.found_regions
        orig_img = self._processor.get_original_with_overlays(found, [])
        mod_img  = self._processor.get_modified_with_overlays(found, [])
        self._orig_panel.update_image(orig_img)
        self._mod_panel.update_image(mod_img)

    def _refresh_images_with_reveal(self) -> None:
        """
        Re-render both panels showing:
          - Red circles on found differences
          - Blue circles on all remaining (revealed) differences
        """
        found    = self._state.found_regions
        revealed = self._state.unfound_regions
        orig_img = self._processor.get_original_with_overlays(found, revealed)
        mod_img  = self._processor.get_modified_with_overlays(found, revealed)
        self._orig_panel.update_image(orig_img)
        self._mod_panel.update_image(mod_img)

    def _refresh_status(self, hint: str = "") -> None:
        """Push current counters to the StatusBar widget."""
        self._status_bar.update(
            remaining    = self._state.remaining,
            mistakes     = self._state.current_mistakes,
            max_mistakes = GameState.MAX_MISTAKES,
            score        = self._state.total_score,
            hint         = hint,
        )

    def _update_history(self) -> None:
        """Update the round-history text at the bottom of the window."""
        self._rounds_var.set(f"Rounds played: {self._state.rounds_played}")
        parts = []
        for i, r in enumerate(self._state.history, 1):
            if r.revealed:
                outcome = "revealed"
            else:
                outcome = f"{r.found}/{r.total} found"
            parts.append(
                f"Round {i}: {outcome}  |  "
                f"mistakes {r.mistakes}  |  "
                f"+{r.score} pts"
            )
        # Show last 3 rounds to keep the bar compact
        self._history_var.set(
            "   ·   ".join(parts[-3:]) if parts else "No rounds completed yet."
        )
"""
image_processor.py
Handles all OpenCV image loading, cloning, and programmatic difference
generation for the Spot-the-Difference game.

Classes:
    DifferenceRegion  - stores one difference's location and found-status
    ImageAlteration   - abstract base class for all alteration types
    ColourShiftAlt    - shifts hue/saturation (concrete alteration)
    BrightnessAlt     - changes local brightness (concrete alteration)
    PixelSwapAlt      - mirrors a patch horizontally (concrete alteration)
    TextureNoiseAlt   - adds structured noise (concrete alteration)
    ContrastAlt       - scales local contrast (concrete alteration)
    ImageProcessor    - loads image, creates clone, applies 5 differences
"""

import cv2
import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


# DifferenceRegion — data class for one hidden difference


@dataclass
class DifferenceRegion:
    """
    Stores the bounding box (x, y, w, h) of one difference region,
    the alteration type applied, and whether the player has found it.
    """
    x: int               # top-left x coordinate
    y: int               # top-left y coordinate
    w: int               # width of the region
    h: int               # height of the region
    alteration_name: str = ""
    found: bool = False  # True once the player clicks it

    @property
    def centre(self) -> Tuple[int, int]:
        """Return the centre point of the region."""
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def radius(self) -> int:
        """Circle radius large enough to enclose the region."""
        return max(self.w, self.h) // 2 + 10

    def contains_point(self, px: int, py: int, tolerance: int = 30) -> bool:
        """
        Return True if the point (px, py) falls within this region
        plus a tolerance buffer so clicks don't have to be pixel-perfect.
        """
        cx, cy = self.centre
        return (abs(px - cx) <= self.w // 2 + tolerance and
                abs(py - cy) <= self.h // 2 + tolerance)


# ImageAlteration — abstract base class (demonstrates inheritance)


class ImageAlteration(ABC):
    """
    Abstract base class defining the interface every alteration must follow.
    Subclasses must implement:
        name  (property) : human-readable label
        apply (method)   : modify the image ROI and return the image
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this alteration type."""

    @abstractmethod
    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        """
        Apply the alteration to *image* inside the rectangle (x, y, w, h).
        Modifies in-place and returns the modified image.
        """



# Concrete alteration types — all inherit from ImageAlteration (polymorphism)


class ColourShiftAlt(ImageAlteration):
    """
    Shifts the hue and boosts saturation of a rectangular region
    using OpenCV's HSV colour space.
    The change is noticeable on careful inspection but not glaringly obvious.
    """

    def __init__(self, hue_delta: int = 20, sat_scale: float = 1.4):
        self._hue_delta = hue_delta    # how much to rotate the hue (0-180)
        self._sat_scale = sat_scale    # saturation multiplier

    @property
    def name(self) -> str:
        return "ColourShift"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].copy()
        # Convert to HSV, shift hue, scale saturation, convert back to BGR
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._sat_scale, 0, 255)
        hsv = hsv.astype(np.uint8)
        image[y:y + h, x:x + w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image


class BrightnessAlt(ImageAlteration):
    """
    Increases or decreases the brightness of a rectangular region
    by adding a fixed offset to all pixel values.
    """

    def __init__(self, delta: int = 55):
        self._delta = delta   # positive = brighter, negative = darker

    @property
    def name(self) -> str:
        return "Brightness"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].astype(np.int32)
        roi = np.clip(roi + self._delta, 0, 255).astype(np.uint8)
        image[y:y + h, x:x + w] = roi
        return image


class PixelSwapAlt(ImageAlteration):
    """
    Horizontally flips (mirrors) a rectangular patch.
    Creates a subtle mirror-image artefact as the difference.
    """

    @property
    def name(self) -> str:
        return "PixelSwap"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].copy()
        image[y:y + h, x:x + w] = cv2.flip(roi, 1)   # 1 = horizontal flip
        return image


class TextureNoiseAlt(ImageAlteration):
    """
    Overlays a structured random noise pattern on a region,
    creating a subtle grainy texture difference.
    """

    def __init__(self, intensity: int = 40):
        self._intensity = intensity   # max pixel offset per channel

    @property
    def name(self) -> str:
        return "TextureNoise"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        noise = np.random.randint(-self._intensity, self._intensity,
                                   (h, w, 3), dtype=np.int32)
        roi = image[y:y + h, x:x + w].astype(np.int32)
        image[y:y + h, x:x + w] = np.clip(roi + noise, 0, 255).astype(np.uint8)
        return image


class ContrastAlt(ImageAlteration):
    """
    Reduces local contrast by scaling pixel values toward the region mean,
    producing a washed-out or flattened look.
    """

    def __init__(self, alpha: float = 0.55):
        self._alpha = alpha   # <1 reduces contrast, >1 increases it

    @property
    def name(self) -> str:
        return "Contrast"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].astype(np.float32)
        mean = roi.mean()
        adjusted = np.clip((roi - mean) * self._alpha + mean, 0, 255)
        image[y:y + h, x:x + w] = adjusted.astype(np.uint8)
        return image


# ImageProcessor — orchestrates loading and difference generation


class ImageProcessor:
    """
    Loads an image file, resizes it for display, creates an exact clone,
    then programmatically introduces exactly 5 non-overlapping differences
    into the clone using randomly selected alteration types.

    Demonstrates:
        - Encapsulation  : all image data and logic kept inside this class
        - Class interaction : uses DifferenceRegion and ImageAlteration objects
        - Polymorphism   : calls alt.apply() on any ImageAlteration subclass

    Attributes (available after load_image):
        original_bgr  (np.ndarray)          original image array
        modified_bgr  (np.ndarray)          cloned image with 5 differences
        differences   (List[DifferenceRegion])  the 5 difference locations
    """

    NUM_DIFFERENCES: int = 5
    MIN_REGION_SIZE: int = 40    # minimum side length in pixels
    MAX_REGION_SIZE: int = 90    # maximum side length in pixels

    def __init__(self) -> None:
        # All available alteration objects — more than 5 so we can randomise
        self._alteration_pool: List[ImageAlteration] = [
            ColourShiftAlt(hue_delta=25),
            ColourShiftAlt(hue_delta=-22),
            BrightnessAlt(delta=60),
            BrightnessAlt(delta=-60),
            PixelSwapAlt(),
            TextureNoiseAlt(intensity=45),
            ContrastAlt(alpha=0.50),
        ]

        # These are set when load_image() is called
        self.original_bgr: np.ndarray = None
        self.modified_bgr: np.ndarray = None
        self.differences: List[DifferenceRegion] = []
        self._img_w: int = 0
        self._img_h: int = 0

    
    # Public methods
    

    def load_image(self, path: str,
                   display_size: Tuple[int, int] = (550, 430)) -> bool:
        """
        Load the image at *path*, resize it to *display_size* (width, height),
        clone it, and apply 5 random non-overlapping differences.
        Returns True on success, False if the file cannot be opened.
        """
        raw = cv2.imread(path)
        if raw is None:
            return False    # file not found or unsupported format

        dw, dh = display_size
        self.original_bgr = cv2.resize(raw, (dw, dh),
                                        interpolation=cv2.INTER_AREA)
        self._img_w, self._img_h = dw, dh

        # Clone the original — this copy will receive the alterations
        self.modified_bgr = self.original_bgr.copy()
        self.differences = []

        self._apply_differences()
        return True

    def draw_circle(self, image: np.ndarray, region: DifferenceRegion,
                    colour: Tuple[int, int, int],
                    thickness: int = 3) -> np.ndarray:
        """Draw an anti-aliased circle centred on *region* onto *image*."""
        cx, cy = region.centre
        cv2.circle(image, (cx, cy), region.radius,
                   colour, thickness, lineType=cv2.LINE_AA)
        return image

    def get_original_with_overlays(
            self,
            found_regions: List[DifferenceRegion],
            revealed_regions: List[DifferenceRegion]) -> np.ndarray:
        """
        Return a copy of the original image with circles drawn:
          - RED   around differences the player has already found
          - BLUE  around differences that were revealed via the Reveal button
        """
        img = self.original_bgr.copy()
        for r in found_regions:
            self.draw_circle(img, r, (0, 0, 220))      # BGR → red
        for r in revealed_regions:
            self.draw_circle(img, r, (220, 100, 0))    # BGR → blue
        return img

    def get_modified_with_overlays(
            self,
            found_regions: List[DifferenceRegion],
            revealed_regions: List[DifferenceRegion]) -> np.ndarray:
        """Same as get_original_with_overlays but for the modified image."""
        img = self.modified_bgr.copy()
        for r in found_regions:
            self.draw_circle(img, r, (0, 0, 220))
        for r in revealed_regions:
            self.draw_circle(img, r, (220, 100, 0))
        return img

    
    # Private helpers
   

    def _apply_differences(self) -> None:
        """
        Randomly pick 5 distinct alterations from the pool, find a valid
        non-overlapping position for each, apply them to the clone, and
        store the resulting DifferenceRegion objects in self.differences.
        """
        random.shuffle(self._alteration_pool)
        chosen = random.sample(self._alteration_pool, self.NUM_DIFFERENCES)

        placed: List[DifferenceRegion] = []

        for alt in chosen:
            region = self._find_non_overlapping_region(placed)
            self.modified_bgr = alt.apply(
                self.modified_bgr,
                region.x, region.y, region.w, region.h
            )
            region.alteration_name = alt.name
            placed.append(region)

        self.differences = placed

    def _find_non_overlapping_region(
            self, placed: List[DifferenceRegion],
            max_attempts: int = 500) -> DifferenceRegion:
        """
        Randomly generate rectangles until one is found that does not
        overlap any region already in *placed* (with a 20-pixel buffer).
        If all attempts fail, return a fallback position.
        """
        for _ in range(max_attempts):
            w = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            h = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            x = random.randint(0, self._img_w - w)
            y = random.randint(0, self._img_h - h)
            candidate = DifferenceRegion(x=x, y=y, w=w, h=h)
            if not self._overlaps_any(candidate, placed):
                return candidate

        # Very rare fallback
        return DifferenceRegion(
            x=random.randint(0, self._img_w - self.MIN_REGION_SIZE),
            y=random.randint(0, self._img_h - self.MIN_REGION_SIZE),
            w=self.MIN_REGION_SIZE,
            h=self.MIN_REGION_SIZE,
        )

    @staticmethod
    def _overlaps_any(candidate: DifferenceRegion,
                      placed: List[DifferenceRegion],
                      padding: int = 20) -> bool:
        """
        Return True if *candidate* intersects any region in *placed*
        (expanded by *padding* pixels on all sides to ensure clear spacing).
        """
        cx1, cy1 = candidate.x - padding, candidate.y - padding
        cx2, cy2 = (candidate.x + candidate.w + padding,
                    candidate.y + candidate.h + padding)
        for p in placed:
            if cx1 < p.x + p.w and cx2 > p.x and cy1 < p.y + p.h and cy2 > p.y:
                return True
        return False


#Entry point for the Spot-the-Difference desktop application.
from gui import GameApp


def main() -> None:
    #Create and start the Tkinter application
    app = GameApp()
    app.mainloop()


if __name__ == "__main__":
    main()
