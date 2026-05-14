"""
gui.py
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


# Colour palette — dark theme with gold accent

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

# ImagePanel — reusable image display widget (inherits tk.Frame)

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


    # Public methods


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


    # Private helpers
   

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


# StatusBar — HUD display widget (inherits tk.Frame)


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



# GameApp — root application window


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

    # UI construction


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


    # Button command handlers
  

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


    # GameState callback handlers (fired by GameState, executed on GUI thread)


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

  
    # Click forwarding
    
    def _on_image_click(self, x: int, y: int) -> None:
        """
        Receives a click coordinate from ImagePanel and passes it to
        GameState for evaluation.  GameState fires the appropriate callback.
        """
        if not self._state.round_over:
            self._state.process_click(x, y)

    # Image rendering helpers


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
