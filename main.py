"""
main.py
Install required libraries (only once):
       pip install opencv-python pillow
 """
#Entry point for the Spot-the-Difference desktop application.
from gui import GameApp


def main() -> None:
    #Create and start the Tkinter application
    app = GameApp()
    app.mainloop()


if __name__ == "__main__":
    main()
