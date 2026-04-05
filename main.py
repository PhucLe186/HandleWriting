import os
import sys
import tkinter as tk

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ui.main_window import SystemCamera

def main():
    app = SystemCamera()
    app.mainloop()

if __name__ == "__main__":
    main()