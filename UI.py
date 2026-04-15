from game2048 import *       
######
# UI #
######

import tkinter as tk
import numpy as np

# Color palette for the tiles (matching the classic 2048 look)
TILE_COLORS = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
}
# Text color: dark gray for 2 and 4, white for the rest
TEXT_COLORS = {2: "#776e65", 4: "#776e65"} 
DEFAULT_TEXT_COLOR = "#f9f6f2"

class Game2048UI(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.title("2048 AI Environment Tester")
        self.geometry("400x500")
        self.configure(bg="#92877d")
        
        # Header for Score
        self.score_var = tk.StringVar()
        self.score_label = tk.Label(
            self, textvariable=self.score_var, font=("Helvetica", 18, "bold"), 
            bg="#92877d", fg="white", pady=10
        )
        self.score_label.pack()

        # Frame for the grid
        self.grid_frame = tk.Frame(self, bg="#bbada0", bd=5)
        self.grid_frame.pack(pady=10)
        
        self.cells = []
        for i in range(self.env.size):
            row_cells = []
            for j in range(self.env.size):
                cell = tk.Label(
                    self.grid_frame, text="", bg=TILE_COLORS[0],
                    font=("Helvetica", 24, "bold"), width=4, height=2, relief="ridge"
                )
                cell.grid(row=i, column=j, padx=5, pady=5)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # Bind keyboard arrows to the game
        self.bind("<Key>", self.handle_keypress)
        
        # Initialize the first state
        self.env.reset()
        self.update_ui()

    def handle_keypress(self, event):
        """Maps keyboard arrows to the AI's action space."""
        if self.env.done:
            return

        key = event.keysym
        # Map Tkinter keys to our environment's actions (0: Up, 1: Right, 2: Down, 3: Left)
        action_map = {
            "Up": 0,
            "Right": 1,
            "Down": 2,
            "Left": 3
        }

        if key in action_map:
            # Step the environment forward just like the AI would
            state, reward, done, info = self.env.step(action_map[key])
            self.update_ui()
            
            if done:
                self.score_var.set(f"Game Over! Final Score: {self.env.score}")

    def update_ui(self):
        """Redraws the grid based on the current environment state."""
        state = self.env.get_state()
        self.score_var.set(f"Score: {self.env.score}")
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                value = state[i][j]
                cell = self.cells[i][j]
                
                if value == 0:
                    cell.configure(text="", bg=TILE_COLORS[0])
                else:
                    # Cap colors at 2048 for tiles that go beyond
                    bg_color = TILE_COLORS.get(value, "#3c3a32") 
                    fg_color = TEXT_COLORS.get(value, DEFAULT_TEXT_COLOR)
                    cell.configure(text=str(value), bg=bg_color, fg=fg_color)

# To run the game:
if __name__ == "__main__":
    # Ensure Game2048Env is defined above this in your file
    env = Game2048Env()
    app = Game2048UI(env)
    app.mainloop()