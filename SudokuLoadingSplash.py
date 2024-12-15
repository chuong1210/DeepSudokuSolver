import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SplashScreen(ttk.Toplevel):
    def __init__(self):
        super().__init__()
        self.style.theme_use('darkly')  # Set the theme using ttkbootstrap



        self.title("Splash Screen")
        self.geometry("500x350")
        self.style.configure("TProgressbar", troughcolor='#2a2a2a', background='#808080')

        self.overrideredirect(True)
        self.center_window()
        
        self.setup_ui()
        self.after(100, self.animate)


    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self, padding=20)
        self.main_frame.pack(fill=BOTH, expand=YES)

        # Load images
        self.img_load1 = ImageTk.PhotoImage(Image.open(os.path.join('Repository', 'point2.png')))
        self.img_load2 = ImageTk.PhotoImage(Image.open(os.path.join('Repository', 'point1.png')))

        # Circular profile image
        profile_image =Image.open(os.path.join('Repository', 'sudoku.png'))
        self.profile_label = self.create_circular_image_label(profile_image, 190, 20)

        # Thesis name label
        self.name_thesis = ttk.Label(
            self.main_frame, 
            text='HUIT CHƯƠNG - THÁI', 
            font=("Game Of Squids", 23, "bold"),
            bootstyle="light"
        )
        self.name_thesis.pack(pady=(100, 10))

        # Loading dots frame
        self.dots_frame = ttk.Frame(self.main_frame)
        self.dots_frame.pack(pady=15)

        # Loading label
        self.loading_label = ttk.Label(
            self.main_frame, 
            text='Loading...', 
            font=("Helvetica", 10),
            bootstyle="light"
        )
        self.loading_label.pack(side=BOTTOM, anchor=SE, pady=(0, 20))

        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient=HORIZONTAL,
            length=400,
            mode='determinate',
            bootstyle="secondary-striped"
        )
        self.progress_bar.place(relx=0.5, rely=0.75, anchor=CENTER)

    def create_circular_image_label(self, image, x, y):
        image = image.resize((80, 80), Image.LANCZOS)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, image.width, image.height), fill=255)
        
        masked_image = Image.new("RGBA", image.size)
        masked_image.paste(image, (0, 0), mask)
        masked_image_tk = ImageTk.PhotoImage(masked_image)
        
        label = ttk.Label(self.main_frame, image=masked_image_tk)
        label.image = masked_image_tk
        label.place(x=x, y=y)
        return label

    def create_dot(self, image):
        return ttk.Label(self.dots_frame, image=image)

    def animate(self):
        dots = [self.create_dot(self.img_load2) for _ in range(5)]
        for dot in dots:
            dot.pack(side=LEFT, padx=2)

        for i in range(5):
            dots[i].configure(image=self.img_load1)
            self.update_idletasks()
            self.progress_bar['value'] += 20

            time.sleep(0.5)
            dots[i].configure(image=self.img_load2)

        for dot in dots:
            dot.destroy()

        self.after(100, self.open_main_window)

    def open_main_window(self):
        # Cancel all scheduled 'after' events
        try:
            for after_id in self.tk.eval('after info').split():
                self.after_cancel(after_id)
        except tk.TclError:
            print("Error cancelling 'after' events: Application may have been destroyed")

        # Ensure the window is still open before destroying
        try:
            if self.winfo_exists():
                self.destroy()
        except tk.TclError:
            print("Error destroying window: Application may have already been destroyed")
            # Ensure run_Home() is called even if there's an error

def runSplash():
    app = SplashScreen()
    app.mainloop()
# if __name__ == "__main__":
#     app = SplashScreen()

#     app.mainloop()

