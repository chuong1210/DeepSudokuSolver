import tkinter as tk
from tkinter import Frame, Label, font, PhotoImage
from PIL import ImageTk, Image,ImageDraw
from SudokuHome import run_Home  # Assuming this is your login page module
import time
from tkinter import Canvas
import os
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
w = tk.Tk()

# Window dimensions and positioning
width_of_window = 477
height_of_window = 285
screen_width = w.winfo_screenwidth()
screen_height = w.winfo_screenheight()
x_coordinate = (screen_width / 2) - (width_of_window / 2)
y_coordinate = (screen_height / 2) - (height_of_window / 2)
w.geometry(f"{width_of_window}x{height_of_window}+{int(x_coordinate)}+{int(y_coordinate)}")
w.overrideredirect(1)  # Hide title bar

# New window to open (not used in this example)
def new_win():
    q = tk.Tk()
    q.title('main window')
    q.mainloop()

# Background frame
Frame(w, width=487, height=270, bg='#9999FF').place(x=0, y=0)

# Load images
img_load1 = ImageTk.PhotoImage(Image.open(os.path.join('Repository', 'point2.png')))
img_load2 = ImageTk.PhotoImage(Image.open(os.path.join('Repository', 'point1.png')))

# Function to create a circular image label
def create_circular_image_label(image, x, y, bg_color='#9999FF'):
    # Resize the image to be smaller for the circle
    image = image.resize((70, 70), Image.LANCZOS) 
    
    # Create a circular mask
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.width, image.height), fill=255)

    # Apply the mask to the image
    masked_image = Image.new("RGBA", image.size)
    masked_image.paste(image, (0, 0), mask)
    masked_image_tk = ImageTk.PhotoImage(masked_image)

    label = Label(w, image=masked_image_tk, bg=bg_color, borderwidth=0)
    label.image = masked_image_tk
    label.place(x=x, y=y)
    return label
# Function to create a label with border color matching the text
def create_image_label(image, x, y, bg_color='#9999FF'):
    canvas = Canvas(w, width=image.width(), height=image.height(), bg=bg_color)
    canvas.create_image(image.width() // 2, image.height() // 2, image=image)
    canvas.place(x=x - 10, y=y)
    return canvas

# Thesis name label with circular image
name_thesis = Label(w, text='HUIT CHƯƠNG', fg='white', bg='#9999FF')
name_thesis.configure(font=("Game Of Squids", 26, "bold"))
name_thesis.place(x=117, y=110)

# Add the circular image next to the label
profile_image = Image.open('Repository/play_icon.png') # Replace with your image path
profile_label = create_circular_image_label(profile_image, 210, 20) 

# Loading label
Loading_lb = Label(w, text='Loading...', fg='white', bg='#9999FF')
Loading_lb.configure(font=("Time", 14))
Loading_lb.place(x=370, y=215)

# Animation function
def animate():
    # Create labels with border color matching the text
    l1 = create_image_label(img_load1, 205, 170, bg_color='#9999FF')
    l2 = create_image_label(img_load2, 225, 170, bg_color='#9999FF')
    l3 = create_image_label(img_load2, 245, 170, bg_color='#9999FF')
    l4 = create_image_label(img_load2, 265, 170, bg_color='#9999FF')
    l5 = create_image_label(img_load2, 285, 170, bg_color='#9999FF')


    w.update_idletasks()
    time.sleep(0.5)

    l1 = create_image_label(img_load2, 205, 170, bg_color='#9999FF')
    l2 = create_image_label(img_load1, 225, 170, bg_color='#9999FF')
    l3 = create_image_label(img_load2, 245, 170, bg_color='#9999FF')
    l4 = create_image_label(img_load2, 265, 170, bg_color='#9999FF')
    l5 = create_image_label(img_load2, 285, 170, bg_color='#9999FF')

    w.update_idletasks()
    time.sleep(0.5)

    l1 = create_image_label(img_load2, 205, 170, bg_color='#9999FF')     
    l2 = create_image_label(img_load2, 225, 170, bg_color='#9999FF')
    l3 = create_image_label(img_load1, 245, 170, bg_color='#9999FF')
    l4 = create_image_label(img_load2, 265, 170, bg_color='#9999FF')
    l5 = create_image_label(img_load2, 285, 170, bg_color='#9999FF')

    w.update_idletasks()
    time.sleep(0.5)

    l1 = create_image_label(img_load2, 205, 170, bg_color='#9999FF')
    l2 = create_image_label(img_load2, 225, 170, bg_color='#9999FF')
    l3 = create_image_label(img_load2, 245, 170, bg_color='#9999FF')
    l4 = create_image_label(img_load1, 265, 170, bg_color='#9999FF')
    l5 = create_image_label(img_load2, 285, 170, bg_color='#9999FF')

    w.update_idletasks()
    time.sleep(0.5)

    
    l1 = create_image_label(img_load2, 205, 170, bg_color='#9999FF')
    l2 = create_image_label(img_load2, 225, 170, bg_color='#9999FF')
    l3 = create_image_label(img_load2, 245, 170, bg_color='#9999FF')
    l4 = create_image_label(img_load2, 265, 170, bg_color='#9999FF')
    l5 = create_image_label(img_load1, 285, 170, bg_color='#9999FF')

    w.update_idletasks()
    time.sleep(0.5)

    # Destroy all labels to avoid memory leaks
    l1.destroy()
    l2.destroy()
    l3.destroy()
    l4.destroy()
    l5.destroy()

    # Open the login page after animation
    w.destroy()
    run_Home()

# Start the animation
animate()

# Keep the window running
w.mainloop()