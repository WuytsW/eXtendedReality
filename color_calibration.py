import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

#beans

COLOR_COLLECTION_FILE = "color_collection.npz"

class DoubleSlider(tk.Canvas):
    """A custom double-ended slider for Tkinter."""
    def __init__(self, master, from_, to, init_l, init_u, marks=None, width=250, height=40, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.from_ = from_
        self.to = to
        self.width = width
        self.height = height
        self.slider_length = width - 40
        self.slider_y = height // 2
        self.radius = 8
        self.marks = marks if marks else []
        self.lower = init_l
        self.upper = init_u
        self.active = None
        self.bind("<Button-1>", self.click)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)
        self.draw_slider()

    def value_to_x(self, value):
        return 20 + int((value - self.from_) / (self.to - self.from_) * self.slider_length)

    def x_to_value(self, x):
        x = min(max(x, 20), 20 + self.slider_length)
        return self.from_ + (x - 20) / self.slider_length * (self.to - self.from_)

    def draw_slider(self):
        self.delete("all")
        # Draw line
        self.create_line(20, self.slider_y, 20 + self.slider_length, self.slider_y, width=4, fill="#aaa")
        # Draw marks
        for mark in self.marks:
            mx = self.value_to_x(mark['value'])
            self.create_line(mx, self.slider_y-12, mx, self.slider_y+12, fill=mark.get('color', '#888'), width=2)
            self.create_text(mx, self.slider_y+18, text=str(mark['label']), fill=mark.get('color', '#888'), font=("Arial", 8))
        # Draw lower and upper handles
        lx = self.value_to_x(self.lower)
        ux = self.value_to_x(self.upper)
        self.create_oval(lx-self.radius, self.slider_y-self.radius, lx+self.radius, self.slider_y+self.radius, fill="#0a0", outline="#000", width=2, tags="lower")
        self.create_oval(ux-self.radius, self.slider_y-self.radius, ux+self.radius, self.slider_y+self.radius, fill="#a00", outline="#000", width=2, tags="upper")
        # Draw range highlight
        self.create_line(lx, self.slider_y, ux, self.slider_y, width=8, fill="#0af", tags="range")

    def click(self, event):
        lx = self.value_to_x(self.lower)
        ux = self.value_to_x(self.upper)
        if abs(event.x - lx) < self.radius*2:
            self.active = 'lower'
        elif abs(event.x - ux) < self.radius*2:
            self.active = 'upper'
        else:
            self.active = None

    def drag(self, event):
        if self.active:
            value = int(round(self.x_to_value(event.x)))
            if self.active == 'lower':
                value = min(value, self.upper)
                value = max(self.from_, value)
                self.lower = value
            elif self.active == 'upper':
                value = max(value, self.lower)
                value = min(self.to, value)
                self.upper = value
            self.draw_slider()

    def release(self, event):
        self.active = None

    def get(self):
        return self.lower, self.upper

class ColorCalibrationApp:
    def __init__(self, root, homography=None):
        self.root = root
        self.root.title("Color Calibration")
        self.cap = cv2.VideoCapture(0)
        self.homography = homography
        self.frame = None
        self.display_width = 640
        self.display_height = 480
        self.selected_hsvs = []  # Store multiple HSVs
        self.selected_bgrs = []

        # UI
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_homography_btn = tk.Button(self.top_frame, text="Load Homography", command=self.load_homography)
        self.load_homography_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.reset_warp_btn = tk.Button(self.top_frame, text="Reset Warping", command=self.reset_warping)
        self.reset_warp_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.save_color_btn = tk.Button(self.top_frame, text="Save Color", command=self.save_color)
        self.save_color_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.view_colors_btn = tk.Button(self.top_frame, text="View Stored Colors", command=self.view_colors)
        self.view_colors_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.delete_last_btn = tk.Button(self.top_frame, text="Delete Last Point", command=self.delete_last_point)
        self.delete_last_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.new_color_btn = tk.Button(self.top_frame, text="New Color Selection", command=self.new_color_selection)
        self.new_color_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.label = tk.Label(self.top_frame, text="Click on the image to sample HSV (multiple allowed).")
        self.label.pack(side=tk.LEFT, padx=10)

        # Main frame to hold camera and HSV info side by side
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, width=self.display_width, height=self.display_height)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # HSV display on the right of the camera feed
        self.hsv_frame = tk.Frame(self.main_frame)
        self.hsv_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Frame for HSV values and color boxes (side by side)
        self.hsv_values_frame = tk.Frame(self.hsv_frame)
        self.hsv_values_frame.pack(side=tk.TOP, anchor="nw", pady=10)

        # Column headers
        tk.Label(self.hsv_values_frame, text="#", width=3, font=("Arial", 10, "bold")).grid(row=0, column=0)
        tk.Label(self.hsv_values_frame, text="Color", width=8, font=("Arial", 10, "bold")).grid(row=0, column=1)
        tk.Label(self.hsv_values_frame, text="H", width=5, font=("Arial", 10, "bold")).grid(row=0, column=2)
        tk.Label(self.hsv_values_frame, text="S", width=5, font=("Arial", 10, "bold")).grid(row=0, column=3)
        tk.Label(self.hsv_values_frame, text="V", width=5, font=("Arial", 10, "bold")).grid(row=0, column=4)

        # Frame for HSV stats matrix (will be placed next to the color column)
        self.stats_frame = tk.Frame(self.hsv_values_frame)
        self.stats_frame.grid(row=0, column=5, rowspan=100, padx=(20, 0), sticky="nw")

        self.update_frame()

    def load_homography(self):
        file_path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if file_path:
            data = np.load(file_path)
            if "homography" in data:
                self.homography = data["homography"]
                self.label.config(text="Homography loaded.")
            else:
                self.label.config(text="No homography found in file.")

    def reset_warping(self):
        self.homography = None
        self.label.config(text="Warping reset. Showing raw camera feed.")

    def save_color(self):
        if not self.selected_hsvs:
            self.label.config(text="No color selected to save.")
            return
        hsvs = np.array(self.selected_hsvs)
        lower_bound = np.min(hsvs, axis=0)
        upper_bound = np.max(hsvs, axis=0)
        avg_hsv = np.mean(hsvs, axis=0).astype(int)
        min_hsv = lower_bound
        max_hsv = upper_bound

        # Load existing collection and conditions
        collection = {}
        existing_conditions = set()
        if os.path.exists(COLOR_COLLECTION_FILE):
            try:
                collection = dict(np.load(COLOR_COLLECTION_FILE, allow_pickle=True))
                for v in collection.values():
                    v = v.item() if hasattr(v, "item") else v
                    cond = v.get("condition", "")
                    if cond:
                        existing_conditions.add(cond)
            except Exception:
                collection = {}

        def on_save_dialog():
            color_name = name_entry.get()
            if not color_name:
                messagebox.showerror("Error", "No name given.")
                return
            l_h, u_h = h_slider.get()
            l_s, u_s = s_slider.get()
            l_v, u_v = v_slider.get()
            lower = np.array([l_h, l_s, l_v], dtype=np.uint8)
            upper = np.array([u_h, u_s, u_v], dtype=np.uint8)
            # Get condition from dropdown or entry
            selected = condition_var.get()
            if selected == "__new__":
                condition = condition_entry.get()
            else:
                condition = selected
            # Store as a dict: name -> (lower, upper, condition)
            collection[color_name] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "condition": condition
            }
            np.savez(COLOR_COLLECTION_FILE, **collection)
            dialog.destroy()
            self.label.config(text=f"Color '{color_name}' saved to collection.")

        dialog = Toplevel(self.root)
        dialog.title("Save Color")

        # Matrix as reference
        tk.Label(dialog, text="HSV Matrix (min/avg/max):", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=(10, 2))
        tk.Label(dialog, text="", width=5, font=("Arial", 10, "bold")).grid(row=1, column=0)
        tk.Label(dialog, text="min", width=7, font=("Arial", 10, "bold")).grid(row=1, column=1)
        tk.Label(dialog, text="avg", width=7, font=("Arial", 10, "bold")).grid(row=1, column=2)
        tk.Label(dialog, text="max", width=7, font=("Arial", 10, "bold")).grid(row=1, column=3)
        for i, (label, arr) in enumerate([("H", hsvs[:, 0]), ("S", hsvs[:, 1]), ("V", hsvs[:, 2])]):
            tk.Label(dialog, text=label, width=5, font=("Arial", 10, "bold")).grid(row=i+2, column=0)
            tk.Label(dialog, text=str(np.min(arr)), width=7).grid(row=i+2, column=1)
            tk.Label(dialog, text=str(int(np.round(np.mean(arr)))), width=7).grid(row=i+2, column=2)
            tk.Label(dialog, text=str(np.max(arr)), width=7).grid(row=i+2, column=3)

        # Avg HSV color box
        avg_bgr = cv2.cvtColor(np.uint8([[avg_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        avg_hex = '#%02x%02x%02x' % (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        tk.Label(dialog, text="Avg HSV Color:").grid(row=5, column=0, pady=(10, 2), sticky="e")
        tk.Label(dialog, width=10, height=2, bg=avg_hex).grid(row=5, column=1, columnspan=3, pady=(10, 2), sticky="w")

        # Sliders for H, S, V with marks and two selectors
        tk.Label(dialog, text="Adjust HSV bounds:", font=("Arial", 10, "bold")).grid(row=6, column=0, columnspan=4, pady=(10, 2))
        # H
        h_marks = [
            {'value': int(min_hsv[0]), 'label': 'min', 'color': '#888'},
            {'value': int(avg_hsv[0]), 'label': 'avg', 'color': '#0af'},
            {'value': int(max_hsv[0]), 'label': 'max', 'color': '#888'}
        ]
        h_slider = DoubleSlider(dialog, from_=0, to=179, init_l=int(min_hsv[0]), init_u=int(max_hsv[0]), marks=h_marks)
        h_slider.grid(row=7, column=0, columnspan=4, pady=2)
        # S
        s_marks = [
            {'value': int(min_hsv[1]), 'label': 'min', 'color': '#888'},
            {'value': int(avg_hsv[1]), 'label': 'avg', 'color': '#0af'},
            {'value': int(max_hsv[1]), 'label': 'max', 'color': '#888'}
        ]
        s_slider = DoubleSlider(dialog, from_=0, to=255, init_l=int(min_hsv[1]), init_u=int(max_hsv[1]), marks=s_marks)
        s_slider.grid(row=8, column=0, columnspan=4, pady=2)
        # V
        v_marks = [
            {'value': int(min_hsv[2]), 'label': 'min', 'color': '#888'},
            {'value': int(avg_hsv[2]), 'label': 'avg', 'color': '#0af'},
            {'value': int(max_hsv[2]), 'label': 'max', 'color': '#888'}
        ]
        v_slider = DoubleSlider(dialog, from_=0, to=255, init_l=int(min_hsv[2]), init_u=int(max_hsv[2]), marks=v_marks)
        v_slider.grid(row=9, column=0, columnspan=4, pady=2)

        # Name entry and save button
        tk.Label(dialog, text="Color Name:").grid(row=10, column=0, pady=(10, 2), sticky="e")
        name_entry = tk.Entry(dialog)
        name_entry.grid(row=10, column=1, columnspan=2, pady=(10, 2), sticky="w")

        # Condition selection: dropdown for existing, or entry for new
        tk.Label(dialog, text="Condition:").grid(row=11, column=0, pady=(10, 2), sticky="e")
        condition_var = tk.StringVar(dialog)
        condition_options = sorted(existing_conditions)
        condition_menu = tk.OptionMenu(dialog, condition_var, *condition_options, "__new__")
        condition_menu.grid(row=11, column=1, pady=(10, 2), sticky="w")
        tk.Label(dialog, text="(Choose existing or select '__new__' to enter new)").grid(row=11, column=2, columnspan=2, sticky="w")
        condition_entry = tk.Entry(dialog)
        condition_entry.grid(row=12, column=1, columnspan=2, pady=(2, 2), sticky="w")
        condition_entry.config(state="disabled")

        def on_condition_change(*args):
            if condition_var.get() == "__new__":
                condition_entry.config(state="normal")
            else:
                condition_entry.delete(0, tk.END)
                condition_entry.config(state="disabled")
        condition_var.trace_add("write", on_condition_change)
        condition_var.set(condition_options[0] if condition_options else "__new__")

        save_btn = tk.Button(dialog, text="Save", command=on_save_dialog)
        save_btn.grid(row=13, column=0, columnspan=4, pady=(10, 10))

    def view_colors(self):
        if not os.path.exists(COLOR_COLLECTION_FILE):
            messagebox.showinfo("No Colors", "No color collection file found.")
            return
        try:
            collection = dict(np.load(COLOR_COLLECTION_FILE, allow_pickle=True))
        except Exception:
            messagebox.showerror("Error", "Could not load color collection file.")
            return

        dialog = Toplevel(self.root)
        dialog.title("Stored Colors")

        tk.Label(dialog, text="Stored Colors", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=5, pady=10)

        tk.Label(dialog, text="Name", font=("Arial", 10, "bold")).grid(row=1, column=0)
        tk.Label(dialog, text="Lower HSV", font=("Arial", 10, "bold")).grid(row=1, column=1)
        tk.Label(dialog, text="Upper HSV", font=("Arial", 10, "bold")).grid(row=1, column=2)
        tk.Label(dialog, text="Color", font=("Arial", 10, "bold")).grid(row=1, column=3)
        tk.Label(dialog, text="Condition", font=("Arial", 10, "bold")).grid(row=1, column=4)

        for i, (name, data) in enumerate(collection.items()):
            lower = data.item().get("lower_bound") if hasattr(data, "item") else data["lower_bound"]
            upper = data.item().get("upper_bound") if hasattr(data, "item") else data["upper_bound"]
            condition = data.item().get("condition", "") if hasattr(data, "item") else data.get("condition", "")
            avg_hsv = ((lower.astype(int) + upper.astype(int)) // 2).astype(np.uint8)
            avg_bgr = cv2.cvtColor(np.uint8([[avg_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
            avg_hex = '#%02x%02x%02x' % (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
            tk.Label(dialog, text=str(name)).grid(row=i+2, column=0)
            tk.Label(dialog, text=str(lower)).grid(row=i+2, column=1)
            tk.Label(dialog, text=str(upper)).grid(row=i+2, column=2)
            tk.Label(dialog, width=6, height=1, bg=avg_hex).grid(row=i+2, column=3, padx=4, pady=2)
            tk.Label(dialog, text=str(condition)).grid(row=i+2, column=4)

    def delete_last_point(self):
        if self.selected_hsvs and self.selected_bgrs:
            self.selected_hsvs.pop()
            self.selected_bgrs.pop()
            self.update_hsv_display()
        else:
            self.label.config(text="No points to delete.")

    def new_color_selection(self):
        self.selected_hsvs = []
        self.selected_bgrs = []
        self.update_hsv_display()
        self.label.config(text="Started new color selection. Click on the image to sample HSV.")

    def update_hsv_display(self):
        # Clear previous widgets
        for widget in self.hsv_values_frame.winfo_children():
            # Don't destroy the stats_frame itself
            if widget is not self.stats_frame and int(widget.grid_info().get("row", 0)) > 0:
                widget.destroy()
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        # Show each color sample as a row: index, color box, H, S, V
        for i, (bgr, hsv) in enumerate(zip(self.selected_bgrs, self.selected_hsvs)):
            tk.Label(self.hsv_values_frame, text=str(i+1), width=3).grid(row=i+1, column=0)
            hex_color = '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            box = tk.Label(self.hsv_values_frame, width=4, height=1, bg=hex_color)
            box.grid(row=i+1, column=1, padx=2)
            tk.Label(self.hsv_values_frame, text=str(hsv[0]), width=5).grid(row=i+1, column=2)
            tk.Label(self.hsv_values_frame, text=str(hsv[1]), width=5).grid(row=i+1, column=3)
            tk.Label(self.hsv_values_frame, text=str(hsv[2]), width=5).grid(row=i+1, column=4)

        # Show 3x3 matrix: rows H/S/V, columns min/avg/max, placed next to the color column
        if self.selected_hsvs:
            hsvs = np.array(self.selected_hsvs)
            stats = [
                ("H", hsvs[:, 0]),
                ("S", hsvs[:, 1]),
                ("V", hsvs[:, 2])
            ]
            # Headers
            tk.Label(self.stats_frame, text="", width=5, font=("Arial", 10, "bold")).grid(row=0, column=0)
            tk.Label(self.stats_frame, text="min", width=7, font=("Arial", 10, "bold")).grid(row=0, column=1)
            tk.Label(self.stats_frame, text="avg", width=7, font=("Arial", 10, "bold")).grid(row=0, column=2)
            tk.Label(self.stats_frame, text="max", width=7, font=("Arial", 10, "bold")).grid(row=0, column=3)
            for i, (label, arr) in enumerate(stats):
                tk.Label(self.stats_frame, text=label, width=5, font=("Arial", 10, "bold")).grid(row=i+1, column=0)
                tk.Label(self.stats_frame, text=str(np.min(arr)), width=7).grid(row=i+1, column=1)
                tk.Label(self.stats_frame, text=str(int(np.round(np.mean(arr)))), width=7).grid(row=i+1, column=2)
                tk.Label(self.stats_frame, text=str(np.max(arr)), width=7).grid(row=i+1, column=3)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame_disp = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        if self.homography is not None:
            frame_disp = cv2.warpPerspective(frame_disp, self.homography, (self.display_width, self.display_height))

        self.frame = frame_disp.copy()
        img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.root.after(30, self.update_frame)

    def on_canvas_click(self, event):
        x = int(event.x)
        y = int(event.y)
        if self.frame is not None and 0 <= x < self.display_width and 0 <= y < self.display_height:
            bgr = self.frame[y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            self.selected_bgrs.append(bgr)
            self.selected_hsvs.append(hsv)
            self.update_hsv_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorCalibrationApp(root)
    root.mainloop()
