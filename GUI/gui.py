import tkinter as tk
from PIL import ImageTk, Image
from PredictFunctions import *
from pathlib import Path
import os


root = tk.Tk()
root.title("MoleDetection")
root.geometry("900x800")
root.minsize(width=900, height=800)
root.configure(bg="#E5D3C3")
cbg = "#715F55"  # canvas bg colour
fbg = "#E5D3C3"  # frame bg colour

# Define Global Data Variables
OGpath = 0
OriginalImage = 0
AnnotatedImage = 0
MaskImage = 0
SegmentedImage = 0
BboxImage = 0
Coordinates = 0

# Define Garbage dump avoidance Variables
DetectBtnVisible = False
ImageCopy = 0
im1 = 0
im2 = 0
im3 = 0
im4 = 0

# Asset Images
NoDetectImage = "./assets/NoMole.png"
NoDetectImage = getabsPath(NoDetectImage)
NoDetectImage = cv2.imread(NoDetectImage)
UploadIcon = "./assets/UploadIt.png"
UploadIcon = cv2.cvtColor(cv2.resize(cv2.imread(getabsPath(UploadIcon)), (600, 500)), cv2.COLOR_BGR2RGB)
UploadIcon = ImageTk.PhotoImage(image=Image.fromarray(UploadIcon))  # TK image
TitleFont = ("MS Serif", 16, "bold")


# Defining the Model path for YOLO
PathFromContentRoot = "./Model5ep32/weights/best.pt"
Absolute_ModelPath = getabsPath(PathFromContentRoot)

def hide_Homepage():
    HomePage.pack_forget()

def show_Homepage():
    global OGpath
    global DetectBtnVisible
    global OriginalImage
    DetectPage.pack_forget()
    center_frame.pack_forget()

    OGpath = ""
    DisplayCanvas.delete("all")
    DisplayCanvas.pack()
    DetectBtn.pack_forget()
    DetectBtnVisible = False
    OriginalImage = 0
    HomePage.pack()

def show_DetectPage():
    hide_Homepage()
    DisplayImages()


def hide_DetectPage():
    DetectPage.pack_forget()
    center_frame.pack_forget()
    show_Homepage()

def BackToHome():
    imPathLabel.config(text="No Image Selected")
    hide_DetectPage()


def UploadImage():
    global OGpath, OriginalImage, ImageCopy, DetectBtnVisible

    # Make sure we start with empty canvas, else display issues
    OGpath = ""
    DisplayCanvas.delete("all")
    OriginalImage = 0

    OGpath = openFile()  # Run file explorer for input image
    try:
        # Assigning image
        OriginalImage = cv2.imread(OGpath)  # The selected image get assigned to global variable to be accessed in Detection
        # Displaying into GUI (below)
        imPathLabel.config(text=OGpath)
        cv2_image = cv2.cvtColor(cv2.resize(OriginalImage, (600, 500)), cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(cv2_image)
        ImageCopy = ImageTk.PhotoImage(image=PIL_image)  # A global variable is needed otherwise image is garbage dumped
        DisplayCanvas.create_image(0, 0, anchor=tk.NW, image=ImageCopy)
        DetectBtn.pack()
        DetectBtnVisible = True
    except:
        # If they cancel their upload/ exit file explorer without choosing image
        print("fail in upload image")
        DetectBtn.pack_forget()
        imPathLabel.config(text="No image selected")
        DetectBtnVisible = False

def DetectBtnClick():
    show_DetectPage()
    DownloadBtn.pack()


def DisplayImages():
    global OriginalImage
    global AnnotatedImage
    global MaskImage
    global SegmentedImage
    global BboxImage
    global Coordinates
    global NoDetectImage
    global im1
    global im2
    global im3
    global im4

    DetectBool = False
    # Getting the relative path of YOLO weights
    global Absolute_ModelPath
    # Reset images
    DetectIt = YOLOsegmentation(Absolute_ModelPath)
    try:
        AnnotatedImage, MaskImage, SegmentedImage, BboxImage, Coordinates, DetectBool = DetectIt.Predictions(OriginalImage)
        if DetectBool == False:
            messagebox.showerror("showerror", "No Mole(s) detected. Try with a different image.")
            BackToHome()

        else:
            # Annotated Image
            im1 = cv2.cvtColor(cv2.resize(AnnotatedImage, (CW, CH)),
                               cv2.COLOR_BGR2RGB)  # Resize image and reshape to fit Display canvas
            A_PIL_Image = Image.fromarray(im1)  # Turns cv2 image to PIL image
            im1 = ImageTk.PhotoImage(image=A_PIL_Image)  # Converts PIL image to Tkinter Photo Image
            AnnotatedCanvas.create_image(0, 0, anchor=tk.NW, image=im1)  # upload the image into the canvas "create_image"
            AnnotatedCanvas.pack()

            # The above code but in a single line
            # Mask Image
            im2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(MaskImage, (CW, CH)), cv2.COLOR_BGR2RGB)))
            MaskCanvas.create_image(0, 0, anchor=tk.NW, image=im2)
            MaskCanvas.pack()

            # Segmented Lesion Image
            im3 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(SegmentedImage, (CW, CH)), cv2.COLOR_BGR2RGB)))
            SegCanvas.create_image(0, 0, anchor=tk.NW, image=im3)
            SegCanvas.pack()

            # Bounding Box Image
            im4 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(BboxImage, (CW, CH)), cv2.COLOR_BGR2RGB)))
            BboxCanvas.create_image(0, 0, anchor=tk.NW, image=im4)
            BboxCanvas.pack()

            DetectPage.pack()
            center_frame.pack()
    except:
        messagebox.showerror("showerror", "No Mole(s) detected. Try with a different image.")
        BackToHome()
        UploadImage()

def DownloadAll():
    global OGpath
    global MaskImage
    global SegmentedImage
    global Coordinates

    Output_fname = os.path.basename(OGpath)  # the name of our Selected image

    BboxPath = "./Detections/Bbox"  # Where the Bbox coordinates are saved
    maskPath = "./Detections/Mask"
    segPath = "./Detections/Seg"

    # Downloading Mask
    maskPath = os.path.abspath(maskPath) / Path(Output_fname)  # abs.path is the absolute path to reach "Detections/Mask"
    cv2.imwrite(maskPath, MaskImage)                           # '/' combines the 2 strings using with a "/"
                                                               # so... a ="okay", b ="today" then print(a/b) = "okay/today"
    # Downloading Segmented lesion
    segPath = os.path.abspath(segPath) / Path(Output_fname)
    cv2.imwrite(segPath, SegmentedImage)

    # Downloading text file of Coordinates ([[x,y,x1,y1],...,[xn,yn,xn+1,yn+1]])
    Output_fname = (Output_fname.split('.')[0] + ".txt")
    BboxPath = os.path.abspath(BboxPath) / Path(Output_fname)
    file = open(BboxPath, 'w')
    file.write(str(Coordinates))
    file.close()

    messagebox.showinfo("Success", "Download Successful!")
    BackToHome()


################################################### HomePage: #################################
HomePage = tk.Frame(root)
BannerLabel = tk.Label(HomePage, text="Mole Detection", font=TitleFont)
BannerLabel.pack()

imPathLabel = tk.Label(HomePage, text="No image Selected")
imPathLabel.pack()

DisplayCanvas = tk.Canvas(HomePage, width=600, height=500, bg=cbg)
DisplayCanvas.create_image(0, 0, anchor=tk.NW, image=UploadIcon)
DisplayCanvas.pack()

UploadBtn = tk.Button(HomePage, text="Upload Image",
                      command=lambda: UploadImage())
UploadBtn.pack()

DetectBtn = tk.Button(HomePage, text="Detect",
                              command=lambda: DetectBtnClick())

HomePage.place()

################################################### DetectPage: #################################
# Uses same BannerLabel for title
CW = 350
CH = 300
# Create a center frame to hold the 2x2 grid
DetectPage = tk.Frame(root)
DetectPage.pack()

Banner2Label = tk.Label(DetectPage, text="Detection", font=TitleFont)
Banner2Label.pack()

center_frame = tk.Frame(DetectPage)
center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create (2x2) grid using frames
row1 = tk.Frame(center_frame)
row1.pack(side=tk.LEFT, padx=10, pady=10)

row2 = tk.Frame(center_frame)
row2.pack(side=tk.LEFT, padx=10, pady=10)

# Labels for each image
label1 = tk.Label(row1, text="Detection(s)")
label1.pack()
AnnotatedCanvas = tk.Canvas(row1, width=CW, height=CH, bg=cbg)
AnnotatedCanvas.pack(padx=10, pady=10)


label2 = tk.Label(row1, text="Mask")
label2.pack()
MaskCanvas = tk.Canvas(row1, width=CW, height=CH, bg=cbg)
MaskCanvas.pack(padx=10, pady=10)


label3 = tk.Label(row2, text="Segmented Lesion(s)")
label3.pack()
SegCanvas = tk.Canvas(row2, width=CW, height=CH, bg=cbg)
SegCanvas.pack(padx=10, pady=10)


label4 = tk.Label(row2, text="Bounding Box")
label4.pack()
BboxCanvas = tk.Canvas(row2, width=CW, height=CH, bg=cbg)
BboxCanvas.pack(padx=10, pady=10)

underFrame = tk.Frame(DetectPage)
underFrame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

DownloadBtn = tk.Button(underFrame, text="Download All",
                        command=lambda: DownloadAll())
DownloadBtn.pack()


HomePageBtn = tk.Button(underFrame, text="Back to Home",
                                command=lambda: BackToHome())
HomePageBtn.pack()

QuitBtn = tk.Button(underFrame, text="Quit",
                            command=lambda: root.destroy())
QuitBtn.pack()

# To initialise the look. Hide the DetectPage widgets and show the HomePage widgets
hide_DetectPage()
show_Homepage()

root.mainloop()
