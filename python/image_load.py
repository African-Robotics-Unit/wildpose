import cv2
import matplotlib.pyplot as plt
import os

def load_image(filepath):
    """
    Loads and shows an image from the specified filepath
    """

    im = cv2.imread(filepath)
    
    print(im)
    return(im)

def show_image(filepath):
    """
    Plots and displays a given image
    """
    im = load_image(filepath)
    plt.imshow(im)
    plt.show()

def convert_tif_to_png(filepath, filename):
    """
    Converts a given .tif image to a .png and saves the file
    """
    im = load_image(filepath)
    imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    out_path = os.path.join("data", filename)
    cv2.imwrite(out_path, imrgb)

if __name__ == "__main__":
    
    for i in range(10,20):

        path = "data//image0"+ str(i) + ".tif"
        convert_tif_to_png(path, "img" + str(i) +".png")
    