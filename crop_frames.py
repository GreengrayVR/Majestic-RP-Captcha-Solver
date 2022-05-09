# Importing Image class from PIL module
from PIL import Image
 
# Opens a image in RGB mode

for i in range(2000):
    print(i)
    im = Image.open(r"C:\Coding\python\MajesticRP\ExternalMeatFarm\dataset\d\frame" + str(i) + ".jpg")
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    
    # Setting the points for cropped image
    left = 920
    top = 963
    right = 1000
    bottom = 1040
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))

    im1.save(r"C:\Coding\python\MajesticRP\ExternalMeatFarm\dataset\cropped\d\frame" + str(i) + ".jpg", "JPEG")