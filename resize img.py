import cv2

def print_file_size(file):

    File_Size = os.path.getsize(file)
    File_Size_MB = round(File_Size/1024/1024,2)

print("Image File Size is " + str(File_Size_MB) + "MB" )

#read image
img=cv2.imread("valo.png")

print_file_size("valo.png")

cv2.imwrite("valo.png", img, [cv2.IMWRITE_PNG_QUALITY, 100])

print("Image Saved Successfully!!")

print_file_size("valo.png") 