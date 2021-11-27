import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

# Load input file
input_file = 'data/' + 'birds.jpg'
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# Check for success
if img is None:
    print('Failed to open', input_file)
    sys.exit()

# Calculate the histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = hist.reshape(256)

# Plot histogram
plt.bar(np.linspace(0,255,256), hist)
plt.title('Histogram')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
#plt.show()

#=====================================================================

# Threshold manually using For Loop - Thres Values taken - 60, 100, 140
# Thresholding at T=140 gave best result with some loss of detail
   
for i in range(60,150,40):
    _, output = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
    output_file_name = 'data/' + 'birds_thresh_' + str(i) + '.jpg'
    cv2.imwrite(output_file_name, output)
    print(output_file_name)

#=====================================================================

# Subtracting Gaussian blurred image from original image
# Subtracting gaussian smoothed image at sd = 18 from original image gave the best result
    
for i in range(6,20,6):
    # Gaussian Smoothing with sigma values as 6, 12, 18
    img = cv2.imread('data/' + 'birds.jpg', cv2.IMREAD_GRAYSCALE)
    gauss = cv2.GaussianBlur(img, (0,0), i)
    output_file_name = 'data/' + 'birds_gaussian_sd' + str(i) + '.jpg'
    cv2.imwrite(output_file_name, gauss)
    print(output_file_name)
    
    # Convert images to int (so we can do signed maths)
    gauss = cv2.cvtColor(gauss, cv2.IMREAD_GRAYSCALE).astype(np.int)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE).astype(np.int)

    # Find the difference of the gaussian and the original image to see what has changed.
    output = abs(img - gauss).astype(np.uint8)
    output_file_name = 'data/' + 'birds_diff_orig_gaussian_sd' + str(i) + '.jpg'
    cv2.imwrite(output_file_name, output)
    print(output_file_name)

#=====================================================================

#Generate histogram of image - obtained by subtracting gaussian smoothed image at sd = 18 from original image

img = cv2.imread('data/' + 'birds_diff_orig_gaussian_sd18.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = hist.reshape(256)

# Plot histogram
plt.bar(np.linspace(0,255,256), hist)
plt.title('Histogram')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
#plt.show()

#Thresholding image obtained by subtracting gaussian smoothed image at sd = 18 from original image
T, output = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
output_file_name = 'data/' + 'birds_diff_orig_gaussian_sd18_thres_otsu.jpg' # We will use this file for object labelling
print("Threshold value found by Otsu's method:", T) #Thres Value obtained - 73
cv2.imwrite(output_file_name, output)
print(output_file_name)

#=====================================================================

struc_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Erode (a number of times) and then Dilate (a number of times)
for i in range(2):
    output = cv2.erode(output, struc_elem)
for i in range(2):
    output = cv2.dilate(output, struc_elem)

output_file_name = 'data/' + 'birds_diff_orig_gaussian_sd18_thres_otsu_opening.jpg'
cv2.imwrite(output_file_name, output)
print(output_file_name)

#=====================================================================

#Image labelling Algorithm
#Input Image =  binary thresholded image obtained by subtracting gaussian smoothed image at sd = 18 from original image

img = cv2.imread('data/' + 'birds_diff_orig_gaussian_sd18_thres_otsu.jpg', cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#print(img.shape)

assigned_label = [1] #Initializing label list

#1st iteration to input image 
#Assign new pixel label to all the non white/black pixels
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        
        if img[r, c] == 0:
            #print('=====Black Pixel=====')
            continue
        
        if img[r, c] == 255:
            #print('=====White Pixel=====')

            #Create Neighbour Pixel List
            NW_pixel = img[abs(r-1), abs(c-1)]
            N_pixel = img[abs(r-1), c]
            NE_pixel = img[abs(r-1), c+1] if c != (img.shape[1]-1) else img[r, c]
            E_pixel = img[r, c+1] if c != (img.shape[1]-1) else img[r, c]
            SE_pixel = img[r+1, c+1] if (r != (img.shape[0]-1) and c != (img.shape[1]-1)) else img[r, c]
            S_pixel = img[r+1, c] if r != (img.shape[0]-1) else img[r, c]
            SW_pixel = img[r+1, abs(c-1)] if r != (img.shape[0]-1) else img[r, c]
            W_pixel = img[r, abs(c-1)]
            
            neighbour_pixels_list = [NW_pixel, N_pixel, NE_pixel, E_pixel, SE_pixel, S_pixel, SW_pixel, W_pixel]
            neighbour_pixels_list = [x for x in neighbour_pixels_list if (x not in (0,255))] #Removing white or black pixels from neighbour pixel list
            #print('Neighbour Pixel List', neighbour_pixels_list) #List of non white/black neighbour pixels

            #Assign new Pixel Label if all neighburhood pixels are white or black
            if len(neighbour_pixels_list) == 0:
                img[r, c] = assigned_label[-1]
                #print(f'assigned new label - {assigned_label[-1]} to {r}, {c}')
                assigned_label.append(assigned_label[-1] + 1)

            #Otherwise Assign minimum label from Neighbour's Pixel list
            else:
                img[r, c] = min(neighbour_pixels_list)
                #print(f'assigned min neighbourhood label - {min(neighbour_pixels_list)} to {r}, {c}')

assigned_label.pop() #Removing last unassigned label
#print('Assigned_label:', assigned_label)
print('Completed 1st Pass - Assigned Labels to All White Pixels')
#cv2.imshow('Birds_Gray_labelled_1', img) #Display Grayscale Labelled Image
#cv2.imwrite('Birds_Gray_labelled_1.jpg', img) #Write Grayscale Labelled Image in a file


#2nd iteration to correct each pixel's labels based on their neighbours
#Change the pixels based on the min label value of their neighbour pixels
#Iterate this process multiple times to get final result
print('Starting 2nd Pass - Re-labelling equivalent labels on Birds')
birds_count_list = [] #initializing birds_count_list
for i in range(1,51):
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):

            if img[r, c] not in (0,255):
                #print(f'Pixel Colour - {img[r, c]}')
                NW_pixel = img[abs(r-1), abs(c-1)]
                N_pixel = img[abs(r-1), c]
                NE_pixel = img[abs(r-1), c+1] if c != (img.shape[1]-1) else img[r, c]
                E_pixel = img[r, c+1] if c != (img.shape[1]-1) else img[r, c]
                SE_pixel = img[r+1, c+1] if (r != (img.shape[0]-1) and c != (img.shape[1]-1)) else img[r, c]
                S_pixel = img[r+1, c] if r != (img.shape[0]-1) else img[r, c]
                SW_pixel = img[r+1, abs(c-1)] if r != (img.shape[0]-1) else img[r, c]
                W_pixel = img[r, abs(c-1)]
                neighbour_pixels_list = [NW_pixel, N_pixel, NE_pixel, E_pixel, SE_pixel, S_pixel, SW_pixel, W_pixel]
                
                #Removing white or black pixels from neighbour pixel list
                neighbour_pixels_list = [x for x in neighbour_pixels_list if (x not in (0,255))]
                #print('Neighbour Pixel List', neighbour_pixels_list)

                if len(neighbour_pixels_list) != 0:
                    img[r, c] = min(neighbour_pixels_list)
                    #print(f'assigned min neighbourhood label - {min(neighbour_pixels_list)} to {r}, {c}')
      
    
    #Count of Birds
    assigned_label_list = []
    for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                assigned_label_list.append(img[r, c])

    assigned_label_list = [x for x in assigned_label_list if x != 0] #Removing black background pixels
    assigned_label_list = list(set(assigned_label_list)) #Removing duplicates from list
    #print('Assigned Label List:', assigned_label_list) #Print assigned_label_list
    
    birds_count = len(set(assigned_label_list)) - 1 #Reducing Pixel Intensity 0 that corresponds to black pixels
    birds_count_list.append(birds_count)
    print(f'Count of Birds after iteration {i}:', birds_count) 

    # Break from the loop if the count of birds is not decreasing
    if len(birds_count_list) >= 2 and birds_count_list[-1] == birds_count_list[-2]:
        print(f'Final count of Birds:', birds_count) 
        break


#cv2.imshow('Birds_Gray_labelled', img) #Display Grayscale Labelled Image
cv2.imwrite('data/' + 'Birds_Gray_labelled.jpg', img) #Save Grayscale Labelled Image

#=====================================================================

# Labelling the original image using the gray_labelled_image
orig_img = cv2.imread('data/' + 'birds.jpg')

keys = assigned_label_list
values = [(238,116,116),(119,97,208),(64,19,243),(31,105,233),(22,6,85),(116,16,119),(119,16,71),(16,119,99),(208,102,57),(208,182,57),(202,246,135),(5,245,181),(46,74,3),(0,0,255),(149, 0, 255),(255, 166, 0),(0,51,25),(255,0,0),(255,255,123),(204,0,0),(25,0,51),(96,96,96),(153,255,153),(204,255,255),(102,102,0),(51,153,255),(0,153,153),(255,204,204),(229,255,204),(255,153,204),(255,204,153),(102,0,51),(145,145,19),(19,141,145),(120,19,145),(117,189,62),(90,63,114),(188,17,85),(33,74,9),(12,2,145),(204,204,0),(0,0,102),(102,204,0),(204,0,204),(255,255,0),(25,51,0),(51,0,25),(0,51,51),(255,0,127),(64,64,64)]
colour_dictionary = dict(zip(keys, values))
#print(colour_dictionary)

for i in assigned_label_list:
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r,c] == i:
                orig_img[r,c] =  colour_dictionary[i]


#Save original labelled image
output_file_name = 'data/' + 'Birds_Colour_labelled.jpg'
cv2.imwrite(output_file_name, orig_img)
print('Output File: ', output_file_name)
