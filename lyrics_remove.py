import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Display the image
"""
def show_img(img, cmap=None):
  plt.figure(figsize=(16, 12));
  if cmap is None:
    plt.imshow(img)
  else:
    plt.imshow(img, cmap=cmap)

"""
Read the image
"""
def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

"""
Resize the image so that minimum heigh is 1000 and maximum weight is 1000
"""

def resize_img(img, tgt_size=1000):
  h, w, c = img.shape
  if h < tgt_size:
      new_h = tgt_size
      ar = w/h
      new_w = int(ar*new_h)
      img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)    
  else:
      if w > tgt_size:
          new_w = tgt_size
          ar = w/h
          new_h = int(new_w/ar)
          img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

  return img

"""
Binarize the image
"""
def threshold(image, max_thresh=200):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, max_thresh, 255,cv2.THRESH_BINARY_INV)
    return thresh

"""
Dilate the image to find the text region
"""
def dilate_img(thresh_img, kernel_size=(2, 15)):
  kernel = np.ones(kernel_size, np.uint8)
  dilated_img = cv2.dilate(thresh_img, kernel, iterations = 1)
  return dilated_img

"""
Get bounding boxes of the lyrics
"""

def get_bounding_boxes(dilated_img, min_h = 7, max_h = 40):
  (contours, _) = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])

  bounding_box_lines = []
  total_h = []
    
  for ctr in sorted_contours_lines:
    x,y,w,h = cv2.boundingRect(ctr)

    if h < min_h or h > max_h:
      continue
    total_h.append(h)

  if len(total_h) != 0:
      h_lower = 0.5*np.percentile(total_h, 50)
      h_higher = min(3*np.percentile(total_h, 50), 40)  
      for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
    
        if h < h_lower or h > h_higher:
          continue
        bounding_box_lines.append([x, y, x+w, h+y])
        
  return bounding_box_lines


# Draw bounding boxes lines
def draw_lines(img, bounding_box_lines, line_color=(40, 100, 250), line_width=2):
  if len(bounding_box_lines) != 0:    
      img2 = img.copy()
      for x_l, y_l, x_r, y_r in bounding_box_lines:
        cv2.rectangle(img2, (x_l,y_l), (x_r, y_r), line_color, line_width)
      return img2
  else:
      return img


# Remove the texts in the bounding boxes
def turn_white(bounding_box_lines, img):
    if len(bounding_box_lines) != 0:  
        removed_lyrics_img = img.copy()
        for x_l, y_l, x_r, y_r in bounding_box_lines:
            removed_lyrics_img[y_l:y_r, x_l:x_r] = 255
        return removed_lyrics_img
    else:
        return img
    

# Remove lyrics
def remove_lyrics(input_path, show_image=True):
    accept_format = ['png', 'jpeg', 'jpg','PNG']
    if input_path.split('.')[-1] in accept_format:
        # Read the image 
        img = read_img(input_path)
        
        # Resize image
        img = resize_img(img)
        
        # Binarization
        thresh_img = threshold(img);
        
        # Dilate
        dilated_img = dilate_img(thresh_img);
        
        # Line segmentation
        bounding_box_lines = get_bounding_boxes(dilated_img)
        img2 = draw_lines(img, bounding_box_lines)
        
        # Remove lyrics
        removed_lyrics_image = turn_white(bounding_box_lines, img)

        if show_image:
            show_img(img)
            show_img(thresh_img, cmap='gray')
            show_img(dilated_img, cmap='gray')
            show_img(img2)
            show_img(removed_lyrics_image)
    else:
       print("The image format is not available")

    return removed_lyrics_image
