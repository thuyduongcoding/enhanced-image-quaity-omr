from lyrics_remove import remove_lyrics
from process_input_image import enhance_image
import os
import cv2 as cv

input_dir = 'test_image'
remove_lyrics_dir = 'remove_lyrics_image'
upscale_dir = 'upscale_image'
denoise_dir = 'denoise_image'
thresh_dir = 'thresh_image'

for image_path in os.listdir(input_dir):
    demo_input_path = os.path.join(input_dir, image_path)

    # Remove lyrics
    remove_lyrics_image = remove_lyrics(demo_input_path, show_image=False)
    remove_lyrics_path = os.path.join(remove_lyrics_dir, image_path)
    cv.imwrite(remove_lyrics_path, remove_lyrics_image)

    # Upscale, denoise, sharpen and binarize image
    upscale_image_path = os.path.join(upscale_dir, image_path)
    denoise_image_path = os.path.join(denoise_dir, image_path)
    thresh_img = enhance_image(remove_lyrics_path, upscale_image_path, denoise_image_path)

    thresh_image_path = os.path.join(thresh_dir, image_path)
    cv.imwrite(thresh_image_path, thresh_img)