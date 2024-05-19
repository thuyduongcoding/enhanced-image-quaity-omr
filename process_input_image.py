import cv2 as cv
import numpy as np
import subprocess

"""
Upscale image using vulkan
"""
def upscale_image(input_path, upscale_image_path, denoise_image_path):
	image = cv.imread(input_path,0)

	current_dir = 'C:/Users/nguye/OneDrive/Documents/Braille Sheet App'
	try:
		h, w = image.shape
		
		# Finding the appropriate scale factor
		scale = int(np.sqrt(20000000/(h*w)))
		if scale >= 4:
			scale = 4
		else:
			scale = 2

		# Upscaling
		vulkan_path = f"{current_dir}/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan-20220728-windows/waifu2x-ncnn-vulkan.exe"
		command = f"{vulkan_path} -v -i {input_path} -o {upscale_image_path} -n 2 -s {scale} -f png"
		print('Upscaling...')
		print(command)
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		
		for line in process.stdout:
			print(line)
		
		print('Done upscaling')
		
		# Denoising
		command = f"{vulkan_path} -v -i {upscale_image_path} -o {denoise_image_path} -n 2 -s 1 -f png"
		print('Denoising...')
		print(command)
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		
		for line in process.stdout:
			print(line)
		
		print('Done denoising')
			
	except subprocess.CalledProcessError as e:
		print("Error invoking ncnn-vulkan.")
		print(f"Return code: {e.returncode}")
		print(f"Output: {e.output}")

def sharpen_image(input_image):
	if isinstance(input_image, str):
		image = cv.imread(input_image, 0)  # Read image in grayscale mode
	else:
		image = input_image

	blur = cv.GaussianBlur(image,(3,3),0)

	kernel = np.array([[0, -1, 0],
					   [-1, 5, -1],
					   [0, -1, 0]])
	sharpened = cv.filter2D(blur, -1, kernel)

	return sharpened

def binarize(input_image):
	if isinstance(input_image, str):
		image = cv.imread(input_image, 0)  # Read image in grayscale mode
	else:
		image = input_image

	# Otsu's thresholding after Gaussian filtering
	blur = cv.GaussianBlur(image,(3,3),0)
	_,thresh_img = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

	return thresh_img


def enhance_image(input_path, upscale_image_path, denoise_image_path):
	# Upscale and denoise image
	upscale_image(input_path, upscale_image_path, denoise_image_path)

	# Sharpen image
	sharpened = sharpen_image(upscale_image_path)

	# Binarize image
	thresh_img = binarize(sharpened)

	return thresh_img
