import scipy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

def main():
    # ------
    # Use high pass filter to extract the contour of the tag 
    # ------
    cap = cv2.VideoCapture('./Tag1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    '''
    Scale down the video file to make the process faster
    '''
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    '''
    Fast Fourier Transform
    '''
    # Compute the 2-D discrete Fourier Transform
    dft = np.fft.fft2(gray)
    
    # Shift the zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    
    
    # Circular HPF mask, center circle is 0, remaining all ones
    row, col = gray.shape
    center_row, center_col = int(row/2), int(col/2)
    mask = np.ones((row, col), np.uint8)
    # the radius for the circular mask
    r = 100
    center = [center_row, center_col]
    # np.ogrid() acts as np.arrange(), usually is used to create a mask
    x, y = np.ogrid[:row, :col]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
    mask[mask_area] = 0
    
    # Multiple fourier transformed image with mask values
    fshift = dft_shift*mask

    
    # The inverse of `fftshift`
    f_ishift = np.fft.ifftshift(fshift)
    # 2-D inverse discrete Fourier Transform
    # Will be complex numbers
    img_back = np.fft.ifft2(f_ishift)
    # Magnitude spectrum of the image domain
    img_back = np.abs(img_back)

    '''
    Plot the result
    '''
    plt.subplot(121), plt.imshow(frame, cmap = 'gray'), plt.title("Original Image")
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray'), plt.title("High Pass Filter (FFT)")
    plt.show()
    
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
