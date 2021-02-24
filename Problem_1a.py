import scipy
import cv2
import numpy as np
from matplotlib import pyplot as plt

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    cap = cv2.VideoCapture('/home/bernard/ENPM673/Project_1/Tag1.mp4')
    cap.set(cv2.CAP_PROP_FRAME_COUNT, 1)
    ret, frame = cap.read()
    
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
    r, c = gray.shape
    row, col = int(r/2), int(c/2)
    mask = np.ones((r,c))
    r = 480
    center = [row, col]
    for x in range(c):
        for y in range(r):
            if dist((y,x), center) < r:
                mask[y, x] = 0
    
    # Multiple fourier transformed image with mask values
    fshift = dft_shift*mask

    
    # The inverse of `fftshift`
    f_ishift = np.fft.ifftshift(fshift)
    # 2-D inverse discrete Fourier Transform
    # Will be complex numbers
    img_back = np.fft.ifft2(f_ishift)
    # Magnitude spectrum of the image domain
    img_back = np.abs(img_back)
    
    plt.subplot(121), plt.imshow(frame, cmap = 'gray'), plt.title("Original Image")
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray'), plt.title("a")
    plt.show()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
