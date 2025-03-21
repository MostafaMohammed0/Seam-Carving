import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Define Sobel filter manually
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    dx = cv2.filter2D(gray, -1, kernel_x)
    dy = cv2.filter2D(gray, -1, kernel_y)
    
    energy = np.abs(dx) + np.abs(dy)
    return energy

def find_seam(energy):
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            left = cost[i - 1, j - 1] if j > 0 else float('inf')
            middle = cost[i - 1, j]
            right = cost[i - 1, j + 1] if j < w - 1 else float('inf')
            
            min_idx = np.argmin([left, middle, right]) - 1
            cost[i, j] += min([left, middle, right])
            backtrack[i, j] = min_idx
    
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + backtrack[i + 1, seam[i + 1]]
    
    return seam

def remove_seam(image, seam):
    h, w, _ = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return image[mask].reshape((h, w - 1, 3))

def seam_carving(image, num_seams, direction='vertical'):
    vis_image = image.copy()
    
    if direction == 'horizontal':
        image = np.transpose(image, (1, 0, 2))
        vis_image = np.transpose(vis_image, (1, 0, 2))
    
    for _ in range(num_seams):
        energy = compute_energy(image)
        seam = find_seam(energy)
        
        vis_image[np.arange(len(seam)), seam] = (0, 0, 255)
        image = remove_seam(image, seam)
    
    if direction == 'horizontal':
        image = np.transpose(image, (1, 0, 2))
        vis_image = np.transpose(vis_image, (1, 0, 2))
    
    return image, vis_image

image = cv2.imread('input.jpg')
num_seams = image.shape[1] // 2  # Reduce width by half


resized_image, seam_visualization = seam_carving(image, num_seams, direction='vertical')
cv2.imwrite('resized_image.jpg', resized_image)
cv2.imwrite('seam_visualization.jpg', seam_visualization)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(seam_visualization, cv2.COLOR_BGR2RGB))
plt.title('Seam Visualization')
plt.axis('off')

plt.show()