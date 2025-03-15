import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    energy = np.abs(dx) + np.abs(dy)
    return energy

def find_seam(energy):
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        left = np.roll(cost[i - 1], shift=1, axis=0)
        right = np.roll(cost[i - 1], shift=-1, axis=0)
        
        left[0], right[-1] = float('inf'), float('inf') 
        
        min_cost = np.minimum(left, np.minimum(cost[i - 1], right))
        backtrack[i] = np.argmin([left, cost[i - 1], right], axis=0) - 1  
        cost[i] += min_cost

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

def seam_carving(image, num_seams):
    vis_image = image.copy()

    for _ in range(num_seams):
        energy = compute_energy(image)
        seam = find_seam(energy)
        
        vis_image[np.arange(len(seam)), seam] = (0, 0, 255) 
        image = remove_seam(image, seam)

    return image, vis_image

image = cv2.imread('input.jpg')
num_seams = image.shape[1] // 2  


resized_image, seam_visualization = seam_carving(image, num_seams)

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
