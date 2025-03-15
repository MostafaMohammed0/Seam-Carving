import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    
    # Compute gradients manually
    dx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    dy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    
    energy = dx + dy
    return energy

def find_seam(energy):
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros_like(cost, dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            left = cost[i - 1, j - 1] if j > 0 else float('inf')
            up = cost[i - 1, j]
            right = cost[i - 1, j + 1] if j < w - 1 else float('inf')
            
            min_index = np.argmin([left, up, right]) - 1  # -1, 0, or 1
            cost[i, j] += [left, up, right][min_index + 1]
            backtrack[i, j] = j + min_index
    
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    
    return seam

def remove_seam(image, seam):
    h, w, _ = image.shape
    output = np.zeros((h, w - 1, 3), dtype=np.uint8)
    
    for i in range(h):
        output[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    
    return output

def seam_carving(image, num_seams):
    vis_image = image.copy()
    for _ in range(num_seams):
        energy = compute_energy(image)
        seam = find_seam(energy)
        
        for i in range(len(seam)):
            vis_image[i, seam[i]] = (0, 0, 255)  # Mark seam in red
        
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
