import matplotlib.pyplot as plt
import numpy as np

# Two images of different shapes
img1 = np.zeros((100, 200))    # shape (height, width)
img2 = np.ones((300, 100))     # shape (height, width)

fig, ax = plt.subplots(1, 2)

# Show the images
ax[0].imshow(img1, cmap='viridis')
ax[1].imshow(img2, cmap='viridis')

# Force same height across both axes
# box_aspect = height / width
ax[0].set_box_aspect(img1.shape[0] / img1.shape[1])
ax[1].set_box_aspect(img2.shape[0] / img2.shape[1])

# Set the figure layout to avoid overlap
plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

img = np.random.rand(100, 200)

fig, axs = plt.subplots(1, 1, figsize=(4, 4))  # Total figure size in inches (W x H)

axs.imshow(img, cmap='viridis')

# Set the axis to have a fixed width-to-height ratio
# For example: force it to be visually taller or shorter
axs.set_box_aspect(1)  # box_aspect = height / width

plt.show()