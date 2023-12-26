import numpy as np
import matplotlib.pyplot as plt

# Define the complex function
def complex_function(z):
    return np.exp(z)

# Create a grid of complex numbers
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Evaluate the function on the grid
output = complex_function(Z)

# Visualize the real and imaginary parts
plt.figure(figsize=(12, 6))

# Real part
plt.subplot(1, 2, 1)
plt.imshow(np.real(output), extent=[x.min(), x.max(), y.min(), y.max()])
plt.title('Real Part')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.colorbar()

# Imaginary part
plt.subplot(1, 2, 2)
plt.imshow(np.imag(output), extent=[x.min(), x.max(), y.min(), y.max()])
plt.title('Imaginary Part')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.colorbar()

plt.tight_layout()
plt.show()
