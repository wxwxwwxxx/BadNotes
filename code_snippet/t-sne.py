import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x = np.random.randn(100, 1024)
y = np.random.randint(0, 2, 100)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(x)
plt.figure(figsize=(6, 4))
for label in np.unique(y):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=label, marker=f"${label}$", s=20)
plt.show()