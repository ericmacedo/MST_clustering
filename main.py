import numpy as np
from sklearn.metrics import adjusted_rand_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from kmeans import KMeans
from kruskal import Kruskal
from prim import Prim

K = 7

input_data = open("data/data.txt", "r")
data = np.loadtxt(input_data)

input_classes = open("data/classes.txt", "r")
classes = np.loadtxt(input_classes)

methods = dict({
    "Kruskal": Kruskal(data, K),
    "Prim": Prim(data, K),
    "KMeans": KMeans(data, K)
})

# PRINT READABLE OUTPUT
table_headers = [
    "Method",
    "Silhouette Coefficient",
    "Rand Index",
    "Delta Time"
]
output = []
for key, value in methods.items():
    output.append([
        key,
        value.silhouette,
        adjusted_rand_score(classes, value.classes),
        value.time
    ])

    # # PLOT GRAPH
    plt.figure(key)
    plt.title(key)
    plt.scatter(data[..., 0], data[..., 1], c=value.classes)

plt.figure("(Gionis, 2007)")
plt.title("(Gionis, 2007)")
plt.scatter(data[..., 0], data[..., 1], c=classes)

print("")
print(tabulate(output, table_headers))
print("")
plt.show()

input_data.close()
input_classes.close()
