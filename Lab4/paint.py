import matplotlib.pyplot as plt

x = [1, 2, 4, 8, 16, 32]
y = [49055500, 43806250, 44797750, 42093500, 40885250, 40504500]

plt.plot(x, y)
plt.title("stencil")
plt.xlabel("unrolling")
plt.ylabel("simTicks")

plt.show()
