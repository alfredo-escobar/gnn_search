import matplotlib.pyplot as plt


x = [5,7,3,2,5,8,4,6]
y = [5,7,3,2,5,8,4,6]

fig, ax = plt.subplots()
ax.plot(y)
ax.legend()

plt.show()