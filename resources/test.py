import matplotlib.pyplot as plt


precisions1 = [0.9, 0.8, 0.8, 0.8, 0.35, 0.4, 0.5, 0.2, 0.25, 0.3, 0.3]
precisions2 = [0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]
recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
recall2 = [0, 0.1, 0.101, 0.2, 0.3, 0.4, 0.401, 0.5, 0.6, 0.7, 0.701, 0.8, 0.9, 1]

name1 = 'p(r)'
name2 = 'p_interp(r)'

plt.figure()
plt.plot(recall, precisions1, label=name1)
plt.plot(recall2, precisions2, label=name2)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(top=1.03)
plt.ylim(bottom=0)
plt.legend(loc="lower left")
plt.title('Precision/Recall curve')

plt.show()
