import matplotlib.pyplot as plt
import numpy as np

# lambda
lambdas = ['λ = 0.5', 'λ = 0.8', 'λ = 1.3']

# värdet år 2100 för varje case
case_1 = [1.86297, 2.23270, 2.64680]
case_2 = [2.74276, 3.23911, 3.77840]
case_3 = [3.22936, 3.79484, 4.40229]

x = np.arange(len(lambdas))  # [0, 1, 2]
width = 0.25  # stapelns bredd

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, case_1, width, label='Case 1', color='blue')
bars2 = ax.bar(x,         case_2, width, label='Case 2', color='red')
bars3 = ax.bar(x + width, case_3, width, label='Case 3', color='green')


ax.set_xlabel('Lambda')
ax.set_ylabel('Värde')
ax.set_title('Resultat per lambda och case')
ax.set_xticks(x)
ax.set_xticklabels(lambdas)
ax.legend()

plt.tight_layout()
plt.show()