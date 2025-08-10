import json
import matplotlib.pyplot as plt
import numpy as np


with open('clique_data.json', 'r') as f:
    data = json.load(f)

# Create black and white heatmap showing whether or not a clique is a flower. Put k,i on the x-axis and n on the y-axis.
max_k = 10
max_n = 16

fig, axs = plt.subplots(max_k, 1, figsize=(max_k,max_n*max_k))

for k in range(1, max_k + 1):
    ax = axs[k - 1]
    flower_matrix = np.zeros((max_n - 1, k))
    for i in range(k):
        for n in range(2, max_n + 1):
            if str(k)+","+str(i) not in data or str(n) not in data[str(k)+","+str(i)] or data[str(k)+","+str(i)][str(n)]['flower'] is None:
                flower_matrix[n - 2, i] = 0.5
            elif data[str(k)+","+str(i)][str(n)]['flower']:
                flower_matrix[n - 2, i] = 1
    ax.imshow(flower_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax.set_ylabel(f'n')
    ax.set_yticks(np.arange(0.5, max_n - 0.5))
    ax.set_yticklabels(range(2, max_n + 1), fontsize=12)
    ax.set_xlabel('i')
    ax.set_xticks(np.arange(0.5, k+0.5))
    ax.set_xticklabels(range(k), fontsize=12)
    ax.grid(which='both', color='black', linestyle='-')
    ax.set_title(f'Black and White Heatmap of Flower Cliques for k={k}')

    # # Enable LaTeX formatting
    # plt.rcParams['text.usetex'] = True

    # # Add text in the center of each grid cell
    # for i in range(flower_matrix.shape[0]):
    #     for j in range(flower_matrix.shape[1]):
    #         if flower_matrix[i, j] == 0.5:
    #             continue
    #         else:
    #             if data[str(k)+","+str(j)][str(i+2)]['min_sum'] is not None:
    #                 text = str(data[str(k)+","+str(j)][str(i+2)]['min_sum'])
    #             else:
    #                 text = "?"

    #             if data[str(k)+","+str(j)][str(i+2)]['flower']:
    #                 text += " *"
    #             if not data[str(k)+","+str(j)][str(i+2)]['prime']: 
    #                 text += " +"
    #             if data[str(k)+","+str(j)][str(i+2)]['vector_space']:
    #                 text += " ->"
    #             if data[str(k)+","+str(j)][str(i+2)]['complement']:
    #                 text += " C"
    #             if not data[str(k)+","+str(j)][str(i+2)]['disaster']:
    #                 text += " lb"

    #             # Use LaTeX formatting for different font sizes
    #             text = r'\textbf{' + text + r'}'  # Example: make the text bold
    #             if flower_matrix[i, j] == 1:
    #                 ax.text(j, i, text, ha='center', va='center', color='black', fontsize=20)
    #             else:
    #                 ax.text(j, i, text, ha='center', va='center', color='white', fontsize=20)

    # Add text in the center of each grid cell
    for i in range(flower_matrix.shape[0]):
        for j in range(flower_matrix.shape[1]):
            if flower_matrix[i, j] == 0.5:
                continue
            else:
                if data[str(k)+","+str(j)][str(i+2)]['min_sum'] is not None:
                    text = str(data[str(k)+","+str(j)][str(i+2)]['min_sum'])
                else:
                    text = "?"

                if not data[str(k)+","+str(j)][str(i+2)]['prime']: 
                    text+=" +"
                if data[str(k)+","+str(j)][str(i+2)]['vector_space']:
                    text+=" ->"
                if data[str(k)+","+str(j)][str(i+2)]['complement']:
                    comps = data[str(k)+","+str(j)][str(i+2)]['comps']
                    text+="\nC("+str(comps[1:])+")"
                if not data[str(k)+","+str(j)][str(i+2)]['disaster']:
                    text+=" lb"

                if flower_matrix[i, j] == 1:
                    ax.text(j, i, text, ha='center', va='center', color='black', fontsize=10)
                else:
                    ax.text(j, i, text, ha='center', va='center', color='white', fontsize=10)

fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked.png')


############################################################################################################


# Create black and white heatmap showing whether or not a clique is a flower. Put k,i on the x-axis and n on the y-axis.

max_k = 20
max_n = 16

fig, axs = plt.subplots(max_n, 1, figsize=(max_k,max_n*max_k))

for n in range(2, max_n + 1):
    ax = axs[n - 2]
    flower_matrix = 0.5*np.ones((max_k+1, max_k+1))
    for k in range(1,max_k+1):
        for i in range(k):
            if str(k)+","+str(i) not in data or str(n) not in data[str(k)+","+str(i)] or data[str(k)+","+str(i)][str(n)]['flower'] is None:
                flower_matrix[k,i] = 0.5
            elif data[str(k)+","+str(i)][str(n)]['flower'] == True:
                flower_matrix[k, i] = 1
            elif data[str(k)+","+str(i)][str(n)]['flower'] == False:
                flower_matrix[k, i] = 0
    ax.imshow(flower_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax.set_ylabel('k')
    ax.set_yticks(np.arange(0.5, max_k+0.5))
    ax.set_yticklabels(range(max_k), fontsize=12)
    ax.set_xlabel('i')
    ax.set_xticks(np.arange(0.5, max_k+0.5))
    ax.set_xticklabels(range(max_k), fontsize=12)
    ax.grid(which='both', color='black', linestyle='-')
    ax.set_title(f'Black and White Heatmap of Flower Cliques for n={n}')

    # Add text in the center of each grid cell
    for k in range(1,max_k+1):
        for i in range(k):
            if flower_matrix[k, i] == 0.5:
                continue
            else:
                if data[str(k)+","+str(i)][str(n)]['min_sum'] is not None:
                    text = str(data[str(k)+","+str(i)][str(n)]['min_sum'])
                else:
                    text = "?"

                if not data[str(k)+","+str(i)][str(n)]['prime']: 
                    text+=" +"
                if data[str(k)+","+str(i)][str(n)]['vector_space']:
                    text+=" ->"
                if data[str(k)+","+str(i)][str(n)]['complement']:
                    comps = data[str(k)+","+str(i)][str(n)]['comps']
                    text+="\nC("+str(comps[1:])+")"
                if not data[str(k)+","+str(i)][str(n)]['disaster']:
                    text+=" lb"

                if flower_matrix[k, i] == 1:
                    ax.text(i, k, text, ha='center', va='center', color='black', fontsize=10)
                else:
                    ax.text(i, k, text, ha='center', va='center', color='white', fontsize=10)

fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked_2.png')


############################################################################################################

for k_minus_i in range(1,20):
    for i in range(20):
        k = k_minus_i + i
        for n in range(10):
            
