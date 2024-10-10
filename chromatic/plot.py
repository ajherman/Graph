import json
import matplotlib.pyplot as plt
import numpy as np

with open('clique_data.json', 'r') as f:
    data = json.load(f)

# for key in data:
#     # print(key)
#     k, i = key.split(',')
#     kk = int(k)
#     ii = int(i)
#     for n in data[key]:
#         # print(n)
#         if type(data[key][n])==dict:
#             nn = int(n)
#             if data[key][n]['min_sum'] is None:
#                 data[key][n]['type'] =None
#                 data[key][n]['flower'] = None
#                 data[key][n]['antiflower'] = None
#             elif data[key][n]['min_sum'] == nn*(kk-ii)+ii:
#                 data[key][n]['type'] = 'flower'
#                 data[key][n]['flower'] = True
#             elif data[key][n]['min_sum'] == 2*kk-ii:
#                 data[key][n]['type'] = 'antflower'
#                 data[key][n]['flower'] = False
#                 data[key][n]['antiflower'] = True
#             else:
#                 data[key][n]['type'] = 'misc'
#                 data[key][n]['flower'] = False
#                 data[key][n]['antiflower'] = False

# json.dump(data, open('clique_data.json', 'w'))


# Create black and white heatmap showing whether or not a clique is a flower. Put k,i on the x-axis and n on the y-axis.

max_k = 8
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

    # Add text in the center of each grid cell
    for i in range(flower_matrix.shape[0]):
        for j in range(flower_matrix.shape[1]):
            if flower_matrix[i, j] == 0.5:
                continue
            else:
                if data[str(k)+","+str(j)][str(i+2)]['min_sum'] is not None:
                    min_sum = data[str(k)+","+str(j)][str(i+2)]['min_sum']
                else:
                    min_sum = "?"

                if flower_matrix[i, j] == 1:
                    ax.text(j, i, min_sum, ha='center', va='center', color='black', fontsize=20)
                else:
                    ax.text(j, i, min_sum, ha='center', va='center', color='white', fontsize=20)


# axs[-1].set_xticks(np.arange(0, max_k))
fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked.png')
