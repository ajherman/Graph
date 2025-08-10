import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


with open('clique_data.json','r') as f:
    data = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})


# Create black and white heatmap showing whether or not a clique is a flower. Put k,i on the x-axis and n on the y-axis.
max_k = 40
max_n = 12

fig, axs = plt.subplots(max_k, 1, figsize=(1.2*max_k,1.2*max_n*max_k))

for k in range(1, max_k + 1):
    ax = axs[k - 1]
    flower_matrix = np.zeros((max_n + 1, k))
    flower_matrix[:2,:] = 1
    for i in range(k):
        for n in range(2,max_n+1):
            try:
                assert(data[k][i][n]['is_flower'] is not None)
                if data[k][i][n]['is_flower']:
                    flower_matrix[n, i] = 1
            except:
                flower_matrix[n, i] = 0.5
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

    # Add text in the center of each grid cell
    for n in range(2,max_n+1):
        for i in range(k):
            if flower_matrix[n, i] == 0.5:
                continue
            else:
                if data[k][i][n]['min_v'] is not None:
                    text = str(data[k][i][n]['min_v'])
                else:
                    text = "?"

                # if not data[k][i][n]['irreducible']: 
                #     text+=" +"

                if data[k][i][n]['vector_space']:
                    text+=" ->"

                for factorization in data[k][i][n]['factorizations']:
                    factorization_str = ""
                    for factor in factorization:
                        coef = factorization[factor]
                        if coef == 1:
                            coef=""
                        factorization_str += str(coef) + str(factor) + "+"
                    text+="\n"+factorization_str[:-1]
           
                if flower_matrix[n, i] == 1:
                    ax.text(i, n-2, text, ha='center', va='center', color='black', fontsize=8)
                else:
                    ax.text(i, n-2, text, ha='center', va='center', color='white', fontsize=8)

fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked_1.png')


############################################################################################################


# Create black and white heatmap showing whether or not a clique is a flower. Put k,i on the x-axis and n on the y-axis.

max_k = 40
max_n = 12

fig, axs = plt.subplots(max_n, 1, figsize=(max_k,max_n*max_k))

for n in range(2, max_n + 1):
    ax = axs[n - 2]
    flower_matrix = 0.5*np.ones((max_k+1, max_k+1))
    for k in range(1,max_k+1):
        for i in range(k):
            try:
                assert(data[k][i][n]['is_flower'] is not None)
                if data[k][i][n]['is_flower']:
                    flower_matrix[k, i] = 1
                elif data[k][i][n]['is_flower'] == False:
                    flower_matrix[k, i] = 0
            except:
                flower_matrix[k, i] = 0.5
            
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
                if data[k][i][n]['min_v'] is not None:
                    text = str(data[k][i][n]['min_v'])
                else:
                    text = "?"

                if not data[k][i][n]['irreducible']: 
                    text+=" +"
                if data[k][i][n]['vector_space']:
                    text+=" ->"

                if flower_matrix[k, i] == 1:
                    ax.text(i, k, text, ha='center', va='center', color='black', fontsize=10)
                else:
                    ax.text(i, k, text, ha='center', va='center', color='white', fontsize=10)

fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked_2.png')


# ############################################################################################################

max_k_minus_i = 20
max_i = 40
# max_k = 45
max_n = 12

fig, axs = plt.subplots(max_k_minus_i, 1, figsize=(0.8*max_i,0.8*max_n*max_k_minus_i))

for k_minus_i in range(max_k_minus_i):
    ax = axs[k_minus_i]
    flower_matrix = 0.5*np.ones((max_n+1, max_i+1))
    for i in range(max_i+1):
        k = i + k_minus_i
        for n in range(max_n+1):
            if n == 0:
                flower_matrix[n, i] = 1
            else:
                try:
                    assert(data[k][i][n]['is_flower'] is not None)
                    if data[k][i][n]['is_flower']:
                        flower_matrix[n, i] = 1
                    elif data[k][i][n]['is_flower'] == False:
                        flower_matrix[n, i] = 0
                except:
                    flower_matrix[n, i] = 0.5
            
    ax.imshow(flower_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax.set_ylabel('n')
    ax.set_yticks(np.arange(0.5, max_n+0.5))
    ax.set_yticklabels(range(max_n), fontsize=15)
    ax.set_xlabel('i')
    ax.set_xticks(np.arange(0.5, max_i+0.5))
    ax.set_xticklabels(range(max_i), fontsize=15)
    ax.grid(which='both', color='black', linestyle='-')
    ax.set_title(f'Black and White Heatmap of Flower Cliques for k-i={k-i}', fontsize=20)

    # Add text in the center of each grid cell
    F = 2000 #max_k+(n-1)*k_minus_i
    rand_perm = np.random.permutation(F)
    for i in range(max_i+1):
        k = i + k_minus_i
        for n in range(max_n+1):
            if n == 0:
                ax.text(i, n, f"i={i}\nk={k}", ha='center', va='center', color='black', fontsize=14)
            else:
                if flower_matrix[n, i] == 0.5:
                    continue
                else:
                    if data[k][i][n]['min_v'] is not None:
                            min_v = data[k][i][n]['min_v']
                            text = str(min_v)
                    else:
                            text = "?"

                    # if k>1 and (i==0 or data[k-1][i-1][n]['min_v']+1!=min_v):
                    if i==0 or k==1 or data[k-1][i-1][n]['min_v']+1!=min_v:
                        MAX,MIN = max(k,min_v-k),min(k,min_v-k)
                        cn = MAX*(MAX+1)//2+MIN
                        
                        # Set to a unique color based on the value of cn which can be any non-negative integer
                        cmap = plt.get_cmap('tab20')
                        cmap = matplotlib.colors.ListedColormap(plt.get_cmap('gist_ncar')(np.linspace(0,1,F)))
                        # text_color = plt.cm.viridis(cn/((F+1)*(F+2)//2))
                        text_color = cmap(rand_perm[cn%F])
                    elif flower_matrix[n, i] == 1:
                        text_color = 'black'
                    elif flower_matrix[n, i] == 0:
                        text_color = 'white'
                    else:
                        raise ValueError("Invalid color")

                    if data[k][i][n]['irreducible']: 
                            text+=" *"
                    if data[k][i][n]['vector_space']:
                            text+=" ->"

                    # for factorization in data[k][i][n]['factorizations']:
                    #     factorization_str = ""
                    #     for factor in factorization:
                    #         coef = factorization[factor]
                    #         if coef == 1:
                    #             coef=""
                    #         factorization_str += str(coef) + str(factor) + "+"
                    #     text+="\n"+factorization_str[:-1]

                    ax.text(i, n, text, ha='center', va='center', color=text_color, fontsize=12)

fig.suptitle('Black and White Heatmap of Flower Cliques')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flower_array_stacked_3.png')