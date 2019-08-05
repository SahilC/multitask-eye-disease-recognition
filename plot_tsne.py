import numpy as np
import matplotlib.pyplot as plt

z = np.load('tsne.npy')
# x = z[4:,0]
# y = z[4:, 1]

plt.axis([-1, 1, -1, 1])
x = []
y = []
key = {4: 'retinal', 5:
        'macroaneurysm', 6: 'cystoid', 7: 'macular', 8: 'edema', 9:
        'nonexudative', 10: 'senile', 11: 'degeneration', 14: 'type', 15: '2', 16: 'diabetes', 17: 'mellitus',19: 'retinopathy', 20: 'choroidal', 21: 'neovascular', 22:
        'membrane', 23: 'hemorrhage', 24: 'exudative', 25: 'age-related', 31: 'non-proliferative', 32: 'diabetic', 33: 'glaucoma', 34: 'suspect',
        35: 'inactive', 39:'atrophic', 40: 'subfoveal', 41: 'involvement', 42: 'mild', 43:
        'nonproliferative', 44: 'associated', 45: 'drusen', 46: 'macula', 48: 'detachment', 60: 'tension', 63: 'chronic', 64: 'angle-closure', 68: 'low', 76:'narrow', 77: 'angle', 78: 'pseudoexfoliation', 79: 'uveitic', 80:
        'recession', 85: 'inflammations', 88: 'high', 89: 'cotton', 90: 'wool', 91: 'spots', 92: 'degenerative', 93:
        'malignant', 94: 'melanoma', 95: 'intermediate',  98: 'ophthalmic', 99: 'manifestations', 100: 'uncontrolled', 105: 'pigment', 106:
        'epithelium', 107: 'hypertrophy', 108: 'underlying', 109: 'condition',
        113: 'choroid', 118: 'presence', 119: 'uvea', 120: 'drug', 121:
        'chemical', 122: 'induced', 125: 'uveal', 126: 'anterior',
        127:'subretinal', 134: 'atrophy', 135: 'iris', 136:
        'oculopathy', 137: 'resolved', 140:'posterior', 141: 'cataract', 142: 'dm', 146: 'juvenile', 147: 'central', 148: 'geographic', 149:'hemorrhagic', 152: 'combined', 153:
        'rhegmatogenous', 154: 'clinically', 155: 'significant', 156:'insulin',
        157: 'involving', 161: 'epitheliopathy', 162: 'quiescent', 165:
        'optic', 166: 'papillopathy', 167: 'exudates', 172: 'detachments', 173:
        'maculae', 175: 'traumatic', 
        179: 'syndrome', 181: 'inflammation', 183: 'disorders', 184: 'increased',185: 'pressure', 187: 'closed-angle'
        }
nam = []
for i in key.keys():
    nam.append(key[i])
    x.append(z[i,0])
    y.append(z[i, 1])

x = np.array(x)
y = np.array(y)

print(x,y)
plt.scatter(x,y)
for i in range(len(key.keys())):
    plt.annotate(nam[i], (x[i], y[i]))

plt.savefig('abc1.png')
