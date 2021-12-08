import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def fp_PCA(data:pd.DataFrame):
    n_comp = 10
    cols = []
    for c in data.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            cols.append(c)
    df = data[['Polymer'] +cols].dropna().drop_duplicates(subset = cols)
    pca = PCA(n_components = n_comp)
    pca.fit(df[cols])
    print('Explained variance (ratio): ' + str(pca.explained_variance_ratio_))
    print('Total variance in ' + str(n_comp) + ' components = ' + str(sum(pca.explained_variance_ratio_)))
    filtered = data.dropna(subset = cols).reset_index(drop = True)
    out = filtered[['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity', 'Polymer']].join(pd.DataFrame(pca.transform(filtered[cols])))
    return out, list(np.linspace(0,10-1,10).astype(int)), pca