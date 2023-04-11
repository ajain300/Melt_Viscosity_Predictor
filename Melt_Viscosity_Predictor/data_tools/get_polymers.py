import pandas as pd
polymers = pd.read_excel('../Melt Viscosity Data.xlsx', ["Homopolymer", "Copolymer", "Polymer Blend"])

SMILES_df = pd.DataFrame(columns = ['Polymer', 'SMILES'])
SMILES_df = pd.concat([SMILES_df, polymers["Homopolymer"][['Polymer','SMILES']]], ignore_index = True)
SMILES_df = pd.concat([SMILES_df, polymers["Copolymer"].rename(columns = {'Polymer 1': 'Polymer', 'SMILES 1': 'SMILES'}).loc[:,['Polymer','SMILES']]], ignore_index = True)
SMILES_df = pd.concat([SMILES_df, polymers["Copolymer"].rename(columns = {'Polymer 2': 'Polymer', 'SMILES 2': 'SMILES'}).loc[:,['Polymer','SMILES']]], ignore_index = True)
SMILES_df = pd.concat([SMILES_df, polymers["Polymer Blend"].rename(columns = {'Polymer 1': 'Polymer', 'SMILES 1': 'SMILES'}).loc[:,['Polymer','SMILES']]], ignore_index = True)
SMILES_df = pd.concat([SMILES_df, polymers["Polymer Blend"].rename(columns = {'Polymer 2': 'Polymer', 'SMILES 2': 'SMILES'}).loc[:,['Polymer','SMILES']]], ignore_index = True)
SMILES_df = SMILES_df.drop_duplicates('Polymer').sort_values('SMILES').reset_index().drop(columns=['index'])

SMILES_df.to_excel('../Polymer-SMILES.xlsx', sheet_name ='Polymers')
