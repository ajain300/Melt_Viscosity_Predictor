from ast import Mod
from re import A
from xml.sax.handler import feature_external_ges
import tensorflow as tf
from model_versions import *
from Visc_PENN import *
from ViscNN import create_ViscNN_concat
from utils.train_utils import FeatureHeaders
from utils.eval_utils import extrap_test
from data_tools.curve_fitting import fit_Mw, fit_shear, fit_WLF
from scipy.stats import entropy
from typing import List
from sklearn.decomposition import PCA
import numpy as np

TRAINING_DIR = "/data/ayush/Melt_Viscosity_Predictor/Melt_Viscosity_Predictor/MODELS/"
DATASET_DIR = "data_splits"
CONSTANT_DIR = "/data/ayush/Melt_Viscosity_Predictor/Data"
class ModelEval:
    vars_func_dict = {FeatureHeaders.mol_weight.value : fit_Mw, 
        FeatureHeaders.shear_rate.value : fit_shear,
        FeatureHeaders.temp.value : fit_WLF}
    
    vars_label_dict = {FeatureHeaders.mol_weight.value : r'$M_w$ (g/mol)', 
        FeatureHeaders.shear_rate.value : r'$\dot{\gamma}$ (1/s)',
        FeatureHeaders.temp.value : r'$T$ (K)'}

    colors = [plt.get_cmap('Accent')(0.3),plt.get_cmap('Accent')(0.2), plt.get_cmap('Accent')(0.1), plt.get_cmap('Accent')(0.5), plt.get_cmap('Accent')(0.4)]
    
    var_ranges = {Visc_Constants.a1.value : (0.5, 2.5), 
        Visc_Constants.a2.value : (0.5, 4),
        Visc_Constants.Mcr.value : (2, 6),
        Visc_Constants.n.value : (-1, 1),
        Visc_Constants.Scr.value : (-6, 6),
        Visc_Constants.c1.value : (0, 50),
        Visc_Constants.c2.value : (0, 500),
        Visc_Constants.Tr.value : (100, 400),
        }

    def __init__(self, name : str, models : list[BaseModel]):
        
        self.models = models
        os.makedirs(os.path.join(TRAINING_DIR, name), exist_ok=True)
        self.sim_dir = os.path.join(TRAINING_DIR, name)

        self.load_data_split()
        print("Loaded data.")

        self.load_scalers()
        print("Loaded scalers")

        print("Loading trained models.")
        for i in range(len(self.models)):
            self.models[i].set_path(os.path.join(self.sim_dir, "models"))
            self.models[i].load_model()
            self.models[i].set_scalers(self.scalers.copy())
            self.models[i].setup_prediction_data()
            self.models[i].prediction_data['constants'] = []

    def load_data_split(self) -> None:
        self.full_data_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl"))
        self.train_df = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "train_df.pkl"))
        self.test_df = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "test_df.pkl"))
        self.train_df_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "train_df_fp.pkl"))
        self.test_df_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "test_df_fp.pkl"))

    def load_scalers(self) -> None:
        scaler_path = os.path.join(self.sim_dir, "dataset", "scalers.pkl")
        with open(scaler_path, 'rb') as file:
            # Load the dictionary from the file using pickle
            self.scalers = pickle.load(file)
    
    def extrapolation_test(self, test_name, sample_col, const_col, extrap_range_dict : Dict[str, np.ndarray]):
        """
        This function performs extrapolation tests using sample data and saves the results as plots in a
        specified directory.
        
        :param sample_col: `sample_col` 
        :param const_col: The `const_col` parameter in the `extrapolation_test` function is used to
        specify the column(s) in the data that are considered constant or fixed during the extrapolation
        test. These columns are not varied or extrapolated over, but rather held constant while the
        `sample_col` is extrapol
        :param extrap_range: The `extrap_range` parameter is a NumPy array that specifies the range of
        values for extrapolation.
        """
        failed_ann_pred = 0
        extrap_dir = os.path.join(self.sim_dir, "extrap_tests", test_name)
        extrap_range = extrap_range_dict[sample_col]
        os.makedirs(extrap_dir, exist_ok= True)

        sample_data, sample_data_fp, sample_ids = self.get_sample_ids(self.test_df, self.test_df_fp, 
                                                        const_col, const_col)
  
        for samp_id in sample_ids:
            trial_df, tests_df, tests_fp_df = extrap_test(sample_data, sample_data_fp,
                                                        samp_id, 
                                                        extrap_col = sample_col,
                                                        const_col=const_col + [FeatureHeaders.PDI.value],
                                                        extrap_range=extrap_range)
            if sample_col == FeatureHeaders.mol_weight.value:
                tests_df[sample_col] = np.log10(tests_df[sample_col])

            # Find training counterparts of the datapoints to plot
            const_train_idx = (np.isclose(self.train_df.loc[:, const_col], tests_df.iloc[0][const_col], rtol=1e-2)).all(axis=1)
            # if sample_col == FeatureHeaders.temp.value:
            #     samp_train_idx = (self.train_df_fp.loc[:, self.train_df_fp.columns] == tests_fp_df.iloc[0]).all(axis=1)
            # else:
            samp_train_idx = const_train_idx & (self.train_df_fp.loc[:, self.train_df_fp.columns] == tests_fp_df.iloc[0]).all(axis=1)
            samp_train_df = self.train_df[samp_train_idx]

            fig = plt.figure(figsize=(5, 5))
            for model_num, model in enumerate(self.models):
                print(model.name)
                pred_mean, pred_std = model.inference(tests_df, tests_fp_df)
                # try:
                if model.model == Visc_ANN:
                    try:
                        #print(np.log10(extrap_range).reshape(-1,))
                        if sample_col != FeatureHeaders.temp.value:
                            constants, _ = self.vars_func_dict[sample_col](np.log10(extrap_range).reshape(-1,), pred_mean.reshape(-1,))
                        else: 
                            constants, _ = self.vars_func_dict[sample_col](extrap_range.reshape(-1,), pred_mean.reshape(-1,))

                        if Visc_Constants.Scr.value in constants.keys():
                            constants[Visc_Constants.Scr.value] = np.log10(constants[Visc_Constants.Scr.value])

                        self.models[model_num].prediction_data["constants"].append(constants)

                    except RuntimeError as e:
                        print("Unable to fit ANN predictions.")
                        failed_ann_pred += 1
                else:
                    constants = model.predict_constants(tests_df, tests_fp_df)
                    for k,v in constants.items():
                        constants[k] = v
                
                    self.models[model_num].prediction_data["constants"].append(constants)

                if sample_col != FeatureHeaders.temp.value:
                    print("plotting", model.name)
                    plt.plot(np.log10(extrap_range), pred_mean, '--',label = model.name, color = self.colors[model_num])
                    plt.fill_between(np.log10(extrap_range), (pred_mean-pred_std).reshape(-1,) , (pred_mean+pred_std).reshape(-1,), alpha = 0.2, color = self.colors[model_num])
                else:
                    plt.plot(extrap_range, pred_mean, '--',label = model.name, color = self.colors[model_num])
                    plt.fill_between(extrap_range, (pred_mean-pred_std).reshape(-1,) , (pred_mean+pred_std).reshape(-1,), alpha = 0.2, color = self.colors[model_num])
                
                    # Determine the position for the text annotation (e.g., at the last point of the plot)
                    # x_pos = np.log10(extrap_range)[-1]
                    # y_pos = pred_mean[-1]
                    
                    # Format the constants as a string for annotation
                    #constants_str = '\n'.join([f'{k}: {np.mean(v):.2f}' for k, v in constants.items()])
                    
                    # Annotate the plot with the model name and constants
                    # plt.text(x_pos, y_pos, f'{model.name}\n{constants_str}', fontsize=8, verticalalignment='bottom')

                #print("train_df",samp_train_df[[sample_col, FeatureHeaders.visc.value]])
            if sample_col == FeatureHeaders.shear_rate.value:
                samp_train_df.loc[:, sample_col] = np.log10(samp_train_df[sample_col])
                trial_df.loc[:, sample_col] = np.log10(trial_df[sample_col])

            plt.scatter(samp_train_df[sample_col], samp_train_df[FeatureHeaders.visc.value], label = "Train points")
            plt.scatter(trial_df[sample_col], trial_df[FeatureHeaders.visc.value], label = "Test points")
            if sample_col == FeatureHeaders.mol_weight.value:
                plt.xticks(list(np.arange(2,9,2)),[rf'$10^{i}$' for i in list(np.arange(2,9,2))], fontsize = 14)
                plt.title(trial_df[FeatureHeaders.smiles.value][0] + '\n' + r'$\dot{\gamma}$ ' + rf'= {trial_df[FeatureHeaders.shear_rate.value][0]} 1/s' + '\n' + f'Temp =  {trial_df[FeatureHeaders.temp.value][0]} K', 
                          pad=20)
                plt.yticks(list(np.arange(-2,15,4)), [rf'$10^{{{i}}}$' for i in list(np.arange(-2,15,4))], fontsize = 14)
            elif sample_col == FeatureHeaders.shear_rate.value:
                plt.xticks(list(np.arange(-5,6,3)), [rf'$10^{{{i}}}$' for i in list(np.arange(-5,6,3))], fontsize = 14)
                plt.title(trial_df[FeatureHeaders.smiles.value][0] + '\n' + rf'$M_w$ = {np.power(10, trial_df[FeatureHeaders.mol_weight.value][0])} g/mol' + '\n' + f'Temp = {trial_df[FeatureHeaders.temp.value][0]} K', 
                          pad=10)
                plt.yticks(list(np.arange(-2,9,2)), [rf'$10^{{{i}}}$' for i in list(np.arange(-2,9,2))], fontsize = 14)
            elif sample_col == FeatureHeaders.temp.value:
                plt.title(trial_df[FeatureHeaders.smiles.value][0] + '\n' + rf'$M_w$ = {np.power(10, trial_df[FeatureHeaders.mol_weight.value][0])} g/mol' + '\n' + r'$\dot{\gamma}$ ' + rf'= {trial_df[FeatureHeaders.shear_rate.value][0]} 1/s', 
                          pad=10)
                plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                y_ticks = plt.gca().get_yticks()
                y_ticks = [int(tick) for tick in y_ticks if tick.is_integer()]
                plt.yticks(y_ticks, [rf'$10^{{{i}}}$' for i in y_ticks], fontsize = 14)

            
            plt.ylabel(r'$\eta$ (Poise)', fontsize = 12)
            plt.xlabel(self.vars_label_dict[sample_col], fontsize = 12)
            plt.legend()
            plt.tight_layout(pad=2)
            plt.savefig(os.path.join(extrap_dir, f"extrap_id{samp_id}.png"))
            plt.savefig(os.path.join(extrap_dir, f"extrap_id{samp_id}.svg"), dpi = 300)
            plt.close()

            print("FAILED ANN PRED = ", failed_ann_pred)
        # Compile the constant predictions in a list of dicts
        for model_num, model in enumerate(self.models):
            self.models[model_num].prediction_data["constants"] = merge_dict_list(self.models[model_num].prediction_data["constants"])
            print(model_num)
            print(self.models[model_num].prediction_data["constants"])
    
    @staticmethod
    def get_sample_ids(data: pd.DataFrame, fp_data : pd.DataFrame, sort_columns: list,
                   constancy_columns: list, filter_conditions : dict = None):
        """
        Assigns sample IDs based on changes in selected constancy columns while other specified columns remain constant
        under given filter and sort conditions.

        Parameters:
        - data (pd.DataFrame): The DataFrame to process.
        - filter_conditions (dict): Conditions to filter the data. E.g., {'Shear_Rate': 0.0}
        - sort_columns (list): Columns to sort the data by. E.g., ['Polymer', 'Temperature']
        - constancy_columns (list): Columns whose constancy defines a new sample ID.
        - fingerprint_cols_prefix (str): Prefix used to identify fingerprint columns.

        Returns:
        - pd.DataFrame: Processed DataFrame with new SAMPLE_ID and transformations applied.
        - np.array: Unique sample IDs assigned.
        """
        # Filter data based on conditions
        data = data.copy()
        fp_data = fp_data.copy()

        if filter_conditions:
            for key, value in filter_conditions.items():
                data = data[data[key] == value]
        data = data.sort_values(sort_columns)
        fp_data = fp_data.loc[data.index.values].reset_index(drop=True)
        data = data.reset_index(drop = True)

        if data.empty:
            print('No data available after filtering and sorting.')
            return pd.DataFrame(), np.array([])

        # Initialize first row for constancy checks
        constancy_values = {col: data.loc[0, col] for col in constancy_columns}
        fp = fp_data.loc[0]
        id = 1000

        # Assign SAMPLE_ID based on constancy conditions
        for i in data.index:
            if not all(data.loc[i, col] == constancy_values[col] for col in constancy_columns) or not fp_data.loc[i].equals(fp):
                id += 1
                constancy_values = {col: data.loc[i, col] for col in constancy_columns}
                fp = fp_data.loc[i, :]
            data.loc[i, 'SAMPLE_ID'] = id

        sample_ids = data['SAMPLE_ID'].unique()

        return data, fp_data, sample_ids

    def evaluate_models(self):
        for model in self.models:
            print(f"Evaluating {model.name}")
            model.evaluate(self.test_df, self.test_df_fp, eval_type = "eval_models_test_losssurf")

            # Get relavant constants for the test_df
    
    @staticmethod
    def get_ground_truth_constants():
        Mw_data = pd.read_pickle('../Data/Mw_const_data.pickle').to_dict()
        Mw_data[Visc_Constants.a1.value] = Mw_data.pop('a1')
        Mw_data[Visc_Constants.a2.value] = Mw_data.pop('a2')
        Mw_data[Visc_Constants.Mcr.value] = Mw_data.pop('Mcr')

        shear_data = pd.read_pickle('../Data/shear_const_data.pickle').to_dict()

        shear_data[Visc_Constants.Scr.value] = shear_data.pop('tau')

        temp_data = pd.read_pickle('../Data/temp_const_data2.pickle').to_dict()
        temp_data[Visc_Constants.Tr.value] = temp_data.pop('Tr')

        ground_truth_const = {**Mw_data, **shear_data, **temp_data}

        for k,d in ground_truth_const.items():
            ground_truth_const[k] = [v for i,v in d.items() if np.isfinite(v)]

        ground_truth_const[Visc_Constants.Scr.value] = np.log10(np.array(ground_truth_const[Visc_Constants.Scr.value]) / np.array(ground_truth_const['z_shear']))

        return ground_truth_const

    def plot_constant_histograms(self, constant_names, output_file, bins=20, **kwargs):
        """
        Plot histograms of predicted constants for multiple models.
        
        :param models: List of model objects, each with a prediction_data attribute containing constants
        :param true_data: Dictionary of true data for each constant
        :param constant_names: List of names of constants to plot
        :param output_file: File path to save the output figure
        :param bins: Number of bins for histograms (default 20)
        """
        model_names_dict = kwargs.get("model_names_dict", {})

        true_data = self.get_ground_truth_constants()
        
        n_constants = len(constant_names)
        n_models = len(self.models)
        
        fig, axes = plt.subplots(nrows = n_models + 1, ncols = n_constants, figsize=(10, 5), squeeze=False, sharex='col')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_models + 1))
        
        for j, constant in enumerate(constant_names):
            
            bin_range = self.var_ranges[constant]
            bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1)
            # Plot ground truth
            ax = axes[0, j]
            true_values = true_data.get(constant, [])
            # y, _ = np.histogram(true_values, bins=bin_edges, density=True)
            # ax.bar(bin_edges[:-1], y, width=np.diff(bin_edges), alpha=0.5, color=colors[0])
            y, _, patches = ax.hist(true_values, bins=bin_edges, alpha=0.5, color=colors[0], density=True)
            for patch in patches:
                patch.set_height(patch.get_height() / sum(y))
            
            if j == 0:
                ax.set_ylabel('Ground Truth', fontsize=10)
            ax.set_title(f'({chr(65+j)})', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Plot for each model
            for i, model in enumerate(self.models, start=1):
                ax = axes[i, j]
                predicted_values = model.prediction_data['constants'].get(constant, [])
                print(model.name, constant)
                print("pred values",predicted_values)
                if isinstance(predicted_values, np.ndarray):
                    predicted_values = predicted_values.tolist()

                y, _, patches = ax.hist(predicted_values, bins=bin_edges, alpha=0.5, color=colors[i], density=True)
                for patch in patches:
                    patch.set_height(patch.get_height() / sum(y))

                # Calculate KL divergence
                hist_true, _ = np.histogram(true_values, bins=bin_edges, density=True)
                hist_pred, _ = np.histogram(predicted_values, bins=bin_edges, density=True)
                
                if np.sum(hist_true) > 0 and np.sum(hist_pred) > 0:
                    kl_div = entropy(hist_true + 1e-10, hist_pred + 1e-10)
                    ax.text(0.05, 0.95, f'KL: {kl_div:.2f}', transform=ax.transAxes, 
                            ha='left', va='top', fontsize=8)
                
                if j == 0:
                    # Name the columns according to the provided names, otherwise name them from the model name
                    ax.set_ylabel(model.name, fontsize=10)
                    for k,v in model_names_dict.items():
                        if k in model.name:
                            ax.set_ylabel(v, fontsize=10)
                    
                ax.set_ylim(0, 1)
            
            # Set x-label only for bottom row
            axes[-1, j].set_xlabel(constant, fontsize=10)
        
            # Add vertical red dashed line if needed (adjust condition as necessary)
            if constant == Visc_Constants.a1.value:
                for ax_row in axes[:, j]:
                    ax_row.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)

            if constant == Visc_Constants.a2.value:
                for ax_row in axes[:, j]:
                    ax_row.axvline(x=3.4, color='red', linestyle='--', alpha=0.7)
        
            # Set consistent x-axis range for all plots of this constant
            for ax_row in axes[:, j]:
                ax_row.set_xlim(bin_range)
    
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def loss_surface_comparison(self):
        # Extract all parameters from both networks

        ann = self.models[0].models[0] #ANN split 1
        penn = self.models[1].models[1] #PENN split 1
        
        ann_params = [param.cpu().detach().numpy() for param in list(ann.parameters())]# Example: first layer weights
        penn_params = [param.cpu().detach().numpy() for param in list(penn.parameters())]  # Example: first layer weights

        # Get two random directions
        penn_dir1 = get_random_directions(penn_params)
        penn_dir2 = get_random_directions(penn_params)

        ann_dir1 = get_random_directions(ann_params)
        ann_dir2 = get_random_directions(ann_params)

        # Normalize the directions
        penn_dir1 = normalize_direction(penn_dir1, penn_params)
        penn_dir2 = normalize_direction(penn_dir2, penn_params)
        ann_dir1 = normalize_direction(ann_dir1, ann_params)
        ann_dir2 = normalize_direction(ann_dir2, ann_params)

        ann_l1_size = ann.layer_1.weight.shape[0]
        penn_l1_size = penn.mlp.layer_1.weight.shape[0]

        alpha_range = (-100,100)
        beta_range = (-100,100)
        n_points = 50

        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
        betas = np.linspace(beta_range[0], beta_range[1], n_points)
        alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)
        # Create a grid of reduced parameters
        # x1 = np.linspace(min(reduced_params1[:, 0]), max(reduced_params1[:, 0]), 20)
        # y1 = np.linspace(min(reduced_params1[:, 1]), max(reduced_params1[:, 1]), 20)
        # X1, Y1 = np.meshgrid(x1, y1)

        # x2 = np.linspace(min(reduced_params2[:, 0]), max(reduced_params2[:, 0]), 20)
        # y2 = np.linspace(min(reduced_params2[:, 1]), max(reduced_params2[:, 1]), 20)
        # X2, Y2 = np.meshgrid(x2, y2)

        # x1 = np.linspace(-1, 1, 50)
        # y1 = np.linspace(-1, 1, 50)
        # X1, Y1 = np.meshgrid(x1, y1)

        # x2 = np.linspace(-1, 1, 50)
        # y2 = np.linspace(-1, 1, 50)
        # X2, Y2 = np.meshgrid(x2, y2)

        # Compute loss for each point in the grid for both networks
        Z1 = np.zeros_like(alpha_mesh)
        Z2 = np.zeros_like(alpha_mesh)
        for i in range(n_points):
            for j in range(n_points):
                alpha, beta = alpha_mesh[i, j], beta_mesh[i, j]
                # Update model parameters
                for param, orig, d1, d2 in zip(ann.parameters(), ann_params, ann_dir1, ann_dir2):
                    delta = alpha * d1 + beta * d2
                    param.data = torch.tensor(orig).float().to(ann.device) + torch.tensor(delta).float().to(ann.device)
                Z1[i, j] = min(self.models[0].compute_loss_for_single_batch(ann), 100)
        print("ANN loss surface", Z1)
 

        for i in range(n_points):
            for j in range(n_points):
                alpha, beta = alpha_mesh[i, j], beta_mesh[i, j]
                # Update model parameters
                for param, orig, d1, d2 in zip(penn.parameters(), penn_params, penn_dir1, penn_dir2):
                    delta = alpha * d1 + beta * d2
                    param.data = torch.tensor(orig).float().to(penn.device) + torch.tensor(delta).float().to(penn.device)
                Z2[i, j] = min(self.models[1].compute_loss_for_single_batch(penn), 100)
                # print("loss",Z2[i, j])
        print("PENN loss surface", Z2)
        # for i in range(.shape[0]):
        #     for j in range(X1.shape[1]):
        #         # Reconstruct original parameters from reduced coordinates
                
        #         with torch.no_grad():
        #             # list(ann.parameters())[0] = torch.tensor(reconstructed_params1).reshape(ann.layer_1.weight.shape)
        #             ann.layer_1.weight.copy_(torch.tensor(reconstructed_params1).float().reshape(ann.layer_1.weight.shape))
        #             assert torch.allclose(list(ann.parameters())[0], torch.tensor(reconstructed_params1).reshape(ann.layer_1.weight.shape).float().to(ann.device))
        #         Z1[i, j] = self.models[0].compute_loss_for_single_batch(ann)
        # print("ANN loss surface", Z1)
        # for i in range(X2.shape[0]):
        #     for j in range(X2.shape[1]):
        #         # Reconstruct original parameters from reduced coordinates
        #         reconstructed_params2 = pca2.inverse_transform(np.array([X2[i, j], Y2[i, j]]))
        #         reconstructed_params2 = np.tile(reconstructed_params2,(penn_l1_size,1))
        #         with torch.no_grad():
        #             # list(penn.parameters())[0] = torch.tensor(reconstructed_params2).reshape(penn.mlp.layer_1.weight.shape)
        #             penn.mlp.layer_1.weight.copy_(torch.tensor(reconstructed_params2).float().to(penn.device))
        #             assert torch.allclose(list(penn.parameters())[0], torch.tensor(reconstructed_params2).reshape(penn.mlp.layer_1.weight.shape).float().to(penn.device))
                
        #         Z2[i, j] = self.models[1].compute_loss_for_single_batch(penn)

        # Plot the 3D surfaces for both networks
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(alpha_mesh, beta_mesh, Z1, cmap='viridis')
        ax1.set_xlabel(f'Dir 1 (ANN)')
        ax1.set_ylabel(f'Dir 2 (ANN)')
        ax1.set_zlabel('Loss (Net1)')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(alpha_mesh, beta_mesh, Z2, cmap='viridis')
        ax2.set_xlabel(f'Dir 1 (PENN)')
        ax2.set_ylabel(f'Dir 2 (PENN)')
        ax2.set_zlabel('Loss (Net2)')

        plt.savefig(f'/data/ayush/Melt_Viscosity_Predictor/Melt_Viscosity_Predictor/Paper_Figs/loss_sufaces.png')

# Helper functions

def get_random_directions(params):
    random_directions = []
    for param in params:
        random_directions.append(np.random.randn(*param.shape))
    return random_directions

def normalize_direction(direction, weights):
    new_dir = []
    for d, w in zip(direction, weights):
        new_dir.append(d * (w / (np.linalg.norm(d) + 1e-10)))
    return new_dir

def merge_dict_list(dict_list):
    
    def flatten_list(nested_list):
        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(flatten_list(item))
            else:
                flattened.append(item)
        return flattened

    
    result = {}
    
    for d in dict_list:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value.tolist())
            

    for key, value in result.items():
        result[key] = flatten_list(value)

    return result


if __name__ == '__main__':
    
    ModelEval.get_ground_truth_constants()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    split_name = "training_mw_split_1"
    SAMPLE = FeatureHeaders.mol_weight.value
    models = [#GPRModel(name = 'GPR', model_obj= HyperParam_GPR()),
    PENNModel(name= split_name + '_ANN', model_obj=Visc_ANN, device = device, training = False),
    # PENNModel(name = split_name + '_PENN_WLF_Hybrid', model_obj=Visc_PENN_WLF_Hybrid, device = device, training = False),
    PENNModel(name = split_name + '_PENN_WLF_critadj', model_obj=Visc_PENN_WLF, device = device, training = False),]
    # PENNModel(name = split_name + '_PENN_WLF_SP', model_obj=Visc_PENN_WLF_SP, device = device, training = False),
    # PENNModel(name = split_name + '_PENN_Arrhenius', model_obj=Visc_PENN_Arrhenius, device = device, training = False),]
    # PENNModel(name = split_name + '_PENN_Arrhenius_SP', model_obj=Visc_PENN_Arrhenius_SP, device = device, training = False)]
    model_eval = ModelEval(split_name, models)

    const_col_dict = {FeatureHeaders.mol_weight.value : [FeatureHeaders.shear_rate.value, FeatureHeaders.temp.value], 
        FeatureHeaders.shear_rate.value : [FeatureHeaders.mol_weight.value, FeatureHeaders.temp.value],
        FeatureHeaders.temp.value : [FeatureHeaders.mol_weight.value, FeatureHeaders.shear_rate.value]}
    # model_eval.evaluate_models()
    model_eval.loss_surface_comparison()

    # model_eval.extrapolation_test(f"{SAMPLE}_extrap",sample_col = SAMPLE,
    #                                 const_col = const_col_dict[SAMPLE],
    #                                 extrap_range_dict= {FeatureHeaders.mol_weight.value : np.power(10, np.linspace(2, 8)),
    #                                                 FeatureHeaders.shear_rate.value : np.power(10, np.linspace(-5, 5)),
    #                                                 FeatureHeaders.temp.value : np.linspace(300, 600)})

    # var_to_const_dict = {FeatureHeaders.mol_weight.value : [Visc_Constants.a1.value, Visc_Constants.a2.value, Visc_Constants.Mcr.value], 
    #     FeatureHeaders.shear_rate.value : [Visc_Constants.n.value, Visc_Constants.Scr.value],
    #     FeatureHeaders.temp.value : [Visc_Constants.c1.value, Visc_Constants.c2.value, Visc_Constants.Tr.value]}

    # for model in model_eval.models:
    #     model_dir = model_eval.sim_dir
    #     os.makedirs(model_dir, exist_ok=True)
    #     pickle_file = os.path.join(model_dir, f"prediction_params_{model.name}.pickle")
    #     with open(pickle_file, "wb") as f:
    #         pickle.dump(model.prediction_data["constants"], f)

    # model_eval.plot_constant_histograms(constant_names=var_to_const_dict[SAMPLE], 
    #                                             output_file=os.path.join(model_eval.sim_dir, f"constants_{SAMPLE}.png"),
    #                                             model_names_dict = {"ANN": "ANN", "PENN_WLF": "PENN"})
    
