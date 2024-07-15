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

TRAINING_DIR = "/data/ayush/Melt_Viscosity_Predictor/Melt_Viscosity_Predictor/MODELS/"
DATASET_DIR = "data_splits"
CONSTANT_DIR = "/data/ayush/Melt_Viscosity_Predictor/Data"
class ModelEval:
    vars_func_dict = {FeatureHeaders.mol_weight.value : fit_Mw, 
        FeatureHeaders.shear_rate.value : fit_shear,
        FeatureHeaders.temp.value : fit_WLF,
        FeatureHeaders.PDI.value : 4}
    
    colors = [plt.get_cmap('Accent')(0.3),plt.get_cmap('Accent')(0.2), plt.get_cmap('Accent')(0.1), plt.get_cmap('Accent')(0.5), plt.get_cmap('Accent')(0.4)]
    
    var_ranges = {Visc_Constants.a1.value : (0.5, 2.5), 
        Visc_Constants.a2.value : (1, 4),
        Visc_Constants.Mcr.value : (2, 6),
        Visc_Constants.n.value : (-3, 3),
        Visc_Constants.Scr.value : (-6, 6),
        Visc_Constants.c1.value : (0, 50),
        Visc_Constants.c2.value : (0, 500)}

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
    
    def extrapolation_test(self, test_name, sample_col, const_col, extrap_range : np.ndarray):
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
        extrap_dir = os.path.join(self.sim_dir, "extrap_tests", test_name)
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
 
            fig = plt.figure()
            for model_num, model in enumerate(self.models):
                print(model.name)
                pred_mean, pred_std = model.inference(tests_df, tests_fp_df)
                try:
                    if model.model == Visc_ANN:
                        if sample_col != FeatureHeaders.temp.value:
                            constants, _ = self.vars_func_dict[sample_col](np.log10(extrap_range).reshape(-1,), pred_mean.reshape(-1,))
                        else: 
                            constants, _ = self.vars_func_dict[sample_col](extrap_range.reshape(-1,), pred_mean.reshape(-1,))
                    else:
                        constants = model.predict_constants(tests_df, tests_fp_df)
                        for k,v in constants.items():
                            constants[k] = v['mean']
                    
                    self.models[model_num].prediction_data["constants"].append(constants)
                except Exception as e:
                    print(f"Params not found for {model.name}", samp_id)
                    print(e)
                
                if sample_col != FeatureHeaders.temp.value:
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


            # Find training counterparts of the datapoints to plot
            samp_train_idx = (self.train_df_fp.loc[:, self.train_df_fp.columns] == tests_fp_df.iloc[0]).all(axis =1)
            samp_train_df = self.train_df[samp_train_idx]

            #print("train_df",samp_train_df[[sample_col, FeatureHeaders.visc.value]])

            if sample_col == FeatureHeaders.shear_rate.value:
                samp_train_df.loc[:, sample_col] = np.log10(samp_train_df[sample_col])
                trial_df.loc[:, sample_col] = np.log10(trial_df[sample_col])

            plt.scatter(samp_train_df[sample_col], samp_train_df[FeatureHeaders.visc.value], label = "Train points")
            plt.scatter(trial_df[sample_col], trial_df[FeatureHeaders.visc.value], label = "Test points")
            plt.legend()
            plt.savefig(os.path.join(extrap_dir, f"extrap_id{samp_id}.png"))
            plt.close()

        # Compile the constant predictions in a list of dicts
        for model_num, model in enumerate(self.models):
            self.models[model_num].prediction_data["constants"] = merge_dict_list(self.models[model_num].prediction_data["constants"])
    
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
            model.evaluate(self.test_df, self.test_df_fp, eval_type = "eval_models_test")

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

        ground_truth_const = {**Mw_data, **shear_data, **temp_data}

        for k,d in ground_truth_const.items():
            ground_truth_const[k] = [v for i,v in d.items() if np.isfinite(v)]

        return ground_truth_const

    def plot_constant_histograms(self, constant_names, output_file, bins=20):
        """
        Plot histograms of predicted constants for multiple models.
        
        :param models: List of model objects, each with a prediction_data attribute containing constants
        :param true_data: Dictionary of true data for each constant
        :param constant_names: List of names of constants to plot
        :param output_file: File path to save the output figure
        :param bins: Number of bins for histograms (default 20)
        """

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
                    ax.set_ylabel(model.name, fontsize=10)
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

# Helper functions

def merge_dict_list(dict_list):
    
    def merge_lists(input_list):
        # Check if the input is a list of lists
        if isinstance(input_list, list) and all(isinstance(i, list) for i in input_list):
            # Merge the list of lists into a single list
            merged_list = [item for sublist in input_list for item in sublist]
            return merged_list
        else:
            # Return the input as is if it's not a list of lists
            return input_list
    
    result = {}
    
    for d in dict_list:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value.tolist())
            

    for key, value in result.items():
        result[key] = merge_lists(value)

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


    split_name = "training_shear_split_1"
    models = [#GPRModel(name = 'GPR', model_obj= HyperParam_GPR()),
    #PENNModel(name= split_name + '_ANN', model_obj=Visc_ANN, device = device, training = False),
    # PENNModel(name = split_name + '_PENN_WLF_Hybrid', model_obj=Visc_PENN_WLF_Hybrid, device = device, training = False),
    PENNModel(name = split_name + '_PENN_WLF', model_obj=Visc_PENN_WLF, device = device, training = False),
    # PENNModel(name = split_name + '_PENN_WLF_SP', model_obj=Visc_PENN_WLF_SP, device = device, training = False),
    PENNModel(name = split_name + '_PENN_Arrhenius', model_obj=Visc_PENN_Arrhenius, device = device, training = False),]
    # PENNModel(name = split_name + '_PENN_Arrhenius_SP', model_obj=Visc_PENN_Arrhenius_SP, device = device, training = False)]
    model_eval = ModelEval(split_name, models)

    # model_eval.evaluate_models()

    model_eval.extrapolation_test("Mw_extrap",sample_col = FeatureHeaders.shear_rate.value,
                                    const_col = [FeatureHeaders.mol_weight.value, FeatureHeaders.temp.value],
                                    extrap_range= np.power(10, np.linspace(-3, 20))) #np.linspace(300, 600)) #

    model_eval.plot_constant_histograms(constant_names=[Visc_Constants.c1.value, Visc_Constants.c2.value, Visc_Constants.Tr.value], 
                                                output_file=os.path.join(model_eval.sim_dir, "constants.png"))
    
