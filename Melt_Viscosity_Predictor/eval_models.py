from re import A
import tensorflow as tf
from model_versions import *
from Visc_PENN import *
from ViscNN import create_ViscNN_concat
from utils.train_utils import FeatureHeaders
from utils.eval_utils import extrap_test


TRAINING_DIR = "/data/ayush/Melt_Viscosity_Predictor/Melt_Viscosity_Predictor/MODELS/"
DATASET_DIR = "data_splits"

class ModelEval:
    colors = [plt.get_cmap('Accent')(0.3),plt.get_cmap('Accent')(0.2), plt.get_cmap('Accent')(0.1), plt.get_cmap('Accent')(0.5), plt.get_cmap('Accent')(0.4)]
    
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
            self.models[i].set_scalers(self.scalers)

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
        values for extrapolation. It is used in the `extrapolation_test` method to define the range over
        which the extrapolation will be performed for the given sample data. The method will iterate over
        the values in
        :type extrap_range: np.ndarray
        """
        extrap_dir = os.path.join(self.sim_dir, "extrap_tests", test_name)
        os.makedirs(extrap_dir, exist_ok= True)

        sample_data, sample_data_fp, sample_ids = self.get_sample_ids(self.test_df, self.test_df_fp, 
                                                        const_col, const_col)
        print("sample_data", sample_data)


        for samp_id in sample_ids:
            trial_df, tests_df, tests_fp_df = extrap_test(sample_data, sample_data_fp,
                                                        samp_id, 
                                                        extrap_col = sample_col,
                                                        const_col=const_col + [FeatureHeaders.PDI.value],
                                                        extrap_range=extrap_range)
            
            print(f"testsdf sample_id {samp_id}", tests_df)
            print(f"trialdf sample_id {samp_id}", trial_df)

            fig = plt.figure()
            for model_num, model in enumerate(self.models):
                pred_mean, pred_std = model.inference(tests_df, tests_fp_df)
                plt.plot(np.log10(extrap_range), pred_mean, '--',label = model.name, color = self.colors[model_num])
                plt.fill_between(np.log10(extrap_range), (pred_mean-pred_std).reshape(-1,) , (pred_mean+pred_std).reshape(-1,), alpha = 0.2, color = self.colors[model_num])
            
            # Find training counterparts of the datapoints to plot
            samp_train_idx = (self.train_df_fp.loc[:, self.train_df_fp.columns] == tests_fp_df.iloc[0]).all(axis =1)
            samp_train_df = self.train_df[samp_train_idx]

            plt.scatter(np.log10(samp_train_df[FeatureHeaders.shear_rate.value]), samp_train_df[FeatureHeaders.visc.value], label = "Train points")
            plt.scatter(np.log10(trial_df[FeatureHeaders.shear_rate.value]), trial_df[FeatureHeaders.visc.value], label = "Test points")
            
            plt.legend()
            plt.savefig(os.path.join(extrap_dir, f"extrap_id{samp_id}.png"))
    
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

if __name__ == '__main__':

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

    models = [#GPRModel(name = 'GPR', model_obj= HyperParam_GPR()),
    ANNModel(name= 'ANN', model_obj=create_ViscNN_concat),
    # PENNModel(name = 'PENN_WLF', model_obj=Visc_PENN_WLF, device = device), 
    PENNModel(name = 'PENN_WLF_lr1e-4_bignet_b8_pat20_lrfact0.5', model_obj=Visc_PENN_WLF, device = device),] 
    # PENNModel(name = 'PENN_WLF_lr5e-5', model_obj=Visc_PENN_WLF, device = device),] 
    #PENNModel(name = 'PENN_Arrhenius', model_obj=Visc_PENN_Arrhenius, device = device),
    # PENNModel(name = 'PENN_PI_WLF', model_obj=Visc_PENN_PI_WLF, device = device)]
    model_eval = ModelEval('training_shear_split_3', models)

    # model_eval.evaluate_models()

    model_eval.extrapolation_test("shear_extrap",sample_col = FeatureHeaders.shear_rate.value,
                                    const_col = [FeatureHeaders.mol_weight.value, FeatureHeaders.temp.value],
                                    extrap_range= np.power(10, np.linspace(-4, 6)))
    
