#!/usr/bin/env python
# coding: utf-8

# # Tabular Playground Series - Feb 2022 and simple MLP
# 
# ![fmicb-11-00257-g002.jpg](attachment:113697be-e876-4a7c-84a6-66e951457528.jpg)
# 
# This project is based on the Tabular Playground Series - Feb 2022 from <a href="https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/overview" target="_blank">Kaggle</a>, which is based on data from original research article <a href="https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full" target="_blank">hyper"Analysis of Identification Method for Bacterial Species and Antibiotic Resistance Genes Using Optical Data From DNA Oligomers"</a>. 
# 
# The targets of this work are formulated at the beginning of the article above.
# 
# `Bacterial antibiotic resistance is becoming a significant health threat, and rapid identification of antibiotic-resistant bacteria is essential to save lives and reduce the spread of antibiotic resistance. This paper analyzes the ability of machine learning algorithms (MLAs) to process data from a novel spectroscopic diagnostic device to identify antibiotic-resistant genes and bacterial species by comparison to available bacterial DNA sequences. Simulation results show that the algorithms attain from 92% accuracy (for genes) up to 99% accuracy (for species). This novel approach identifies genes and species by optically reading the percentage of A, C, G, T bases in 1000s of short 10-base DNA oligomers instead of relying on conventional DNA sequencing in which the sequence of bases in long oligomers provides genetic information. The identification algorithms are robust in the presence of simulated random genetic mutations and simulated random experimental errors. Thus, these algorithms can be used to identify bacterial species, to reveal antibiotic resistance genes, and to perform other genomic analyses. Some MLAs evaluated here are shown to be better than others at accurate gene identification and avoidance of false negative identification of antibiotic resistance.`
# 
# Let's begin. Load required modules.

# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "from IPython.display import display\nfrom IPython.display import HTML\n\nfrom sklearn.decomposition import PCA\nfrom sklearn.feature_selection import mutual_info_classif\n\nfrom sklearn.metrics import accuracy_score\n\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.preprocessing import PowerTransformer\nfrom sklearn.preprocessing import RobustScaler\nfrom sklearn.preprocessing import StandardScaler\n\nfrom sklearn.model_selection import RepeatedKFold\nfrom sklearn.model_selection import RepeatedStratifiedKFold\nfrom sklearn.model_selection import train_test_split\n\nimport copy\nimport datetime\nimport numpy as np\nimport math\nimport matplotlib.pyplot as plt\nimport pandas as pd \nimport re\nimport seaborn as sns\nimport time\nimport tensorflow as tf\nimport warnings\nwarnings.filterwarnings('ignore')  \n\nfrom tensorflow import keras\n\nfrom tensorflow.keras import activations\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras import initializers\nfrom tensorflow.keras import Input\nfrom tensorflow.keras import layers\nfrom tensorflow.keras import regularizers\nfrom tensorflow.keras import Sequential\n\nfrom tensorflow.keras.callbacks import EarlyStopping\n\nfrom tensorflow.keras.layers import BatchNormalization\nfrom tensorflow.keras.layers import Dense\nfrom tensorflow.keras.layers import Dropout\n\npd.options.display.max_columns = 300\npd.options.display.max_rows = 300\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))")


# ## 1. Exploratory Data Analyst
# 
# ### 1.1. Load analyze and prepare data - remove duplicates and coding target (y) with OneHotEncoder:

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Open datasets link for download bellow\n# https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/data\ntrain_raw = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2022/train.csv")\ntest_raw = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2022/test.csv")\nsubmission_raw = pd.read_csv("/kaggle/input/tabular-playground-series-feb-2022/sample_submission.csv")\ntrain_raw.set_index("row_id", inplace=True)\ntest_raw.set_index("row_id", inplace=True)\n# Check nan values\nprint("The train has {} features with nan values."\\\n      .format(list(train_raw.isnull().sum().values > 0).count(True)))\nprint("The test has {} features with nan values."\\\n      .format(list(test_raw.isnull().sum().values > 0).count(True)))\nprint("The sample_submission has with  {} features nan values."\\\n      .format(list(submission_raw.isnull().sum().values > 0).count(True)))\n# Exploraratory data analyst\n# Check duplicates\ntrain_duplicated_rows = train_raw[train_raw.duplicated()==True].shape[0]\nprint("\\ntrain dataset contains {:,} rows from total {:,} rows.\\\nShare duplicates = {:.2f}%".format(train_duplicated_rows,\n                                  train_raw.shape[0],\n                                  100.0*train_duplicated_rows/train_raw.shape[0]))\ntest_duplicated_rows = test_raw[test_raw.duplicated()==True].shape[0]\nprint("\\ntest dataset contains {:,} rows from total {:,} rows.\\\nShare duplicates = {:.2f}%".format(test_duplicated_rows,\n                                  test_raw.shape[0],\n                                  100.0*test_duplicated_rows/test_raw.shape[0]))\n# Remove duplicates from train and test datasets\ntrain = train_raw.drop_duplicates() \ntest = test_raw.drop_duplicates() \n# Check y range and  dispersion - the uniform distribution\ntrain_target_dispersion = train.target.value_counts(normalize=True)*100\nprint(train_target_dispersion)\nprint("\\nTotal classes for target :{:}.".format(len(train_target_dispersion.values)))\n# Encode target one hot encoder and separate to X and Y\ntrain_x_all = train.iloc[:,:-1].values\ntrain_y_all = train.iloc[:,[-1]].values\ntest_x_all = test.values\nohe = OneHotEncoder()\nohe.fit(train_y_all)\ntrain_y_all_ohe = ohe.transform(train_y_all).toarray()')


# Define scaling function for further using.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Scaler for array\ndef npscaler(x_values, scaler="ss"):\n    """\n    Scale/transform np array. \n    Possible scale/transform option for x features:\n    1. None – not scale or trainsform\n    2. “ptbc”   Power-transformer by Box-Cox\n    3. “ptbc” - .PowerTransformer by Yeo-Johnson’\n    4. “rb” - .RobustScaler(\n    5. "ss" - StandardScaler    \n    For prevent data leakage using separate instance scaler/transformer \n    for each train and test parts.\n    Parameters\n    ----------\n        x_values :np.array with numeric values of features.\n        scaler : TYPE - None or str, optional.  The default is None.\n    Returns\n    -------\n        x_vals - scaled/transformed np.array\n    """\n    scalers = ["ptbc", "ptyj", "rb", "ss"]\n    x_vals = np.copy(x_values)\n    mms = MinMaxScaler(feature_range=(1, 2))\n    ptbc = PowerTransformer(method=\'box-cox\')\n    ptyj = PowerTransformer()\n    rb = RobustScaler(unit_variance=True)\n    ss = StandardScaler()\n        \n    if scaler == "ptbc":\n        x_vals = ptbc.fit_transform(mms.fit_transform(x_vals[:,:]))\n                         \n    elif scaler == "ptyj":\n        x_vals = ptyj.fit_transform(x_vals[:,:])\n    \n    elif scaler == "rb":\n        x_vals = rb.fit_transform(x_vals[:,:]), \\\n    \n    elif scaler == "ss":\n        x_vals =  ss.fit_transform(x_vals[:,:])\n        \n    if scaler not in scalers:\n        return "Value error for \'scaler\'!Enter \\\n\'ptbc\' or", " \'ptyj\' or \'rb\' or \'ss\' value for scaler!"\n    return x_vals')


# ### 1.2 Find input and output biases

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Find out put bias vector for y - divide pos==1 to neg==0\noutput_bias = []\nfor i in range (train_y_all_ohe.shape[1]):\n    neg, pos = np.bincount(train_y_all_ohe[:,0].astype(int))/train_y_all_ohe[:,0].shape[0]\n    output_bias.append(np.log(pos/neg))\noutput_bias = np.array(output_bias)\n\n\n\n# Find input bias  for Standard Scaler  using this link\n# https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full\n\ntrain_features = list(train.columns)[:-1]\ninput_bias=[]\nreg_a = (r"(A\\d+)")\nreg_t = (r"(T\\d+)")\nreg_g = (r"(G\\d+)")\nreg_c = (r"(C\\d+)")\nfor name in train_features:\n    int_a = int(re.findall(reg_a, name)[0].replace("A",""))\n    int_t = int(re.findall(reg_t, name)[0].replace("T",""))\n    int_g = int(re.findall(reg_g, name)[0].replace("G",""))\n    int_c = int(re.findall(reg_c, name)[0].replace("C",""))\n    bias = ((1/4)**10) * math.factorial(10)/(math.factorial(int_a)\n                                          * math.factorial(int_t)\n                                          * math.factorial(int_g)\n                                          * math.factorial(int_c))\n    input_bias.append(bias)\n\n# 1. Scale input bias with standard scaler and extract\ntrain_x_all_with_bias = np.append(np.array(input_bias).reshape(1,-1),\n                                       train_x_all, axis=0)\ntrain_x_all_with_bias_ss = npscaler(train_x_all_with_bias)\ninput_bias_ss = train_x_all_with_bias_ss[-1,:]')


# ### 1.3 Estimate values of  PCA components for train and test datasets

# In[ ]:


get_ipython().run_cell_magic('time', '', '# PCA analysis for train dataset\npca_train=PCA()\npca_train.fit(npscaler(train_x_all))\npca_train_cumsum = np.cumsum(pca_train.explained_variance_ratio_)\npca_train_comp_no = np.array(list(range(1,len(pca_train_cumsum)+1)))\n# define number of components with 95% variance\npca_train_comp = np.argmax(pca_train_cumsum >= 0.95) + 1\npca_train_df = pd.DataFrame(data=pca_train_cumsum , \n                           columns=["pca_var_ratio"])\n\n# Check PCA for test dataset\npca_test=PCA()\npca_test.fit(npscaler(test_x_all))\npca_test_cumsum = np.cumsum(pca_test.explained_variance_ratio_)\npca_test_comp_no = np.array(list(range(1,len(pca_test_cumsum)+1)))\n# define number of components with 95$ variance\npca_test_comp = np.argmax(pca_test_cumsum >= 0.95) + 1\npca_test_df = pd.DataFrame(data=pca_test_cumsum , \n                           columns=["pca_var_ratio"])\n\nfig, ax = plt.subplots(figsize=(15,10))\nax.plot(pca_test_comp_no, pca_test_cumsum, label="test")\nax.plot(pca_train_comp_no, pca_train_cumsum, label="train")\nax.legend()\n# Set major sticks\nmajor_xticks = np.arange(0,len(pca_train_comp_no)+15, 50)\nmajor_yticks = np.arange(0,1.0, 0.1)\n\n# Set minor sticks\nminor_xticks = np.arange(0,len(pca_train_comp_no)+15, 5)\nminor_yticks = np.arange(0,1.0, 0.025)\n\n# Define major and minor sticks\nax.tick_params(axis = \'both\', which = \'major\', labelsize = 12)\nax.tick_params(axis = \'both\', which = \'minor\', labelsize = 12)\n\nax.set_xticks(major_xticks)\nax.set_xticks(minor_xticks, minor = True)\n\nax.set_yticks(major_yticks);\nax.set_yticks(minor_yticks, minor = True);\n\nax.grid(visible=True, which="both", axis="both")\n\n# ax labels\nax.set_ylabel(\'Cumulative sum\', fontsize=12, fontweight="bold")\nax.set_xlabel("PCA futures number", fontsize=12, fontweight="bold")\nax.legend()\nax.set_title("Cumulative sum of PCA components for test and train dataset.\\\n\\n Numbers PCA\'s cumponents  for  95% cumsum for test = {:}, for train = {:}.".format(\n    pca_test_comp, pca_train_comp),\n             fontsize=13,\n             fontweight="bold")\nax.spines[\'right\'].set_visible(False)\nax.spines[\'top\'].set_visible(False)\n\nplt.show()')


# As see above 95% of cumulative sum of variance ratio for train dataset equal 226 features, but for test dataset equal 231. Remain last value for further using and apply PCA for train and test dataset and calculate input bias for train dataset after PCA conversion.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Remain 231 PCA components for PCA and transform it\ntrain_pca=PCA(n_components=231, svd_solver='randomized')\n\ntrain_pca_x_all=train_pca.fit_transform(npscaler(train_x_all))\n\n# Calculate input_bias for pca, convert with SS for futrher data scaling\ntrain_pca_x_all_with_bias = train_pca.fit_transform(train_x_all_with_bias_ss)\ntrain_pca_x_all_with_bias_ss = npscaler(train_pca_x_all_with_bias) \ninput_bias_pca = train_pca_x_all_with_bias_ss[-1,:]\n\n# Convert train with PCA\ntest_pca=PCA(n_components=231, svd_solver='randomized')\ntest_pca_x_all=test_pca.fit_transform(npscaler(test_x_all))")


# ### 1.4 Estimate mutual information for classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', 'mut_inf_clf = mutual_info_classif(train_x_all, train_y_all)\nmut_inf_df = pd.DataFrame(data=list(train.columns)[:-1], columns=[\'features\'])\nmut_inf_df["mi_clfs"]=mut_inf_clf\nnut_inf_zero_vals = mut_inf_df[mut_inf_df["mi_clfs"]==0].count().values[1] \n\n# Plot histogram values\nfig, ax = plt.subplots(figsize=(12,8))\nax=sns.histplot(data=mut_inf_df, x="mi_clfs", kde=True, ax=ax)\nax.set_title("Histogram of mutual information values. Numbers values with \\\nzero mutual information values = {}.".format(nut_inf_zero_vals))\nax.spines[\'right\'].set_visible(False)\nax.spines[\'top\'].set_visible(False)\nax.set_xlabel("Mutual information values", fontsize=12, fontweight="bold")\nplt.show()')


# ### 1.5 Preliminary conclusion from EDA
# 
# As see above dataset contains 286 features with continuous values and ten targets. PCA variance of features almost evenly distributed. Mutual information for a discrete target variable by features hasn't zero values and have range 0.1 - 1.4.  Classical multilabel classification task for classical Multi Layer Perceptron (MLP).
# 
# ## 2. Define and normalization model
# ​
# I spent the huge  of time on the selection of hyperparameters and the optimal network topology. I wrote my own functions for cross validation, tried  use Keras Tuner - but the result was not very good in all cases - these programs worked slowly, did not use multithreading and etc.  In the end, I remembered `Ockham's razor` - `plurality should not be posited without necessity`. Regarding the number of neurons, I used the thumb rule - no more than the number of futures, but the number of layers, I searched with the help of <a href="https://docs.ray.io/en/latest/tune/index.html" target="_blank">Ray Tune</a>. This <a href="https://github.com/Vadim-Maklakov/Data-Science/blob/main/08_Kaggle_Tabular_Playground_Series%20-Feb%202022/tune_tps_02_2022.py" target="_blank">file</a> contains the code for finding the optimal topology, learnrate, Dropout. Execute partially in the IDE with IPython like Spyder. 
# 
# ## 3. Train and predict
# 
# Prepare input and output bias for TF.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Convert input and output  biases to to tf constant\ntf_output_bias = tf.keras.initializers.Constant(output_bias)\ntf_input_bias = tf.keras.initializers.Constant(input_bias_ss)\ntf_input_bias_pca = tf.keras.initializers.Constant(input_bias_pca)')


# Define model and required functions

# In[ ]:


def npds_train_test_split(x, y, test_ratio=0.2, batch_sz=512, scaler="ss"):
    """
    Convert, shaffle and scale numpy arrays to tf.data.Dataset.
    Parameters
    ----------
    x : input np.array.
    y : input np.array.
    test_ratio : float, optional, the default is 0.2.
        Ratio for test part to all array  rows. If None return all 
        tf.data.Dataset.
    batch_sz : int, optional. The default is 1024.
        batch size
    scaler : string, optional. The default is "ss".
    Returns
    -------
    tf.data.Dataset.
    """

    if test_ratio != None and test_ratio < 1.0 and isinstance(test_ratio,float):
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=test_ratio, stratify=y,random_state=42)
        x_train, x_test = npscaler(x_train, scaler), npscaler(x_test, scaler)
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train = ds_train.shuffle(buffer_size=x_train.shape[0])            .batch(batch_sz).prefetch(1)
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.shuffle(buffer_size=x_test.shape[0])            .batch(batch_sz).prefetch(1)    
        return ds_train, ds_test
    elif test_ratio == None:
        x = npscaler(x)
        ds_all = tf.data.Dataset.from_tensor_slices((x, y))            .shuffle(buffer_size=x.shape[0]).batch(batch_sz)
        return ds_all


def clf_adam(shape_x=train_x_all.shape[1], learn_rate=0.005623, 
             drop_out_in=0.35, drop_out_1=0.05, drop_out_2=0.0, drop_out_3=0.1,
             drop_out_4=0.1, input_bias=tf_input_bias, 
             output_bias=tf_output_bias):
    # Define metrics
    metrics_short = [
          tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
    # Create model
    model = Sequential()
    #input layer
    model.add(layers.Dense(units=shape_x, input_shape=(shape_x,),
                           kernel_initializer="GlorotUniform",
                           bias_initializer=input_bias, 
                           activation="relu"))
    model.add(Dropout(rate=drop_out_in))
    model.add(BatchNormalization())
    
    # First hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_1))
    model.add(BatchNormalization())
    
    # Second hidden layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_2))
    model.add(BatchNormalization())
    
    # Trird layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_3))
    model.add(BatchNormalization())
    
    # Fourth layer
    model.add(layers.Dense(units=72, kernel_initializer="GlorotUniform",
                                   activation="relu"))
    model.add(Dropout(rate=drop_out_4))
    model.add(BatchNormalization())
    
    # add final layer
    model.add(layers.Dense(units=10, bias_initializer=output_bias))
    model.add(layers.Activation(activations.softmax))
            
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learn_rate),
        metrics=metrics_short)
    
    return model


def train_model(model, x_train, y_train_ohe, test_sz=0.2, batch_sz=1024, 
              stop_no=30) :
    callbacks = [EarlyStopping(monitor='categorical_accuracy',mode='max',
                                patience=stop_no,restore_best_weights=True)]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train_ohe,
                                                        test_size=test_sz,
                                                        stratify=y_train_ohe,
                                                        random_state=42)
    
    x_train, x_test = npscaler(x_train), npscaler(x_test)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=x_train.shape[0]).batch(batch_sz)        .prefetch(1)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.shuffle(buffer_size=x_test.shape[0]).batch(batch_sz)        .prefetch(1)
    ds_start=time.time()
    model_check = copy.copy(model)
    ds_history = model_check.fit(ds_train,
                 epochs=10000,
                 validation_data=ds_test,
                 callbacks=callbacks,
                 verbose=0)
    ds_end=time.time()
    ds_total_time= datetime.timedelta(seconds = (ds_end-ds_start))
    ds_history_df = pd.DataFrame(data=ds_history.history)
    ds_history_df.sort_values(by='val_loss', ascending=True, 
                          inplace=True)
    ds_history_df["epochs"]=ds_history_df.index + 1
    ds_history_df["time"]=ds_total_time
    return model_check, ds_history_df


# ## 3.1 Train model without PCA

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_adam, model_adam_hist = train_model(clf_adam(), train_x_all, \n                                          train_y_all_ohe)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Plot accuracy and loss for all 286 features \nfig, ax = plt.subplots(figsize=(18,12))\nax = sns.lineplot(data=model_adam_hist.set_index("epochs"), ax = ax)\n\nax.set_title(" Accuracy and Loss for all 286 features.", fontsize=16, \n             fontweight="bold")\nax.spines[\'right\'].set_visible(False)\nax.spines[\'top\'].set_visible(False)\n\n\n# Set major sticks\nmajor_xticks = np.arange(0,(divmod(model_adam_hist.shape[0], 50)[0]+1)*50, 50)\nmajor_yticks = np.arange(0,1.1, 0.1)\n\n# Set minor sticks\nminor_xticks = np.arange(0,(divmod(model_adam_hist.shape[0], 50)[0]+1)*50, 10)\nminor_yticks = np.arange(0,1.1, 0.02)\n\n# Define major and minor sticks\nax.tick_params(axis = \'both\', which = \'major\', labelsize = 18)\nax.tick_params(axis = \'both\', which = \'minor\', labelsize = 12)\n\nax.set_xticks(major_xticks)\nax.set_xticks(minor_xticks, minor = True)\n\nax.set_yticks(major_yticks);\nax.set_yticks(minor_yticks, minor = True);\n\nax.grid(visible=True, which="both", axis="both")\n\n# ax labels\nax.set_ylabel(\'Accuracy/Loss\', fontsize=12, fontweight="bold")\nax.set_xlabel("Epochs", fontsize=12, fontweight="bold")\nlegend = ax.legend(shadow=True, fontsize=15)')


# Evaluate model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ds_xy = npds_train_test_split(train_x_all, train_y_all_ohe, batch_sz=32, test_ratio=None)\nmodel_adam_acc = model_adam.evaluate(ds_xy)')


# In[ ]:


print("Accuracy model: ",model_adam_acc[1])
print("Loss model: ",model_adam_acc[0])


# ## 3.2 Train model with PCA

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_adam_pca, model_adam_hist_pca = train_model(clf_adam(\n    shape_x=train_pca_x_all.shape[1], input_bias = tf_input_bias_pca),\n                                          train_pca_x_all, \n                                          train_y_all_ohe)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Plot accuracy and loss for  231 PCA features \nfig, ax = plt.subplots(figsize=(18,12))\nax = sns.lineplot(data=model_adam_hist_pca.set_index("epochs"), ax = ax)\n\nax.set_title(" Accuracy and Loss for 231 PCA features.", fontsize=16, \n             fontweight="bold")\nax.spines[\'right\'].set_visible(False)\nax.spines[\'top\'].set_visible(False)\n\n\n# Set major sticks\nmajor_xticks = np.arange(0,(divmod(model_adam_hist_pca.shape[0], 50)[0]+1)*50, 50)\nmajor_yticks = np.arange(0,1.1, 0.1)\n\n# Set minor sticks\nminor_xticks = np.arange(0,(divmod(model_adam_hist_pca.shape[0], 50)[0]+1)*50, 10)\nminor_yticks = np.arange(0,1.1, 0.02)\n\n# Define major and minor sticks\nax.tick_params(axis = \'both\', which = \'major\', labelsize = 18)\nax.tick_params(axis = \'both\', which = \'minor\', labelsize = 12)\n\nax.set_xticks(major_xticks)\nax.set_xticks(minor_xticks, minor = True)\n\nax.set_yticks(major_yticks);\nax.set_yticks(minor_yticks, minor = True);\n\nax.grid(visible=True, which="both", axis="both")\n\n# ax labels\nax.set_ylabel(\'Accuracy/Loss\', fontsize=12, fontweight="bold")\nax.set_xlabel("Epochs", fontsize=12, fontweight="bold")\nlegend = ax.legend(shadow=True, fontsize=15)')


# Evaluate PCA model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ds_pca_xy = npds_train_test_split(train_pca_x_all, train_y_all_ohe, batch_sz=32, \n                                  test_ratio=None)\nmodel_adam_pca_acc = model_adam_pca.evaluate(ds_pca_xy)')


# In[ ]:


print("Accuracy PCA model: ",model_adam_pca_acc[1])
print("Loss PCA model: ",model_adam_pca_acc[0])


# View final PCA model:

# In[ ]:


model_adam_pca.summary()


# ## 3.3 Predict values
# 
# For predict values I use model with PCA

# In[ ]:


get_ipython().run_cell_magic('time', '', 'y_predict = model_adam_pca.predict(npscaler(test_pca_x_all))\n\n# convert continuous values to binary using argmax\nfor row in y_predict:\n    row = np.where(row < row[np.argmax(row)],0,1)\n\ny_predict_char = ohe.inverse_transform(y_predict)\n\n# Create pandas dataframe submission\ntest_drop_columns = list(test.columns)\nsample_submission = test.drop(labels=test_drop_columns, axis=1)\nsample_submission["target"] = y_predict_char\n# Save predict values..Cauthon, duplicates from test dataset removed!!\nsample_submission.to_csv("sample_submission.csv")\nsample_submission.head(10)')


# ## 4. Conclusion
# 
# * 1. Always use Ockham's razor - `plurality should not be posited without necessity`. In this case of Tabular  continuous values is no need to fence a garden with fifty layers and 50 thousand neurons - it’s just that such a model will take a the huge  time for train and  always quick fall to overfiting and constantly writing in the console error 'Not enough GPU memory' . As see above, the number of neurons equal to the numbers of futures in my case gives a quite acceptable accuracy result of 99.39%.
# 
# 
# * 2. The using  of PCA in this case gives a gain in time.
# 
# 
# * 3. Ray Tune gives excellent results for searching  hyperparameters. Yes it takes 90% of the time but it's worth it.
# 
# 
# * 4. As see from plots above  both models converge up to 30 epochs, after 30 epochs begin divergence in both models.
# 
# 
# * 5. In principle, it was possible to achieve 100% accuracy, but for this it was necessary to use AdamW, SGDW, learn rate decay and weight decay - my experiments led to the fact that I received a formal reduction the validotion loss from 10% to 6%, but the validation accuracy and accuracy did not increasing. And the selection of optimal values or using  of vanilla SDGs is just a huge time consumption.
# 
# Created on May 07, 2022
# 
# @author: Vadim Maklakov, used some ideas from public Internet resources.
# 
# © 3-clause BSD License
# 
# Software environment: Debian 11, Python 3.8.13, TensorFlow 2.5.3, CUDA 11.3 on localhost.
# 
# See required installed and imported python modules in the cell No 1.
