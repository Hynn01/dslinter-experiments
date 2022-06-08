import os
import concurrent.futures
from itertools import repeat
import shutil


# make kernels_hotness_080522 dataset from the kernel list
def make_dataset():
    f_txt = open("./kernels_hotness_0805222.txt", "a")
    with open("./kernels_hotness_080522.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            kernel = line.split(",")[0][5:] + "\n"
            print(kernel)
            f_txt.write(kernel)


# experiment for notebook
# retrieve_kernels("./kernels_hotness_010520.txt")
def retrieve_kernels(filename: str):
    count = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line[:-1] # skipping \n
                count += 1
                print(count)
                # the retrieve notebooks are in notebooks folder
                path = "./notebooks/"+line.split("/")[0]+"_"+line.split("/")[1]
                os.mkdir(path)
                command = "kaggle kernels pull "+line+" -p "+path
                print(command)
                os.system(command)
            except:
                print("Not able to retrieve")


# remove the kernels that are no longer exist or change permission
def remove_empty_dir(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        file = os.path.join(dir_path, file)
        if os.path.isdir(file):
            if not os.listdir(file):
                os.rmdir(file)


# collect_all_notebook_project_names("./notebooks_2020_original", "notebook_dataset_010520.yml")
def collect_all_notebook_project_names(root_path, filename):
    dirs = os.listdir(root_path)
    f_txt = open(filename, "a")
    for d in dirs:
        if os.path.isdir(os.path.join(root_path, d)):
            string = "- folder: " + d + "\n"
            f_txt.writelines(string)
    f_txt.close()


# add_init_file("./notebooks_2022")
# add __init__.py for each folder
def add_init_file(path):
    dirs = os.listdir(path)
    for d in dirs:
        try:
            dir_path = os.path.join(path, d)
            print(dir_path)
            file_path = os.path.join(dir_path, "__init__.py")
            f = open(file_path, "x")
        except:
            pass


# convert_ipynb_to_py("./notebooks_2022")
def convert_ipynb_to_py(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            print(file_name)
            if file[-6:] == ".ipynb":
                os.system("jupyter nbconvert --to script " + file_name)
                os.remove(file_name)


def add_url():
    f = open("projects1.txt", "r")
    repos = f.readlines()
    f2 = open("projects1.yml", "a")
    for url in repos:
        url = "- url: " + url
        f2.write(url)

# get requirement.txt file
# get_reqs_from_notebooks("./notebooks_test")
def get_reqs_from_notebooks(path):
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
        if os.path.isdir(file):
            command = "pipreqs "+file
            print(command)
            os.system(command)

# add requirement.txt file in the projects
def add_requirement_file_for_each_project():
    path = "./projects2"
    dirs = os.listdir(path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_new_requirement_file, repeat(path), dirs)

def generate_new_requirement_file(path, d):
    try:
        dir_path = os.path.join(path, d)
        req_file_path = os.path.join(dir_path, "requirements.txt")
        if os.path.exists(req_file_path):
            os.remove(req_file_path)
        os.system("pipreqs --force "+dir_path)
        return "success"
    except:
        print("exception")

# count libraries usage from requirements.txt files
def count_libraries(root_path):
    results = {}
    # results = {"tensorflow1":0, "tensorflow2":0}
    dirs = os.listdir(root_path)
    for d in dirs:
        req_path = os.path.join(root_path, d, "requirements.txt")
        print(req_path)
        analyze_requirements(req_path, results)
        # analyze_tensorflows(req_path, results)
    results = dict(sorted(results.items(), key=lambda item:item[1], reverse=True))
    print(results)

def analyze_requirements(path, results):
    try:
        f = open(path, 'r')
        for line in f:
            library = line.split("==")[0]
            if library not in results:
                results[library] = 1
            else:
                results[library] += 1
    except:
        print("exception")

def analyze_tensorflows(path, results):
    try:
        f = open(path, 'r')
        for line in f:
            library = line.split("==")[0]
            version = line.split("==")[1].split(".")[0]
            name = library + version
            print(name)
            if name == "tensorflow1":
                results["tensorflow1"] += 1
            elif name == "tensorflow2":
                results["tensorflow2"] += 1
    except:
        print("exception")


# delete old projects which cannot generate a requirement.txt file (probably bc they are using py2)
def delete_old_projects():
    path = "/Users/zhanghaiyin/Desktop/experiments/projects"
    dirs = os.listdir(path)
    projects_to_delete = []
    for d in dirs:
        dir_path = os.path.join(path, d)
        req_file_path = os.path.join(dir_path, "requirements.txt")
        if not os.path.exists(req_file_path):
            try:
                projects_to_delete.append(d)
                shutil.rmtree(dir_path)
            except:
                print("not work")
    f = open("dataset.txt", "r")
    repos = f.readlines()
    f2 = open("dataset2.txt", "a")
    for url in repos:
        url = url[:-1] # need to delete the \n in the last place of the line
        name = url.split("/")[-1]
        if name not in projects_to_delete:
            try:
                url += "\n"
                f2.write(url)
            except:
                print("not work2")


def replace_comma():
    f = open("results_20200513.csv", "r")
    lines = f.readlines()
    f2 = open("results_20200513.txt", "a")
    for l in lines:
        l = l.replace(",", " & ")
        l = l[:-1] + r" \\" + l[-1]
        print(l)
        f2.write(l)

import json
import ast

def read_dicts():
    dicts = []
    merge_dict = {}
    f = open("dicts.txt", "r")
    lines = f.readlines()
    for l in lines:
        dict = ast.literal_eval(l)
        print(dict)
        dicts.append(dict)
        merge_dict = {k: add_two_dict(merge_dict.get(k, 0), dict.get(k, 0)) for k in set(merge_dict) | set(dict)}
    print(merge_dict)
    new_dict = {}
    for k, w in merge_dict.items():
        new_dict[k] = sorted(w.items(), key=lambda item: item[1], reverse=True)
    print(new_dict)

def add_two_dict(dict1, dict2):
    # print("dict1: "+str(dict1))
    # print("dict2: "+str(dict2))
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        return {i: dict1.get(i, 0) + dict2.get(i, 0) for i in set(dict1) | set(dict2)}
    else:
        return dict1 if isinstance(dict1, dict) else dict2

def add_dicts():
    merge_dict = {}
    dict1 = {'pandas': 895, 'numpy': 854, 'matplotlib': 678, 'scikit_learn': 579, 'seaborn': 487, 'scipy': 190, 'tqdm': 180, 'plotly': 160, 'keras': 147, 'lightgbm': 139, 'ipython': 133, 'xgboost': 116, 'tensorflow': 103, 'nltk': 65, 'Pillow': 62, 'torch': 51, 'wordcloud': 49, '\n': 44, 'catboost': 38, 'statsmodels': 36, 'scikit_image': 34, 'skimage': 34, 'fastai': 20, 'eli5': 19, 'mlxtend': 19, 'shap': 19, 'folium': 18, 'gensim': 17, 'numba': 16, 'networkx': 15, 'transformers': 14, 'spacy': 14, 'kaggle': 13, 'torchvision': 13, 'bokeh': 12, 'altair': 12, 'albumentations': 12, 'missingno': 12, 'cufflinks': 11, 'fbprophet': 11, 'beautifulsoup4': 10, 'squarify': 10, 'requests': 10, 'imblearn': 9, 'pydicom': 9, 'psutil': 8, 'python_dateutil': 8, 'hyperopt': 8, 'tokenizers': 8, 'joblib': 8, 'dask': 7, 'matplotlib_venn': 7, 'imgaug': 7, 'geopandas': 7, 'graphviz': 7, 'tensorflow_hub': 6, 'efficientnet': 6, 'pandas_profiling': 5, 'pycountry': 5, 'librosa': 5, 'tokenization': 5, 'textblob': 5, 'ipywidgets': 5, 'pdpbox': 4, 'Unidecode': 4, 'dicom': 4, 'six': 4, 'h5py': 4, 'Shapely': 4, 'fastprogress': 4, 'pretrainedmodels': 4, 'efficientnet_pytorch': 4, 'bubbly': 3, 'protobuf': 3, 'category_encoders': 3, 'imageio': 3, 'pydotplus': 3, 'spellchecker': 3, 'pycaret': 3, 'PyWavelets': 3, 'pyramid': 3, 'surprise': 3, 'pytorch_pretrained_bert': 3, 'pmdarima': 2, 'pymongo': 2, 'traitlets': 2, 'umap': 2, 'autocorrect': 2, 'contractions': 2, 'wordninja': 2, 'utils': 2, 'colorcet': 2, 'holoviews': 2, 'pandarallel': 2, 'chainer': 2, 'model': 2, 'trackml': 2, 'jpegio': 2, 'joypy': 2, 'chainer_chemistry': 2, 'pydot': 2, 'kaggle_environments': 2, 'ase': 2, 'torchtext': 2, 'ppscore': 2, 'pyarrow': 2, 'mrcnn': 2, 'chardet': 2, 'fuzzywuzzy': 2, 'pytrends': 2, 'reverse_geocoder': 2, 'user_agents': 2, 'googletrans': 2, 'textstat': 2, 'yellowbrick': 2, 'featuretools': 2, 'fastai2': 2, 'pyLDAvis': 2, 'ggplot': 2, 'langdetect': 2, 'tensorflow_addons': 2, 'modeling': 2, 'concorde': 1, 'chart_studio': 1, 'pygam': 1, 'cycler': 1, 'more_itertools': 1, 'pandas_summary': 1, 'plotnine': 1, 'sklearn_pandas': 1, 'data': 1, 'imutils': 1, 'timm': 1, 'absl_py': 1, 'absl': 1, 'tensorflow_probability': 1, 'skorch': 1, 'rgf': 1, 'weka': 1, 'plotly_express': 1, 'datashader': 1, 'geoviews': 1, 'spacy_cld': 1, 'mlcrate': 1, 'goodreads_api_client': 1, 'isbnlib': 1, 'newspaper': 1, 'progressbar33': 1, 'keras_radam': 1, 'mlens': 1, 'vecstack': 1, 'imagehash': 1, 'lap': 1, 'nibabel': 1, 'nilearn': 1, 'mlcomp': 1, 'hdbscan': 1, 'mxnet': 1, 'semanticscholar': 1, 'sentence_transformers': 1, 'geopy': 1, 'glob2': 1, 'fastparquet': 1, 'humanize': 1, 'memory_profiler': 1, 'parquet': 1, 'simplejson': 1, 'rake_nltk': 1, 'summarizer': 1, 'holidays': 1, 'pandas_datareader': 1, 'astropy': 1, 'gatspy': 1, 'colorlover': 1, 'scispacy': 1, 'pycountry_convert': 1, 'wand': 1, 'patsy': 1, 'urllib3': 1, 'autoviz': 1, 'dabl': 1, 'emot': 1, 'flashtext': 1, 'numerizer': 1, 'pyflux': 1, 'face_recognition': 1, 'pytesseract': 1, 'cuml': 1, 'colorama': 1, 'polyglot': 1, 'gpt_2_simple': 1, 'palettable': 1, 'dask_xgboost': 1, 'edm': 1, 'branca': 1, 'paperai': 1, 'txtai': 1, 'catalyst': 1, 'segmentation_models_pytorch': 1, 'pynvml': 1, 'wordbatch': 1, 'lda': 1, 'nlp_id': 1, 'Sastrawi': 1, 'tifffile': 1, 'segmentation_models': 1, 'ignite': 1, 'opencage': 1, 'sympy': 1, 'covsirphy': 1, 'lib': 1, 'toolz': 1, 'fastcache': 1, 'pydash': 1, 'colorthief': 1, 'apex': 1, 'moviepy': 1, 'haversine': 1, 'ml': 1, 'auto_mix_prep': 1, 'ujson': 1, 'helpers': 1, 'rasterio': 1, 'pytorch_lightning': 1, 'lyft_dataset_sdk': 1, 'pyquaternion': 1, 'facenet_pytorch': 1}
    dict2 = {'numpy': 827, 'pandas': 826, 'matplotlib': 674, 'scikit_learn': 528, 'seaborn': 436, 'tensorflow': 148, 'tqdm': 118, 'scipy': 115, 'plotly': 106, 'ipython': 93, 'xgboost': 92, '\n': 75, 'torch': 67, 'lightgbm': 66, 'keras': 58, 'Pillow': 57, 'catboost': 35, 'statsmodels': 34, 'nltk': 30, 'imblearn': 27, 'torchvision': 25, 'wordcloud': 23, 'joblib': 22, 'transformers': 20, 'fastai': 18, 'missingno': 17, 'shap': 15, 'tensorflow_addons': 14, 'requests': 14, 'albumentations': 14, 'optuna': 11, 'folium': 10, 'scikit_image': 10, 'skimage': 10, 'yellowbrick': 9, 'geopandas': 9, 'tensorflow_hub': 8, 'lightautoml': 8, 'pandas_profiling': 8, 'kaggle_environments': 8, 'category_encoders': 7, 'Shapely': 7, 'beautifulsoup4': 7, 'colorama': 7, 'cycler': 6, 'dataclasses': 6, 'segmentation_models_pytorch': 6, 'geopy': 5, 'h2o': 5, 'gensim': 5, 'datatable': 5, 'pytorch_lightning': 5, 'cudf': 4, 'librosa': 4, 'fastcore': 4, 'monai': 4, 'ipywidgets': 4, 'pandas_datareader': 4, 'timm': 4, 'wandb': 4, 'networkx': 4, 'spacy': 4, 'tensorflow_datasets': 4, 'flaml': 3, 'fastdownload': 3, 'dataprep': 3, 'mlxtend': 3, 'autogluon': 3, 'hyperopt': 3, 'plotly_express': 3, 'pycaret': 3, 'cupy': 3, 'sklearn_pandas': 3, 'tokenizers': 3, 'umap': 3, 'termcolor': 3, 'imageio': 3, 'numba': 3, 'textblob': 3, 'datasets': 3, 'pydicom': 2, 'torchaudio': 2, 'altair': 2, 'python_dateutil': 2, 'gym': 2, 'cufflinks': 2, 'matplotlib_venn': 2, 'torchinfo': 2, 'pyspark_stubs': 2, 'pyspark': 2, 'yfinance': 2, 'mplfinance': 2, 'sweetviz': 2, 'keras_tuner': 2, 'yolov5': 2, 'dtreeviz': 2, 'SoundFile': 2, 'graphviz': 2, 'prettytable': 2, 'kornia': 2, 'Unidecode': 2, 'pandarallel': 2, 'rasterio': 2, 'powershap': 2, 'Keras_Preprocessing': 2, 'cuml': 2, 'pytorch_tabnet': 2, 'fbprophet': 2, 'tsfresh': 2, 'psutil': 2, 'feyn': 2, 'path.py': 1, 'pytorch_tabular': 1, 'audiomentations': 1, 'neuralprophet': 1, 'pmdarima': 1, 'spectral': 1, 'stable_baselines': 1, 'momepy': 1, 'osmnx': 1, 'ftfy': 1, 'multiprocess': 1, 'fredapi': 1, 'xlrd': 1, 'covid19dh': 1, 'dbnomics': 1, 'dask': 1, 'd2l': 1, 'hvplot': 1, 'tensorflow_recommenders': 1, 'protobuf': 1, 'pytesseract': 1, 'tr': 1, 'ccxt': 1, 'mpl_finance': 1, 'tensorflow_text': 1, 'distance': 1, 'spellchecker': 1, 'faiss': 1, 'dabl': 1, 'torchviz': 1, 'econml': 1, 'kornia_moons': 1, 'fastbook': 1, 'tabula': 1, 'squarify': 1, 'geoplot': 1, 'pynmea2': 1, 'py7zr': 1, 'tensorflow_decision_forests': 1, 'pdpbox': 1, 'SQLAlchemy': 1, 'bokeh': 1, 'MarkupSafe': 1, 'Jinja2': 1, 'matplotlib_inline': 1, 'vosk': 1, 'lxml': 1, 'gplearn': 1, 'statannotations': 1, 'numpy_financial': 1, 'keras_self_attention': 1, 'autocorrect': 1, 'simdkalman': 1, 'tabulate': 1, 'openpyxl': 1, 'efficientnet': 1, 'pycountry': 1, 'mlens': 1, 'dataset': 1, 'net': 1, 'text_hammer': 1, 'contractions': 1, 'PyYAML': 1, 'visualkeras': 1, 'efficientnet_pytorch': 1, 'mplcyberpunk': 1, 'miceforest': 1, 'fast_tabnet': 1, 'torchmetrics': 1, 'cartopy': 1, 'feature_engine': 1, 'pygad': 1, 'boruta': 1, 'factor_analyzer': 1, 'fastcluster': 1, 'eli5': 1, 'asposestorage': 1, 'pyLDAvis': 1, 'geocoder': 1, 'mytflib': 1, 'one_cycle_tf': 1, 'sympy': 1, 'holidays': 1, 'fuzzywuzzy': 1, 'cugraph': 1, 'pynvml': 1, 'accelerate': 1, 'lime': 1, 'engine': 1, 'pydot': 1, 'antropy': 1, 'catch22': 1, 'seglearn': 1, 'tsflex': 1, 'bertviz': 1, 'flash': 1, 'roboflow': 1}
    dict3 = {'numpy': 62, 'tensorflow': 34, 'matplotlib': 28, 'scikit_learn': 23, 'scipy': 22, 'Pillow': 18, 'keras': 17, 'tqdm': 14, 'six': 14, 'pandas': 13, 'requests': 10, 'torch': 9, 'pytest': 8, 'scikit_image': 8, 'skimage': 8, 'h5py': 7, 'nltk': 6, 'torchvision': 6, 'sphinx_rtd_theme': 5, 'protobuf': 5, 'Cython': 5, 'PyYAML': 4, 'librosa': 3, 'nose': 3, 'ipython': 3, 'pycocotools': 3, 'Sphinx': 3, 'docutils': 3, 'progressbar33': 3, 'psutil': 3, 'tabulate': 3, 'absl_py': 3, 'absl': 3, 'seaborn': 3, 'pyspark_stubs': 3, 'pyspark': 3, 'numba': 2, 'chartio': 2, 'tensorboardX': 2, 'Unidecode': 2, 'moviepy': 2, 'cryptography': 2, 'boto3': 2, 'gensim': 2, 'mock': 2, 'sonnet': 2, 'joblib': 2, 'chainer': 2, 'tensorboard': 2, 'graphviz': 2, 'lxml': 2, 'Mako': 2, 'easydict': 2, 'dataclasses': 2, 'Jinja2': 2, 'click': 2, 'docopt': 1, 'gentle': 1, 'inflect': 1, 'jaconv': 1, 'lws': 1, 'MeCab': 1, 'nnmnkwii': 1, 'beautifulsoup4': 1, 'botocore': 1, 'Django': 1, 'accumulation_tree': 1, 'pyudorandom': 1, '\n': 1, 'choix': 1, 'tensorflow_probability': 1, 'networkx': 1, 'cairosvg': 1, 'rdkit': 1, 'imutils': 1, 'cupy': 1, 'numpydoc': 1, 'accelerate': 1, 'dstoolbox': 1, 'fire': 1, 'flaky': 1, 'gpytorch': 1, 'mlflow': 1, 'palladium': 1, 'sacred': 1, 'wandb': 1, 'cachetools': 1, 'ekphrasis': 1, 'frozendict': 1, 'kutilities': 1, 'pykeyboard': 1, 'pymouse': 1, 'TFANN': 1, 'win32gui': 1, 'flatbuffers': 1, 'urllib3': 1, 'caffe2': 1, 'efficientnet': 1, 'keras_applications': 1, 'keras_resnet': 1, 'keras_segmentation': 1, 'keras_self_attention': 1, 'mrcnn': 1, 'onnx': 1, 'onnxconverter_common': 1, 'onnxruntime': 1, 'onnxruntime_extensions': 1, 'parameterized': 1, 'pyinstrument': 1, 'skl2onnx': 1, 'tensorflow_hub': 1, 'tensorflow_text': 1, 'tensorflowjs': 1, 'transformers': 1, 'wget': 1, 'yolo3': 1, 'future': 1, 'typing_extensions': 1, 'tensorflow_estimator': 1, 'apache_beam': 1, 'mpi4py': 1, 'cassandra_driver': 1, 'grpcio': 1, 'glog': 1, 'grpc': 1, 'wordninja': 1, 'coverage': 1, 'regex': 1, 'dateparser': 1, 'elasticsearch': 1, 'memory_profiler': 1, 'num2words': 1, 'pycountry': 1, 'python_dateutil': 1, 'reporters_db': 1, 'spacy': 1, 'zahlwort2num': 1, 'dominate': 1, 'nibabel': 1, 'pydensecrf': 1, 'SimpleITK': 1, 'visdom': 1, 'attrs': 1, 'tornado': 1, 'keyring': 1, 'brotlipy': 1, 'lockfile': 1, 'toml': 1, 'ipywidgets': 1, 'attr': 1, 'brotli': 1, 'jnius': 1, 'ordereddict': 1, 'pyOpenSSL': 1, 'railroad': 1, 'trove_classifiers': 1, 'Flask': 1, 'dlib': 1, 'flask_socketio': 1, 'rainbow_logging_handler': 1, 'attrdict': 1, 'catboost': 1, 'imgaug': 1, 'lightgbm': 1, 'pydot_ng': 1, 'xgboost': 1, 'rawpy': 1, 'tt': 1, 'torchfile': 1, 'picamera': 1, 'pygame': 1, 'loky': 1, 'python_Levenshtein': 1, 'randomgen': 1, 'ray': 1, 'textblob': 1, 'tables': 1, 'SoundFile': 1, 'speechpy': 1, 'ply': 1, 'contextlib2': 1, 'nbformat': 1, 'args': 1, 'classification': 1, 'onnx_tf': 1, 'pydicom': 1, 'pydub': 1, 'pypng': 1, 'recommonmark': 1, 'recompute': 1, 'tf2onnx': 1, 'tflite2onnx': 1}
    dict4 = {'numpy': 34, 'torch': 30, 'torchvision': 24, 'scipy': 22, 'Pillow': 20, 'tqdm': 19, 'pandas': 17, 'scikit_learn': 16, 'matplotlib': 14, 'albumentations': 12, 'efficientnet_pytorch': 7, 'scikit_image': 6, 'skimage': 6, 'tensorflow': 6, 'apex': 6, 'pretrainedmodels': 5, 'timm': 5, 'pydicom': 5, 'imageio': 4, 'requests': 4, 'click': 4, 'six': 4, 'PyYAML': 4, 'seaborn': 4, 'pycocotools': 4, 'lightgbm': 4, 'ipython': 3, 'cnn_finetune': 3, 'absl_py': 3, 'absl': 3, 'dataclasses': 3, 'h5py': 3, 'pytest': 3, 'segmentation_models_pytorch': 3, 'Cython': 3, 'tensorboardX': 3, 'mmcv': 3, 'cityscapesscripts': 3, 'omegaconf': 2, 'fastai': 2, 'catalyst': 2, 'dm_tree': 2, 'attrs': 2, 'attr': 2, 'chex': 2, 'gym': 2, 'haiku': 2, 'protobuf': 2, 'sacred': 2, 'tensorflow_datasets': 2, 'tree': 2, 'transformers': 2, 'onnxruntime': 2, 'fvcore': 2, 'Sphinx': 2, 'imagecorruptions': 2, 'terminaltables': 2, 'xgboost': 2, 'moviepy': 2, 'librosa': 1, 'SoundFile': 1, 'hydra': 1, 'pesq': 1, 'pystoi': 1, 'torchaudio': 1, 'Flask': 1, 'ffmpeg': 1, 'youtube_dl': 1, 'dill': 1, 'SQLAlchemy': 1, 'astropy': 1, 'networkx': 1, 'wrapt': 1, 'nltk': 1, 'acme.hello': 1, 'annoy': 1, 'atari_py': 1, 'cleverhans': 1, 'ctypes_snappy': 1, 'distrax': 1, 'dm_control': 1, 'dm_env': 1, 'dopamine': 1, 'einops': 1, 'graph_nets': 1, 'jaxline': 1, 'jraph': 1, 'labmaze': 1, 'ml_collections': 1, 'ogb': 1, 'open_spiel': 1, 'optax': 1, 'ordered_set': 1, 'pyscf': 1, 'rdkit': 1, 'reverb': 1, 'rlax': 1, 'shapeguard': 1, 'sonnet': 1, 'tensor2tensor': 1, 'tensorflow_addons': 1, 'tensorflow_gan': 1, 'tensorflow_hub': 1, 'tensorflow_probability': 1, 'trfl': 1, 'unrestricted_advex': 1, 'hickle': 1, 'torchfile': 1, 'kornia': 1, 'psutil': 1, 'av': 1, 'iopath': 1, 'simplejson': 1, 'slowfast': 1, 'sympy': 1, 'numba': 1, 'OWSLib': 1, 'python_dateutil': 1, 'safitty': 1, 'ttach': 1, 'Box2D': 1, 'graphviz': 1, 'gym_minigrid': 1, 'nevergrad': 1, 'ray': 1, 'termcolor': 1, 'colorama': 1, 'flowiz': 1, 'GitPython': 1, 'SimpleITK': 1, 'vtk': 1, 'submitit': 1, 'imgaug': 1, 'prefetch_generator': 1, 'whale': 1, 'flax': 1, 'jax': 1, 'sphinx_autodoc_typehints': 1, 'sphinxcontrib_programoutput': 1, 'tensorflow_privacy': 1, 'typing_extensions': 1, 'easydict': 1, '\n': 1, 'joblib': 1, 'toml': 1, 'keyring': 1, 'lockfile': 1, 'lxml': 1, 'brotlipy': 1, 'Jinja2': 1, 'ipywidgets': 1, 'docutils': 1, 'cryptography': 1, 'tornado': 1, 'brotli': 1, 'jnius': 1, 'ordereddict': 1, 'pyOpenSSL': 1, 'railroad': 1, 'trove_classifiers': 1, 'chartio': 1, 'adabound': 1, 'japanize_matplotlib': 1, 'logzero': 1, 'optuna': 1, 'yacs': 1, 'xlrd': 1, 'asynctest': 1, 'instaboostfast': 1, 'kwarray': 1, 'onnx': 1, 'xlutils': 1, 'addict': 1, 'catboost': 1, 'lpips': 1, 'mxnet': 1, 'warmup_scheduler': 1, 'keras': 1, 'keras_applications': 1, 'ignite': 1, 'jpeg4py': 1, 'json_log_plots': 1, 'batchgenerators': 1, 'wget': 1, 'fastai2': 1, 'feather': 1, 'beautifulsoup4': 1, 'dominate': 1, 'visdom': 1}
    merge_dict = add_two_dict(merge_dict, dict1)
    merge_dict = add_two_dict(merge_dict, dict2)
    merge_dict = add_two_dict(merge_dict, dict3)
    merge_dict = add_two_dict(merge_dict, dict4)
    new_dict = sorted(merge_dict.items(), key=lambda item: item[1], reverse=True)
    print(new_dict)

if __name__ == '__main__':
    add_dicts()
