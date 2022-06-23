import ast
import os

HYPERPARAMETERS_MAIN = {
    # sklearn.ensemble
    "AdaBoostClassifier": {"positional": 5, "keywords": ["learning_rate"]},
    "AdaBoostRegressor": {"positional": 5, "keywords": ["learning_rate"]},
    "GradientBoostingClassifier": {"positional": 20, "keywords": ["learning_rate"]},
    "GradientBoostingRegressor": {"positional": 21, "keywords": ["learning_rate"]},
    "HistGradientBoostingClassifier": {"positional": 18, "keywords": ["learning_rate"]},
    "HistGradientBoostingRegressor": {"positional": 18, "keywords": ["learning_rate"]},
    "RandomForestClassifier": {"positional": 18, "keywords": ["min_samples_leaf", "max_features"],},
    "RandomForestRegressor": {"positional": 17, "keywords": ["min_samples_leaf", "max_features"],},
    # sklearn.linear_model
    "ElasticNet": {"positional": 12, "keywords": ["alpha", "l1_ratio"]},
    # sklearn.neighbors
    "NearestNeighbors": {"positional": 8, "keywords": ["n_neighbors"]},
    # sklearn.svm
    "NuSVC": {"positional": 15, "keywords": ["nu", "kernel", "gamma"]},
    "NuSVR": {"positional": 11, "keywords": ["C", "kernel", "gamma"]},
    "SVC": {"positional": 15, "keywords": ["C", "kernel", "gamma"]},
    "SVR": {"positional": 11, "keywords": ["C", "kernel", "gamma"]},
    # sklearn.tree
    "DecisionTreeClassifier": {"positional": 12, "keywords": ["ccp_alpha"]},
    "DecisionTreeRegressor": {"positional": 11, "keywords": ["ccp_alpha"]},
}

count_dict = {}

def walk(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if(
                hasattr(node, "func")
                and hasattr(node.func, "id")
            ):
                hyperparameter_in_class(node, node.func.id)

def hyperparameter_in_class(node, function_name: str):
    """Cheches whether the required hyperparameters are used in the class."""
    if function_name in HYPERPARAMETERS_MAIN:
        if function_name not in count_dict:
            count_dict[function_name] = {}
        for kw in node.keywords:
            if kw.arg not in count_dict[function_name]:
                count_dict[function_name][kw.arg] = 1
            else:
                count_dict[function_name][kw.arg] += 1

count = 0
for root, dirs, files in os.walk("./projects2"):
    for file in files:
        if file[-3:] == ".py" and file != "__init__.py":
            file_path = os.path.join(root, file)
            count += 1
            print(count)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    t = ast.parse(content)
                    walk(t)
            except:
                print("exception")

print(count_dict)
