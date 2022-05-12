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


# collect_all_notebook_project_names("./notebooks_2020_original", "notebook_dataset_010520.txt")
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

############################
# clone projects into project folder
# create requirements.txt file for each projects
def clone_repos():
    f = open("dataset.txt", "r")
    repos = f.readlines()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(clone_repo, repos)


def clone_repo(url):
    url = url[:-1] # need to delete the \n in the last place of the line
    name = url.split("/")[-1]
    destination_dir = "/Users/zhanghaiyin/Desktop/projects/"+name
    os.mkdir(destination_dir)
    command_line = "git clone "+url+" "+destination_dir
    print(command_line)
    os.system(command_line)


# # create projects from jupyter notebooks and get requirement.txt file
# def create_projects_from_notebooks():
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         path = "/Users/zhanghaiyin/Desktop/notebooks/notebooks"
#         for _, dirs, files in os.walk(path):
#             # for f in files:
#             #     unit_operation(path, f)
#             for f, s in zip(files, executor.map(create_project, repeat(path), files)):
#                 print(f)
#
#
# def create_project(path, f):
#     try:
#         print(f)
#         origin_file_path = os.path.join(path, f)
#         dir_path = os.path.join(path, f.split(".")[0])
#         destination_file_path = os.path.join(dir_path, f)
#         os.mkdir(dir_path)
#         shutil.move(origin_file_path, destination_file_path)
#         # pipreqs /home/project/location
#         os.system("pipreqs "+dir_path)
#         return "success"
#     except:
#         print("exception")


# count libraries usage from requirements.txt files
def count_libraries():
    # results = {}
    results = {"tensorflow1":0, "tensorflow2":0}
    root_path = "/Users/zhanghaiyin/Desktop/experiments/projects"
    # root_path = "/Users/zhanghaiyin/Desktop/experiments/notebooks/convert_copy"
    dirs = os.listdir(root_path)
    for d in dirs:
        req_path = os.path.join(root_path, d, "requirements.txt")
        print(req_path)
        # analyze_requirements(req_path, results)
        analyze_tensorflows(req_path, results)
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


# add requirement.txt file in the projects
def add_requirement_file_for_each_project():
    path = "/Users/zhanghaiyin/Desktop/projects"
    dirs = os.listdir(path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_new_requirement_file, repeat(path), dirs)


def generate_new_requirement_file(path, d):
    try:
        dir_path = os.path.join(path, d)
        req_file_path = os.path.join(dir_path, "requirements.txt")
        if os.path.exists(req_file_path):
            os.remove(req_file_path)
        os.system("pipreqs "+dir_path)
        return "success"
    except:
        print("exception")

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


def add_url():
    f = open("dataset2.txt", "r")
    repos = f.readlines()
    f2 = open("dataset3.txt", "a")
    for url in repos:
        url = "- url: " + url
        f2.write(url)


if __name__ == '__main__':
     convert_ipynb_to_py("./notebooks_2020")
