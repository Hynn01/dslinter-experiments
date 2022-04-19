# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os

def retrive_kernels():
    with open("/Users/zhanghaiyin/Desktop/硕士毕设/mark's paper/kernels_hotness_010520.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            os.system("kaggle kernels pull "+ line)

def convert_ipynb_to_py():
    for _,_,files in os.walk("/Users/zhanghaiyin/Desktop/test/dslinter_experiments/ipynb"):
        for file in files:
            os.system("jupyter nbconvert --to script ./ipynb/" + file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    convert_ipynb_to_py()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
