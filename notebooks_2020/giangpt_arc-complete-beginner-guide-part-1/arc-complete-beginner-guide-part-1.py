#!/usr/bin/env python
# coding: utf-8

# # **ROAD TO ABSTRACTION AND REASONING CORPUS PART 1**

# Okay this is my first time on Kaggle, join the competition named Abstraction and Reasoning Competition. Where I have to generate the test output from test input with very few given training input/output pairs.
# 
# After going through notebooks after notebooks to understand the problem thoroughly, I have learnt some basic steps to manipulate the data. So let's try it out.

# # 1. Step 1: Import the necessary library

# In[ ]:


import numpy as np                              #numpy library is used to work with multidimensional array.
import pandas as pd                             #panda used for data manipulation and analysis.
                 
import os                                       #os library is used for loading file to use in the program
import json                                     #json library parses json into a string or dict, and convert string or dict to json file.
from pathlib import Path                        #support path

import matplotlib.pyplot as plt                 #support ploting a figure
from matplotlib import colors                   #colors support converting number or argument into colors


# # 2. Step 2: Explore

# In this step, we will go through the datasets and explore what is in there and how to manipulate the data using our code.

# * First,we load the training tasks, the evaluation tasks, and the test_tasks.

# In[ ]:


# get the path for training_task, evaluation_task, and test_task
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

#from the path above, we load the tests file's directory into our training_tasks, evaluation_tasks, and test_tasks variables
#the sorted() function is just for the list of directory to maintain some order
training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

print("Number of examples in training corpus is ",len(training_tasks))
print("Number of examples in evaluation corpus is ",len(evaluation_tasks))
print("Number of examples in testing corpus is ",len(test_tasks))


# * Now, let's have a quick look through what training_tasks, evaluation_tasks, and test_tasks are containing by printing some of their first elements.

# In[ ]:


print(training_tasks[:3])     #printing the first 3 elements of training_tasks
print(evaluation_tasks[:3])   #printing the first 3 elements of evaluation_tasks
print(test_tasks[:3])         #printing the first 3 elements of test_tasks


#   As we can see from the results above, training_tasks, evaluation_tasks, and test_tasks are name lists of files which contain the tasks.
# 
# * Let's dig deeper and see what are in those files.

# In[ ]:


#Get the first file of the training_tasks
training_task_file = str(training_path / training_tasks[0])

#Get the first file of the evaluation_tasks
evaluation_task_file = str(evaluation_path / evaluation_tasks[0])

#Get the first file of the test_tasks
test_task_file = str(test_path / test_tasks[0])

#open the file and load it
with open(training_task_file, 'r') as f:   
    #can change training_task_file to evaluation_task_file or test_task_file to have a look at evaluation file or test file
    task = json.load(f)

#using json to load the file, the task variable now is a dictionary with keys and values, we go on and print out the keys
print(task.keys())


# In[ ]:


# The number of "train" and "test" in one training example.
n_train_pairs = len(task['train'])
n_test_pairs = len(task['test'])

print(f'task contains {n_train_pairs} training pairs')
print(f'task contains {n_test_pairs} test pairs')


# As we can see, the task consists of the "train" part and the "test" part.
# 
# The "train" part contains different training examples, each of which has given input and output.
# The "test" part contains one or more input, and the competitors task is to figure out the solution corresponding to each of the test input, with the rules figured out by working on the "train" part.

# * Next is the data structure that we are going to work on.

# In[ ]:


#display the data structure of a training's input and output
display(task['train'][0]['input'])
display(task['train'][0]['output'])


# The training input and output are both given as a 2D list. The input is mapped to the output by some particular rules that competitors need to find out.

# * In order to get more intuitive on the task, we will turn the 2D lists of numbers into grids of colors. Each number corresponds to a particular number.

# In[ ]:


cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

#plotting the training task and the test task.
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()


# In[ ]:


#plotting the first three training tasks.
for i, json_path in enumerate(training_tasks[:3]):   # can change to evaluation_tasks or test_tasks to view the evaluation and test task
    
    task_file = str(training_path / json_path)       # can change to evaluation_path or test_path to view the evaluation and test task

    with open(task_file, 'r') as f:
        task = json.load(f)

    print(f"{i:03d}", task_file)
    plot_task(task)


# Now it becomes more like a super easy IQ test that most people can solve in no time. But what about machine? How they can reason through very few training examples to figure out the correct answer? That is the challenge for ARC.

# * Prediction Format:

# In[ ]:


#using panda to read file and display the desired output
submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
display(submission.head())


# The output_id consists of 2 parts A_B. A is the ID of the task, B is the index of the test input used to generate the output. Remember that typically, there is only one test input, but in some special case, the test input will be more than 1.
# The output column represents the desired output. The output in 2D list should be convert to string, each row is converted into string of number, two consecutive rows use the sign '|' to separate them.
# For example:
#   list A = [[1,2].[3,4]] --> |12|34|
#   
# In order to perform such transformation, we can use the following code

# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred]) # turn the list into string
    str_pred = str_pred.replace(', ', '') # the replace(a,b) replace a by b in a string
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
display(example_grid)
print(flattener(example_grid))


# * For each test input, there are 3 trials for the test output, if one of them turns out to be the right answer, the task's attempt is succesful.
# For example:
# If three of your guesses are: A = [[1,2].[3,4]]; B = [[1,2].[3,5]]; C = [[1,2].[3,6]]
# Then the correct output gonna be: |12|34| |12|35| |12|36|

# In[ ]:


# the final_output takes all the guesses and use flattener to connect them.
def final_output(guess_1, guess_2, guess_3):
    return flattener(guess_1) + ' ' + flattener(guess_2) + ' ' + flattener(guess_3)


# In[ ]:


#Reuse the previous example
print(final_output(example_grid, example_grid, example_grid))


# * If we want to visualize the output using the output_id, here is how: 

# In[ ]:


def show_output(outputid):
    
    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    l=0
    for sub in submission.loc[outputid]['output'].strip().split(' '):
        out=[]
        for i in sub.split('|')[1:-1]:
                x=list(map(int,list(i)))
                out.append(x)

        axs[l].imshow(out,cmap=cmap)
        axs[l].axis('off')
        l=l+1


# In[ ]:


show_output('009d5c81_0')


# * This demonstrates how to loop over the sample submission and make predictions.

# In[ ]:


for output_id in submission.index:
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    # skipping over the training examples, since this will be naive predictions
    # we will use the test input grid as the base, and make some modifications
    data = task['test'][pair_id]['input'] # test pair input
    # for the first guess, predict that output is unchanged
    pred_1 = flattener(data)
    # for the second guess, change all 0s to 5s
    data = [[5 if i==0 else i for i in j] for j in data]
    pred_2 = flattener(data)
    # for the last gues, change everything to 0
    data = [[0 for i in j] for j in data]
    pred_3 = flattener(data)
    # concatenate and add to the submission output
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    submission.loc[output_id, 'output'] = pred

submission.to_csv('submission.csv')

