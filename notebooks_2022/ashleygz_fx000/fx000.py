# %% [code] {"execution":{"iopub.status.busy":"2022-05-08T03:44:42.032318Z","iopub.execute_input":"2022-05-08T03:44:42.032695Z","iopub.status.idle":"2022-05-08T03:44:42.038661Z","shell.execute_reply.started":"2022-05-08T03:44:42.032659Z","shell.execute_reply":"2022-05-08T03:44:42.037708Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# > ## Tools that can be applied at work place.

# %% [code] {"execution":{"iopub.status.busy":"2022-05-08T03:25:23.55674Z","iopub.execute_input":"2022-05-08T03:25:23.557089Z","iopub.status.idle":"2022-05-08T03:25:23.562561Z","shell.execute_reply.started":"2022-05-08T03:25:23.557056Z","shell.execute_reply":"2022-05-08T03:25:23.561745Z"}}
def type_out(s):
    print(s)
type_out('范雪')

# %% [code] {"execution":{"iopub.status.busy":"2022-05-08T03:30:28.800893Z","iopub.execute_input":"2022-05-08T03:30:28.801367Z","iopub.status.idle":"2022-05-08T03:30:28.80555Z","shell.execute_reply.started":"2022-05-08T03:30:28.801334Z","shell.execute_reply":"2022-05-08T03:30:28.804758Z"}}
print(os.path)

# %% [code]
