#!/usr/bin/env python
# coding: utf-8

# # Reading Parquet Files - RAM/CPU Optimization 
# 
# The Bengali AI dataset is used to explore the different methods available for reading Parquet files (pandas + pyarrow).
# 
# A common source of trouble for Kernel Only Compeitions, is Out-Of-Memory errors, and the 120 minute submission time limit.
# 
# This notebook contains:
# - Syntax and performance for reading Parquet via both Pandas and Pyarrow
# - Kaggle Kernel RAM/CPU allocation
#   - 18G RAM
#   - 2x Intel(R) Xeon(R) CPU @ 2.00GHz CPU
# - RAM optimized generator function around `pandas.read_parquet()`
#   - trade 50% RAM (1700MB -> 780MB) for 2x disk IO time (5.8min -> 10.2min runtime) 
# - RAM/CPU profiling of implict dataframe dtype casting 
#   - beware of implicit cast between `unit8` -> `float64` = 8x memory usage
#   - `skimage.measure.block_reduce(train, (1,2,2,1), func=np.mean, cval=0)` can downsample images
# 

# # RAM/CPU Available In Kaggle Kernel
# 
# In theory there is 18GB of Kaggle RAM, but loading the entire dataset at once often causes out of memory errors, and doesn't leave anything for the tensorflow model. In practice, datasets need to be loaded one file at a time (or even 75% of a file) to permit a successful compile and submission run.

# In[ ]:


get_ipython().system('free -h')


# 2x Intel(R) Xeon(R) CPU @ 2.00GHz CPU
# 
# In theory this might allow for optimizations using `pathos.multiprocessing`

# In[ ]:


get_ipython().system('cat /proc/cpuinfo')


# # Available Libaries

# Both `pandas` and `pyarrow` are the two possible libaries to use
# 
# NOTE: `parquet` and `fastparquet` are not in the Kaggle pip repo, even with the latest available docket images. Whilst these can be obtained via `!pip install parquet fastparquet`, this requires an internet connection which is not allowed for Kernel only competitions.

# In[ ]:


import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

try:   import parquet
except Exception as exception: print(exception)
    
try:   import fastparquet
except Exception as exception: print(exception)    


# Other imports

# In[ ]:


import pandas as pd
import numpy as np
import pyarrow
import glob2
import gc
import time
import sys
import humanize
import math
import time
import psutil
import gc
import simplejson
import skimage
import skimage.measure
from timeit import timeit
from time import sleep
from pyarrow.parquet import ParquetFile
import pyarrow
import pyarrow.parquet as pq
import signal
from contextlib import contextmanager

pd.set_option('display.max_columns',   500)
pd.set_option('display.max_colwidth',  None)


# In[ ]:


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError


# # Memory Profiler Decorator
# Its also worth mentioning the memory_profiler `@profile` decorator for interactive debugging. 
# - NOTE: @profile / %mprun can only be used on functions defined in physical files, and not in the IPython environment.
# - https://pypi.org/project/memory-profiler/

# In[ ]:


from memory_profiler import profile


# # Read Parquet via Pandas
# 
# Pandas is the simplest and recommended option
# - it takes 40s seconds to physically read all the data
# - pandas dataset is 6.5GB in RAM. 
# 

# In[ ]:


get_ipython().system('python --version  # Python 3.6.6 :: Anaconda, Inc == original + latest docker (2020-03-14)')


# In[ ]:


pd.__version__  # 0.25.3 == original + latest docker (2020-03-14)


# In[ ]:


filenames = sorted(glob2.glob('../input/bengaliai-cv19/train_image_data_*.parquet')); filenames


# In[ ]:


def read_parquet_via_pandas(files=4, cast='uint8', resize=1):
    gc.collect(); sleep(5);  # wait for gc to complete
    memory_before = psutil.virtual_memory()[3]
    # NOTE: loading all the files into a list variable, then applying pd.concat() into a second variable, uses double the memory
    df = pd.concat([ 
        pd.read_parquet(filename).set_index('image_id', drop=True).astype('uint8')
        for filename in filenames[:files] 
    ])
    memory_end= psutil.virtual_memory()[3]        

    print( "  sys.getsizeof():", humanize.naturalsize(sys.getsizeof(df)) )
    print( "  memory total:   ", humanize.naturalsize(memory_end - memory_before), '+system', humanize.naturalsize(memory_before) )        
    return df


gc.collect(); sleep(2);  # wait for gc to complete
print('single file:')
time_start = time.time()
read_parquet_via_pandas(files=1); gc.collect()
print(f"  time:            {time.time() - time_start:.1f}s" )
print('------------------------------')
print('pd.concat() all files:')
time_start = time.time()
read_parquet_via_pandas(files=4); gc.collect()
print(f"  time:            {time.time() - time_start:.1f}s" )
pass


# # Read Parquet via PyArrow

# Creating a `ParquetFile` is very quick, and memory efficent. It only creates a pointer to the file, but allows us to read the metadata.
# 
# However there the dataset only contains a single `row_group`, meaning the file can only be read out as a single chunk (no easy row-by-row streaming)

# In[ ]:


import pyarrow
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

pyarrow.__version__  # 0.16.0 == original + latest docker (2020-03-14)


# In[ ]:


# DOCS: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
def read_parquet_via_pyarrow_file():
    pqfiles = [ ParquetFile(filename) for filename in filenames ]
    print( "sys.getsizeof", humanize.naturalsize(sys.getsizeof(pqfiles)) )
    for pqfile in pqfiles[0:1]: print(pqfile.metadata)
    return pqfiles

gc.collect(); sleep(2);  # wait for gc to complete
time_start = time.time()
read_parquet_via_pyarrow_file(); gc.collect()
print( "time: {time.time() - time_start:.1f}s" )
pass


# Using a pyarrow.Table is faster than pandas (`28s` vs `45s`), but uses more memory (`7.6GB` vs `6.5GB`) and causes an Out-Of-Memory exception if everything is read at once

# In[ ]:


# DOCS: https://arrow.apache.org/docs/python/parquet.html
# DOCS: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
# NOTE: Attempting to read all tables into memory, causes an out of memory exception
def read_parquet_via_pyarrow_table():
    shapes  = []
    classes = []
    sizes   = 0
    for filename in filenames:
        table = pq.read_table(filename) 
        shapes.append( table.shape )
        classes.append( table.__class__ )
        size = sys.getsizeof(table); sizes += size
        print("sys.getsizeof(): ",   humanize.naturalsize(sys.getsizeof(table))  )        
    print("sys.getsizeof() total:", humanize.naturalsize(sizes) )
    print("classes:", classes)
    print("shapes: ",  shapes)    


gc.collect(); sleep(2);  # wait for gc to complete
time_start = time.time()
read_parquet_via_pyarrow_table(); gc.collect()
print( f"time:   {time.time() - time_start:.1f}s" )
pass


# A generator can be written around pyarrow, but this still reads the contents of an entire file into memory and this function is really slow

# In[ ]:


import time, psutil, gc

gc.collect(); sleep(2)  # wait for gc to complete
mem_before   = psutil.virtual_memory()[3]
memory_usage = []

def read_parquet_via_pyarrow_table_generator(batch_size=128):
    for filename in filenames[0:1]:  # only loop over one file for demonstration purposes
        gc.collect(); sleep(1)
        for batch in pq.read_table(filename).to_batches(batch_size):
            mem_current = psutil.virtual_memory()[3]
            memory_usage.append( mem_current - mem_before )
            yield batch.to_pandas()


time_start = time.time()
count = 0
for batch in read_parquet_via_pyarrow_table_generator():
    count += 1

print( "time:             ", time.time() - time_start )
print( "count:            ", count )
print( "min memory_usage: ", humanize.naturalsize(min(memory_usage))  )
print( "max memory_usage: ", humanize.naturalsize(max(memory_usage))  )
print( "avg memory_usage: ", humanize.naturalsize(np.mean(memory_usage)) )
pass    


# # Pandas Batch Generator Function

# It is possible to write a batch generator using pandas. In theory this should save memory, at the expense of disk IO. 
# 
# - Timer show that disk IO increase linarly with the number of filesystem reads. 
# - Memory measurements require `gc.collect(); sleep(1)`, but show that average/min memory reduces linearly with filesystem reads
# 
# There are 8 files to read (including test files in the submission), so the tradeoffs are as follows:
# - reads_per_file 1 |  44s * 8 =  5.8min + 1700MB RAM (potentually crashing the kernel)
# - reads_per_file 2 |  77s * 8 = 10.2min +  781MB RAM (minimum required to solve the memory bottleneck)
# - reads_per_file 3 | 112s * 8 = 14.9min +  508MB RAM (1/8th of total 120min runtime)
# - reads_per_file 5 | 183s * 8 = 24.4min +  314MB RAM (1/5th of total 120min runtime)
# 
# This is a memory/time tradeoff, but demonstrates a practical solution to out-of-memory errors

# In[ ]:


memory_before = psutil.virtual_memory()[3]
memory_usage  = []

def read_parquet_via_pandas_generator(batch_size=128, reads_per_file=5):
    for filename in filenames:
        num_rows    = ParquetFile(filename).metadata.num_rows
        cache_size  = math.ceil( num_rows / batch_size / reads_per_file ) * batch_size
        batch_count = math.ceil( cache_size / batch_size )
        for n_read in range(reads_per_file):
            cache = pd.read_parquet(filename).iloc[ cache_size * n_read : cache_size * (n_read+1) ].copy()
            gc.collect(); sleep(1);  # sleep(1) is required to allow measurement of the garbage collector
            for n_batch in range(batch_count):            
                memory_current = psutil.virtual_memory()[3]
                memory_usage.append( memory_current - memory_before )                
                yield cache[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()

                
for reads_per_file in [1,2,3,5]: 
    gc.collect(); sleep(5);  # wait for gc to complete
    memory_before = psutil.virtual_memory()[3]
    memory_usage  = []
    
    time_start = time.time()
    count = 0
    for batch in read_parquet_via_pandas_generator(batch_size=128, reads_per_file=reads_per_file):
        count += 1
        
    print( "reads_per_file", reads_per_file, '|', 
           'time', int(time.time() - time_start),'s', '|', 
           'count', count,  '|',
           'memory', {
                "min": humanize.naturalsize(min(memory_usage)),
                "max": humanize.naturalsize(max(memory_usage)),
                "avg": humanize.naturalsize(np.mean(memory_usage)),
                "+system": humanize.naturalsize(memory_before),               
            }
    )
pass    


# # Dtypes and Memory Usage

# Memory useage can vary by an order of magnitude based on the implcit cast dtype. 
# 
# - Raw pixel values are read from the parquet file as `uint8`
# - `/ 255.0` or `skimage.measure.block_reduce()` will do an implict cast of `int` -> `float64`
# - `float64` results in a datastructure 8x as large as `uint8` (`13.0 GB` vs `1.8 GB`)
#   - This can be avoided by doing an explict cast to `float16` (`3.3 GB`)
# - `skimage.measure.block_reduce(df, (1,n,n,1), func=np.mean, cval=0)` == `AveragePooling2D(n)` 
#   - reduces data structure memory by `n^2` 
# 
# CPU time: 
# - `float32` (+0.5s) is the fastest cast; `float16` (+8s) is 2x slower than cast `float64` (+4s).
# - `skimage.measure.block_reduce()` is an expensive operation (3-5x IO read time)

# In[ ]:


def read_single_parquet_via_pandas_with_cast(dtype='uint8', normalize=False, denoise=False, invert=True, resize=1, resize_fn=None):
    gc.collect(); sleep(2);
    
    memory_before = psutil.virtual_memory()[3]
    time_start = time.time()        
    
    train = (pd.read_parquet(filenames[0])
               .set_index('image_id', drop=True)
               .values.astype(dtype)
               .reshape(-1, 137, 236, 1))
    
    if invert:                                         # Colors | 0 = black      | 255 = white
        train = (255-train)                            # invert | 0 = background | 255 = line
   
    if denoise:                                        # Set small pixel values to background 0
        if invert: train *= (train >= 25)              #   0 = background | 255 = line  | np.mean() == 12
        else:      train += (255-train)*(train >= 230) # 255 = background |   0 = line  | np.mean() == 244     
        
    if isinstance(resize, bool) and resize == True:
        resize = 2    # Reduce image size by 2x
    if resize and resize != 1:                  
        # NOTEBOOK: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
        # Out of the different resize functions:
        # - np.mean(dtype=uint8) produces produces fragmented images (needs float16 to work properly - but RAM intensive)
        # - np.median() produces the most accurate downsampling
        # - np.max() produces an enhanced image with thicker lines (maybe slightly easier to read)
        # - np.min() produces a  dehanced image with thiner lines (harder to read)
        resize_fn = resize_fn or (np.max if invert else np.min)
        cval      = 0 if invert else 255
        train = skimage.measure.block_reduce(train, (1, resize,resize, 1), cval=cval, func=resize_fn)  # train.shape = (50210, 137, 236, 1)
        
    if normalize:
        train = train / 255.0          # division casts: int -> float64 


    time_end     = time.time()
    memory_after = psutil.virtual_memory()[3] 
    return ( 
        str(round(time_end - time_start,2)).rjust(5),
        # str(sys.getsizeof(train)),
        str(memory_after - memory_before).rjust(5), 
        str(train.shape).ljust(20),
        str(train.dtype).ljust(7),
    )


for dtype in ['uint8', 'uint16', 'uint32', 'float16', 'float32']:  # 'float64' caused OOM error
    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype)
    print(f'dtype {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')

for denoise in [False, True]:
    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(denoise=denoise)
    print(f'denoise {denoise}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')

for normalize in [False, True]:
    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(normalize=normalize)
    print(f'normalize {normalize}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')    


# In[ ]:


# division casts: int -> float64 
for dtype in ['float16', 'float32']:
    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype, normalize=True)
    print(f'normalize {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')    


# In[ ]:


# skimage.measure.block_reduce() casts: unit8 -> float64    
for resize in [2, 3, 4]:
    for dtype in ['float16', 'float32', 'uint8']:  # 'float32' almosts causes OOM error 
        gc.collect()
        with timeout(10*60):
            seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype, resize=resize)
            print(f'resize {resize} {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')

