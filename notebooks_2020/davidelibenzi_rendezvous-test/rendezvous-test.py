#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'train.py', 'import torch\nimport torch_xla\nimport torch_xla.core.xla_model as xm\nimport torch_xla.distributed.xla_multiprocessing as xmp\nimport time\n\ndef simple_map_fn(rank, flags):\n  device = xm.xla_device()  \n  print("Process", rank ,"is using", xm.xla_real_devices([str(device)])[0])\n  xm.rendezvous(\'init\')\n  if rank == 0:\n    time.sleep(1)\n\nflags = {}\nxmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method=\'fork\')')


# In[ ]:


get_ipython().system('TF_CPP_VMODULE=mesh_service=5,computation_client=0 python train.py')

