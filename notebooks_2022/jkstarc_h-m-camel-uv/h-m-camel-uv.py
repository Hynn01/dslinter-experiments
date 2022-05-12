#!/usr/bin/env python
# coding: utf-8

# # What does your creative space look like?

# CREATIVE, WORKSPACE.
# 
# https://hmgroup.com/investors/reports/
# 
# https://www.youtube.com/channel/UC_VTDGsD6LDj7_U5NMXoFaw
# 
# https://twitter.com/jkstarclub

# In[ ]:


# TURBULENCE 25

# Huge, Sand Turbulence!
# 25 camels ran off! Directions?
# One desert cat, checking his arm length,
# turned her eyes to Dog's Flat Mass,
# Q. Dog, Why did you X?
# A. I luv Ham.

# Portraits of a cat of a dog of a ham wearing pink hoodies tracing purple keys over the desert: https://www.cameluv.com


# ![237347022.jpg](attachment:de7ac7ea-8efb-41f1-b2ad-3cf378de0179.jpg)

# ![512.png](attachment:34293ff4-4373-4225-b04b-819ba10b7cca.png)

# # Shred-5 Cylinder Design

# In[ ]:


# 지구의 5 바다
# Earth Steam Engine
# Density 4:6 2:8
# 5 cylinders
# 12 000 rpc
# 60 000 onesq
# 10 000 sixsq
# 100 cubewidth
# 1 000 000 vol
# 1 cube weight


# In[ ]:


# 10만 아이템 종류
# Item
# 25 cylinders
# 105 500 rpc
# 2 638 550 onesq
# 439 562 sixsq
# 663 cubewidth
# 291 434 247 vol
# 291 cube weight


# In[ ]:


# 137만명 고객
# Customer
# 7 cylinders
# 1 371 980 rpc
# 9 603 860 onesq
# 1 600 634 sixsq
# 1265 cubewidth
# 2 024 284 625 vol
# 2 024 cube weight


# In[ ]:


# 3천만번 거래
# Transaction
# 5 cylinders
# 31 788 324 rpc
# 158 941 620 onesq
# 26 490 270 sixsq
# 5146 cubewidth
# 136 272 852 136 vol
# 136 272 cube weight
import pandas as pd
Pattern = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")
Pattern.shape


# # Step: Two Years Receipts

# In[ ]:


# 2년 동안 3천만번 거래
# Transaction History 31M
# Duration: 2018 2020
# 3 * 10 ^ 7
len("31788324")


# In[ ]:


# 137만명 12구매
# 1644만 구매예측
# Sample
# Set 1:12
# 2 cylinders
# 1 371 980 rpc
# 2 743 960 onesq
# 457 326 sixsq
# 676 cubewidth
# 308 915 776 vol
# 308 cube weight


# # Step: ID Count

# In[ ]:


# 137만명 아이디
# df1 sample customers 1.37M
import pandas as pd
customer_id = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
df1 = customer_id.drop(customer_id.columns[1:2], axis =1)
df1


# # Step: I can see the future~

# In[ ]:


# 137만명 12 구매예측
# 1644만 예상 바코드
# df2 sample prediction 
import pandas as pd
prediction = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
df2 = prediction.drop(prediction.columns[0:1], axis =1)
df2


# In[ ]:


# 제출양식
# df1 sample customers
# df2 sample barcodes
import pandas as pd
sample = pd.concat([df1, df2], axis=1, join='inner')
display(sample)


# # Step: Vaccine Confirm

# In[ ]:


# 137만명 아이디 재확인
# df3 customer_id
import pandas as pd
customer_id = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
df3 = customer_id.drop(customer_id.columns[1:7], axis =1)
df3


# In[ ]:


# 10만 바코드
# df4 article_id
import pandas as pd
article_id = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
df4 = article_id.drop(article_id.columns[1:25], axis =1)
df4


# # Code: Interview

# In[ ]:


# 10만 바코드 상세내역 
# df5 detail_desc
import pandas as pd
article_id = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
df5 = article_id.drop(article_id.columns[1:24], axis =1)
df5


# In[ ]:


# 3천만번 거래내역
# Transaction Pattern
onedrop = Pattern.drop(Pattern.columns[0:1], axis =1)
twodrop = onedrop.drop(onedrop.columns[2:4], axis =1)
twodrop


# In[ ]:


# barcode scan
barcodecount = twodrop['article_id'].value_counts()
barcodecount


# In[ ]:


# id count
idcount = twodrop['customer_id'].value_counts()
idcount


# # Code: File

# In[ ]:


# barcode scan report
barcodecount.to_csv('barcodecount.csv')


# In[ ]:


# customer count report
idcount.to_csv('idcount.csv')


# # Code: VIP

# In[ ]:


# top 25 barcodes
barcoderank = pd.read_csv('./barcodecount.csv')
brank = barcoderank.drop(barcoderank.columns[1:2], axis =1)
brank.head(25)


# In[ ]:


# top 25 customers
idrank = pd.read_csv('./idcount.csv')
irank = idrank.drop(idrank.columns[1:2], axis =1)
irank.head(25)


# In[ ]:


# the customer rank
irank.rename(columns = {'Unnamed: 0':'customer_id'}, inplace = True)
irank


# In[ ]:


# a complete list of customers
df3


# # Code: Union

# In[ ]:


# customer union
unioncustomer = pd.merge(irank, df3, how='outer')
unioncustomer


# In[ ]:


# the barcode rank
brank.rename(columns = {'Unnamed: 0':'article_id'}, inplace = True)
brank


# In[ ]:


# A complete list of barcode
df4


# # Barcode ECU^ (Element Subset Union Intersect)

# In[ ]:


# barcode union
unionbarcode = pd.merge(brank, df4, how='outer')
unionbarcode


# In[ ]:


unioncustomer


# In[ ]:


wordrank = pd.merge(unionbarcode, df5, how = "left")
wordrank


# In[ ]:


worddrop = wordrank.drop(wordrank.columns[0:1], axis =1)
worddrop


# # merge 1 unioncustomer : 12 unionbarcode, using worddrop techtree

# In[ ]:


unioncustomer


# In[ ]:


unionbarcode


# In[ ]:


worddrop

