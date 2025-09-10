################################################################################
def smallest_items_in_each_row_of_table(table,num_keep_per_row):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  from .smallest_items_in_list_ import smallest_items_in_list
  import numpy as np
  
  num_row,num_col = table.shape
  num_keep_per_row = np.min((num_keep_per_row,num_col))
  
  num_keep = num_keep_per_row*num_row
  yRow = np.zeros((1,num_keep),dtype=int)
  yCol = np.zeros((1,num_keep),dtype=int)
  yVal = np.zeros((1,num_keep))
  
  for row in range(num_row):
    i0 = num_keep_per_row*row
    i1 = num_keep_per_row*(row+1)
    item_index,item_value = smallest_items_in_list(table[row,:],num_keep_per_row)
    yRow[0,i0:i1] = row
    yCol[0,i0:i1] = item_index
    yVal[0,i0:i1] = item_value
  # note row and item_index are 0-based.
  # 1-based indices stored.
  yRow += 1
  yCol += 1
  
  return yRow,yCol,yVal

