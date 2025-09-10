################################################################################
def multi_smallest_items_in_each_row_of_table(table,num_keep_per_row):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  from .multi_smallest_items_in_list_ import multi_smallest_items_in_list
  import numpy as np
  
  num_row,num_col = table.shape
  # num_keep_per_row = np.min((num_keep_per_row,num_col))
  num_keep_per_row = [min(i,num_col) for i in num_keep_per_row]
  
  # yRow, yCol and yVal are list instead of number in the parallel version of code
  yRow = []
  yCol = []
  yVal = []
  for i_nN in num_keep_per_row:
    num_keep = i_nN*num_row
    yRow.append(np.zeros((1,num_keep),dtype=int))
    yCol.append(np.zeros((1,num_keep),dtype=int))
    yVal.append(np.zeros((1,num_keep)))
  
  for row in range(num_row):
    item_index,item_value = multi_smallest_items_in_list(table[row,:],num_keep_per_row)
    for i,nN in enumerate(num_keep_per_row):
      i0 = nN*row
      i1 = nN*(row+1)
      yRow[i][0,i0:i1] = row
      yCol[i][0,i0:i1] = item_index[i]
      yVal[i][0,i0:i1] = item_value[i]
  # note row and item_index are 0-based.
  # 1-based indices stored.
  for yRow_i in yRow:
    yRow_i += 1
  for yCol_i in yCol:  
    yCol_i += 1
  
  return yRow,yCol,yVal

