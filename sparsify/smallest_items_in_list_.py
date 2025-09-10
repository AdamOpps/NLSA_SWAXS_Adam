################################################################################
def smallest_items_in_list(list,num_keep):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  import numpy as np
  
  list = np.array(list)
  # note sorting done in ascending order.
  item_index = np.argsort(list)[:num_keep]
  item_value = list[item_index]
  
  # note item_index is 0-based.
  return item_index,item_value

