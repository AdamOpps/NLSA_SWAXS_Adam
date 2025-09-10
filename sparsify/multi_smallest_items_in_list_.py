################################################################################
def multi_smallest_items_in_list(list,num_keep):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  import numpy as np
  
  list = np.array(list)

  item_value = []
  item_index = []
  # note sorting done in ascending order.
  sorted_index = np.argsort(list)

  for nN in num_keep:
    current_index = sorted_index[:nN]
    item_index.append(current_index)
    item_value.append(list[current_index])
  
  # note item_index is 0-based.
  return item_index,item_value

