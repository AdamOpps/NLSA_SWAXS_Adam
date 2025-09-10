################################################################################
def block_file_template(block_type):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  if (block_type=="square"):
    return '.square_block_n{}_row{}_col{}_c{}.h5'
  elif (block_type=="pipe"):
    return '.pipe_block_n{}_diag{}_block{}_c{}.h5'
  elif (block_type=="data_chunk"):
    return '.data_{}_n{}_chunk{}.h5'
  else:
    print(SnA_error('UnknownBlockType'))
    return

