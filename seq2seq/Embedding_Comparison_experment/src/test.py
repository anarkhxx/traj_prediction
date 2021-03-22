import reader
train_input_file='./data/dl-data/couplet/train/in.txt'
train_target_file='./data/dl-data/couplet/train/out.txt'
vocab_file='./data/dl-data/couplet/vocabs'
batch_size=32
train_reader = reader.SeqReader(train_input_file,train_target_file, vocab_file, batch_size)
#
# There are many properties in this class. Just look at data[] and it's almost the same
# It is found that each data in [] has the following form:
# #{
# # 'in_seq': [71, 459, 157, 325, 55, 1],
# # 'in_seq_len': 6,
# # 'target_seq': [0, 47, 772, 472, 285, 202, 1],
# # 'target_seq_len': 6}
train_data = train_reader.read()
data=next(train_data)
print (train_reader.vocab_indices)
print (data['in_seq'])
print (data['in_seq_len'])
print (data['target_seq'])
print (data['target_seq_len'])

# Decode it
#infer_vocabs = reader.read_vocab(vocab_file)
#print(len(infer_vocabs))
#output_text = reader.decode_text(data, infer_vocabs)

# In order to test the output, it will be very difficult
# Generate training map
in_seq = data['in_seq']
in_seq_len = data['in_seq_len']
target_seq = data['target_seq']
target_seq_len = data['target_seq_len']

#To customize the specific weight of loss function, the first thing is to put in_ The last one in the QQ takes it out and stitches the target_ seq



import numpy as np
a=np.array([[1,2,3],[4,5,6,],[7,8,9]])
a=a[:,1:]
#print (a)