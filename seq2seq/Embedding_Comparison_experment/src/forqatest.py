'''
Created on 2019年3月9日

@author: zdj819
'''
from model import Model




vocab_file = './data/dl-data/couplet/vocabs'
model_dir = './data/dl-data/models'

m = Model(
        None, None, None, None, vocab_file,
        num_units=100, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)
res=open("result.txt",'w')
def inferTheStr(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        output = u'您的输入太长了'
    else:
       
        output = m.infer(' '.join(in_str))
        #output = ''.join(output.split(' '))
    #print('The first couplet：%s' % in_str,file=res)
    #print ('The second couplet：%s'%output,file=res)
    print(output, file=res)

#Introduce test set to test
testin=open('./data/dl-data/couplet/test/intest.txt','r')
testout=open('./data/dl-data/couplet/test/outtest.txt','r')
inline=[[line.strip()] for line in testin]
outline=[[line.strip()] for line in testout]
#print (len(inline))
#print (inline)
inline=[i[0].split() for i in inline]

#qlist=[['C8062A13844','C10365A22535','C10361A22535','C18524A22299','C10361A22535']]
for i in range(len(inline)):
    inferTheStr(inline[i])
    #真值
    print (' '.join(outline[i]),file=res)



