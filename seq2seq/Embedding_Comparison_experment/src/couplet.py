from model import Model

m = Model(
        './data/dl-data/couplet/train/in.txt',
        './data/dl-data/couplet/train/out.txt',
        './data/dl-data/couplet/test/int'
        'est.txt',
        './data/dl-data/couplet/test/outtest.txt',
        './data/dl-data/couplet/vocabs',
        num_units=100, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='./data/dl-data/models',
        restore_model=False)

m.train(1100)