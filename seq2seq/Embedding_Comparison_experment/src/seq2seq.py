import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from tensorflow.python.layers import core as layers_core


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj

def getLayeredCell(layer_size, num_units, input_keep_prob,
                   output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units),
                                                input_keep_prob, output_keep_prob) for i in range(layer_size)])


def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    # encode input into a vector
    bi_layer_size = int(layer_size / 2)
    encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encode_cell_fw,
        cell_bw=encode_cell_bw,
        inputs=embed_input,
        sequence_length=in_seq_len,
        dtype=embed_input.dtype,
        time_major=False)

    # concat encode output and state
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state


def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
                           input_keep_prob):
    attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                              encoder_output, in_seq_len, normalize=True)
    # attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,
    #         encoder_output, in_seq_len, scale = True)
    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
                                               attention_layer_size=num_units)
    return cell


def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,
                           use_bias=False, name='output_mlp')


def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,
                  encoder_state, num_units, layers, embedding, output_size,
                  input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)
    helper = tf.contrib.seq2seq.TrainingHelper(
        target_seq, target_seq_len, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    return outputs.rnn_output


def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,
                  embedding, output_size, input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)

    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)

    # TODO: start tokens and end tokens are hard code
    """
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    """

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.fill([batch_size], 0),
        end_token=1,
        initial_state=init_state,
        beam_width=10,
        output_layer=projection_layer,
        length_penalty_weight=1.0)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    return outputs.sample_id


# The first four input are all set characters of TF, and the others are variables
def seq2seq(in_seq, in_seq_len, target_seq, target_seq_len, vocab_size,
            num_units, layers, dropout):
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]

    if target_seq != None:
        input_keep_prob = 1 - dropout
    else:
        input_keep_prob = 1

    # Projection layer
    projection_layer = layers_core.Dense(vocab_size, use_bias=False)

    # embedding input and target sequence
    embedding_matrix = load_pickle('./embedding_matrix_1.pkl')
    with tf.device('/cpu:0'):
        # This part should be where to load the trained word vector. This part should be where to load the trained word vector

        # embedding = tf.get_variable(
        #     name='embedding',
        #     shape=[vocab_size, num_units])

        embedding = tf.Variable(embedding_matrix,name='embedding', dtype=tf.float32)

    # In to be entered_ Seq corresponds to word vector one by one, and completes word embedding
    embed_input = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

    # encode and decode
    encoder_output, encoder_state = bi_encoder(embed_input, in_seq_len,
                                               num_units, layers, input_keep_prob)

    # The decoder part of attention is loaded, that is, the part of encoder is weighted and taken as input
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)
    # If in_ If the corresponding word of SEQ is not empty, the word is embedded. If it is empty, the initial embedding, in is passed in_ seq_ Len all 0
    if target_seq != None:
        embed_target = tf.nn.embedding_lookup(embedding, target_seq,
                                              name='embed_target')
        helper = tf.contrib.seq2seq.TrainingHelper(
            embed_target, target_seq_len, time_major=False)
    else:
        # TODO: start tokens and end tokens are hard code
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    # Decode
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    # Returns output or sample_ id
    if target_seq != None:
        return outputs.rnn_output
    else:
        return outputs.sample_id


def seq_loss(output, weight, seq_len):
    #target = target[:, 1:]
    #The passed in target has been changed to weighted
    weight = weight[:, 1:,:]

    #logits = tf.nn.softmax(output, axis=-1)

    #These two costs are equivalent, but Logits is not softmax
    #cost = -tf.reduce_sum(tf.log(logits)*weight,-1)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=weight)

    print(cost.shape)
    batch_size = tf.shape(weight)[0]
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = cost * tf.to_float(loss_mask)

    return tf.reduce_sum(cost) / tf.to_float(batch_size)
