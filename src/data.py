import tensorflow as tf
import numpy as np

def create_reg_data(rng, i_size, c_size, size_distract, input_range, w_scale):
    rng, new_rng, new_rng2, new_rng3, new_rng4 = tf.random.experimental.stateless_split(rng, 5)
    w = tf.random.stateless_normal([i_size], seed=rng) * w_scale

    x = tf.random.stateless_uniform([c_size, i_size], seed=new_rng,
                                    minval=-input_range/2, maxval=input_range/2)
    x_querry = tf.random.stateless_uniform([1, i_size], seed=new_rng2,
                                           minval=-input_range/2, maxval=input_range/2)

    y_data = tf.squeeze(tf.matmul(x, tf.expand_dims(w, axis=-1)))
    choice = tf.random.stateless_uniform([size_distract], seed=new_rng4, minval=0, maxval=c_size, dtype=tf.int32)
    y_data = tf.tensor_scatter_nd_update(y_data, tf.expand_dims(choice, axis=-1),
                                         tf.random.stateless_normal([size_distract], seed=new_rng3))

    y_target = tf.matmul(x_querry, tf.expand_dims(w, axis=-1))
    y_target = tf.expand_dims(y_target, axis=-1)

    seq = tf.concat([x, tf.expand_dims(y_data, axis=-1)], axis=-1)
    target = tf.concat([x_querry, y_target], axis=-1)
    x_querry_init = -1 * tf.matmul(x_querry, tf.transpose(tf.ones_like(x_querry)) * 0.0)
    zero = tf.concat([x_querry, x_querry_init], axis=-1)
    seq = tf.concat([seq, zero], axis=0)
    return tf.squeeze(seq), tf.squeeze(target), w

def create_ood_data(rng, i_size, c_size, input_range, w_scale):
    rng, new_rng, new_rng2, new_rng3 = tf.random.experimental.stateless_split(rng, 4)
    w = tf.random.stateless_normal([i_size], seed=rng) * w_scale

    selector = tf.zeros([3])
    choice = tf.random.stateless_uniform([3], seed=new_rng3, minval=0, maxval=3, dtype=tf.int32)
    selector = tf.tensor_scatter_nd_update(selector, tf.expand_dims(choice, axis=-1), tf.ones([3]))

    x_sample = tf.random.stateless_exponential([c_size, i_size], seed=new_rng)
    norm_x_sample = tf.norm(x_sample)
    x = x_sample / norm_x_sample * input_range * selector[0]
    x_q_sample = tf.random.stateless_exponential([1, i_size], seed=new_rng2)
    x_querry = x_q_sample / norm_x_sample * input_range * selector[0]

    x_sample = tf.random.stateless_normal([c_size, i_size], seed=new_rng)
    norm_x_sample = tf.norm(x_sample)
    x += x_sample / norm_x_sample * input_range * selector[1]
    x_q_sample = tf.random.stateless_normal([1, i_size], seed=new_rng2)
    x_querry += x_q_sample / norm_x_sample * input_range * selector[1]

    x_sample = tf.random.stateless_laplace([c_size, i_size], seed=new_rng)
    norm_x_sample = tf.norm(x_sample)
    x += x_sample / norm_x_sample * input_range * selector[2]
    x_q_sample = tf.random.stateless_laplace([1, i_size], seed=new_rng2)
    x_querry += x_q_sample / norm_x_sample * input_range * selector[2]

    y_data = tf.squeeze(tf.matmul(x, tf.expand_dims(w, axis=-1)))

    y_target = tf.matmul(x_querry, tf.expand_dims(w, axis=-1))
    y_target = tf.expand_dims(y_target, axis=-1)

    seq = tf.concat([x, tf.expand_dims(y_data, axis=-1)], axis=-1)
    target = tf.concat([x_querry, y_target], axis=-1)
    x_querry_init = -1 * tf.matmul(x_querry, tf.transpose(tf.ones_like(x_querry)) * 0.0)
    zero = tf.concat([x_querry, x_querry_init], axis=-1)
    seq = tf.concat([seq, zero], axis=0)
    return tf.squeeze(seq), tf.squeeze(target), w

def create_reg_data_sin(rng, i_size, c_size, size_distract, input_range=10, w_scale=1):
    rng, new_rng, new_rng2, new_rng3, new_rng4 = tf.random.experimental.stateless_split(rng, 5)
    amp = tf.random.stateless_uniform([1], seed=rng, minval=0.1, maxval=0.5) * w_scale
    phase = tf.random.stateless_uniform([1], seed=rng, minval=0.0, maxval=1) * np.pi * w_scale

    x = tf.random.stateless_uniform([c_size, 1], seed=new_rng, minval=-input_range/2, maxval=input_range/2)
    x_querry = tf.random.stateless_uniform([1, 1], seed=new_rng2, minval=-input_range/2, maxval=input_range/2)

    y_data = tf.sin(x + phase) * amp
    choice = tf.random.stateless_uniform([size_distract], seed=new_rng4, minval=0, maxval=c_size, dtype=tf.int32)
    y_data = tf.tensor_scatter_nd_update(y_data, tf.expand_dims(choice, axis=-1),
                                         tf.random.stateless_normal([size_distract, 1], seed=new_rng3))

    y_target = tf.sin(x_querry + phase) * amp
    seq = tf.concat([x, y_data], axis=-1)
    target = tf.concat([x_querry, y_target], axis=-1)
    y_querry_init = tf.zeros_like(y_target)

    zero = tf.concat([x_querry, y_querry_init], axis=-1)
    seq = tf.concat([seq, zero], axis=0)
    return tf.squeeze(seq), tf.squeeze(target), (phase, amp)

def create_reg_data_classic_token(rng, i_size, c_size, size_distract, input_range, w_scale):
    rng, new_rng, new_rng2, new_rng3, new_rng4 = tf.random.experimental.stateless_split(rng, 5)
    w = tf.random.stateless_normal([i_size], seed=rng) * w_scale

    x = tf.random.stateless_uniform([c_size, i_size], seed=new_rng, minval=-input_range/2, maxval=input_range/2)
    x_querry = tf.random.stateless_uniform([1, i_size], seed=new_rng2, minval=-input_range/2, maxval=input_range/2)
    y_data = tf.squeeze(tf.matmul(x, tf.expand_dims(w, axis=-1))) 
    y_data_zero = tf.zeros_like(x[:, :-1])
    y_data = tf.concat([y_data_zero, tf.expand_dims(y_data, axis=-1)], axis=-1)
    y_target = tf.matmul(x_querry, tf.expand_dims(w, axis=-1))
    choice = tf.random.stateless_uniform([size_distract], seed=new_rng4, minval=0, maxval=c_size, dtype=tf.int32)

    y_data = tf.tensor_scatter_nd_update(y_data, tf.expand_dims(choice, axis=-1),
                                         tf.random.stateless_normal([size_distract, i_size], seed=new_rng3))
    y_target_zero = tf.zeros_like(x_querry[:, :-1])
    y_target = tf.expand_dims(y_target, axis=-1)

    seq = tf.concat([x, y_data], axis=1)
    seq = tf.reshape(seq, [-1, i_size])
    target = tf.concat([y_target_zero, y_target], axis=-1)
    seq = tf.concat([seq, x_querry], axis=0)
    return tf.squeeze(seq), tf.squeeze(target), w

def create_weights(i_size, o_size, c_size, lr, w_init, second_zero=False, lin_diag=False, gd_deq=False, num_layers=1, input_mlp_rnd=None, in_proj=False):
    one = tf.ones([i_size+o_size])
    one_in_size = tf.ones([i_size])
    zero_out_size = tf.zeros([o_size])
    one_out_size = tf.ones([o_size])

    value_upper = tf.zeros([i_size, i_size+o_size])
    value_lower_left = w_init[0]
    if lin_diag:
        value_lower_right = tf.linalg.diag(one_out_size) * -2
    else:
        value_lower_right = tf.linalg.diag(one_out_size) * -1

    if second_zero:
        value_lower_right = tf.linalg.diag(zero_out_size)

    value_lower_part = tf.concat([value_lower_left, value_lower_right], axis=1)
    value_matrix = tf.concat([value_upper, value_lower_part], axis=0)
    if lin_diag:
        value_matrix += tf.linalg.diag(one)

    query_upper_part = tf.zeros([o_size, i_size+o_size])
    query_lower_left = tf.linalg.diag(one_in_size)
    query_lower_right = tf.zeros([i_size, o_size])
    query_lower_part = tf.concat([query_lower_left, query_lower_right], axis=1)
    query_matrix = tf.concat([query_lower_part, query_upper_part], axis=0)
    key_matrix = query_matrix

    projection_upper_part = tf.zeros([i_size, i_size+o_size])
    projection_lower_left = tf.zeros([o_size, i_size])
    projection_lower_right = tf.linalg.diag(one_out_size) * ((1/c_size) * lr)

    if lin_diag:
        shifted_lr = tf.linalg.diag(one_out_size) * ((1/c_size) * (1/c_size) * lr)
        projection_lower_right += shifted_lr

    projection_lower_part = tf.concat([projection_lower_left, projection_lower_right], axis=1)
    projection_matrix = tf.concat([projection_upper_part, projection_lower_part], axis=0)
    if lin_diag:
        projection_matrix -= tf.linalg.diag(one) * ((1/c_size) * (1/c_size) * lr)

    params_new = {}
    for l in range(num_layers):
        if num_layers == 1 or gd_deq:
            tra_name = 'Transformer_gd/multi_head_attention/'
        else:
            tra_name = 'Transformer_gd/~trans_block/layer_'+str(l)+'/'
        params_new[tra_name + 'query'] = {'w': query_matrix}
        params_new[tra_name + 'value'] = {'w': value_matrix}
        params_new[tra_name + 'key'] = {'w': key_matrix}
        params_new[tra_name + 'linear'] = {'w': projection_matrix}

    if in_proj:
        rng1, rng2, rng3 = tf.random.experimental.stateless_split(input_mlp_rnd, 3)
        w_embedding = tf.random.stateless_normal([11, 11], seed=rng1) * tf.sqrt(0.002/11)
        params_new['Transformer_gd/emb'] = {'w': w_embedding}
    elif input_mlp_rnd is not None:
        rng1, rng2, rng3 = tf.random.experimental.stateless_split(input_mlp_rnd, 3)
        w1 = tf.random.stateless_normal([40, 160], seed=rng1) * tf.sqrt(0.002/2)
        w2 = tf.random.stateless_normal([160, 40], seed=rng2) * tf.sqrt(0.002/40)
        b1 = tf.random.stateless_normal([160], seed=rng1) * 0
        b2 = tf.random.stateless_normal([40], seed=rng2) * 0
        w_embedding = tf.random.stateless_normal([2, 40], seed=rng1) * tf.sqrt(0.002/2)
        params_new['Transformer_gd/input_mlp/linear'] = {'w': w1, 'b': b1}
        params_new['Transformer_gd/input_mlp/linear_1'] = {'w': w2, 'b': b2}
        params_new['Transformer_gd/emb'] = {'w': w_embedding}
  
    return params_new

create_weights(10, 1, 10, 0.1, tf.ones([1, 1, 10])*0.1)
