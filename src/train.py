from functools import partial
from typing import Any, MutableMapping, NamedTuple, Tuple

from absl import app
from absl import flags
from datetime import datetime

import numpy as np
import tensorflow as tf

from transformer import Transformer
from data import (create_reg_data, create_reg_data_sin, 
                  create_reg_data_classic_token, 
                  create_reg_data_sin_test, 
                  create_ood_data, create_weights)
from config import config

from datetime import datetime, timezone
import pytz
cet = pytz.timezone('CET')

file_time = str(datetime.now(tz=cet))

data_creator = tf.vectorized_map(lambda args: create_reg_data(*args))

data_creator_ood = tf.vectorized_map(lambda args: create_ood_data(*args))

data_creator_sin_test = tf.vectorized_map(lambda args: create_reg_data_sin_test(*args))

class TrainState(NamedTuple):
    """Container for the training state."""
    params: Any
    opt_state: Any
    rng: tf.Tensor
    step: tf.Tensor

class TestState(NamedTuple):
    """Container for the test state."""
    prediction: tf.Tensor
    inter_losses: tf.Tensor
    test_loss: tf.Tensor
    rng: tf.Tensor
    step: tf.Tensor

class DataState(NamedTuple):
    """Container for the data state."""
    train_data: tf.Tensor
    test_data: tf.Tensor
    rng: tf.Tensor
    step: tf.Tensor

_Metrics = MutableMapping[str, Any]

def change_dataloader():
    global data_creator
    if config.classic_token_const:
        data_creator = tf.vectorized_map(lambda args: create_reg_data_classic_token(*args))

    if config.non_linear_reg_task:
        data_creator = tf.vectorized_map(lambda args: create_reg_data_sin(*args))

def forward(tokens: tf.Tensor, is_training: bool, gd: bool):
    """Transformer forward."""
    if config.classic_token_const:
        in_context_length = config.dataset_size*2 + 1
    else:
        in_context_length = config.dataset_size + 1
    tr = Transformer(
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        widening_factor=config.widening_factor,
        key_size=config.key_size,
        embedding_size=config.emb_size,
        only_attention=config.att_only_trans,
        in_context_length=in_context_length,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate,
        use_pe=config.pos_enc,
        pe_size=config.pos_enc_size,
        concat_pe=config.concat_pos_enc,
        output_mapping=config.out_proj,
        input_mapping=config.in_proj,
        use_layer_norm=config.layer_norm,
        use_bias_p=config.use_bias,
        deq=config.deq,
        y_update=config.y_update,
        use_softmax=config.use_softmax,
        use_non_lin_mix=config.use_non_lin_mix,
        first_layer_sm=config.first_layer_sm,
        zero_embeddings=config.zero_pos_enc,
        init_scale=config.init_scale,
        input_mlp=config.input_mlp,
        input_mlp_out_dim=config.input_mlp_out_dim,
        sum_norm=config.sum_norm,
        dampening=config.dampening,
        clip=config.clip,
        ana_copy=config.ana_copy
    )

    tr_gd = Transformer(
        num_heads=1,
        num_layers=config.num_layers,
        key_size=config.key_size,
        embedding_size=config.emb_size,
        widening_factor=config.widening_factor,
        only_attention=True,
        in_context_length=in_context_length,
        output_size=config.output_size,
        dropout_rate=0,
        use_pe=False,
        pe_size=0,
        concat_pe=False,
        output_mapping=False,
        input_mapping=config.in_proj,
        use_layer_norm=False,
        use_bias_p=False,
        deq=config.gd_deq,
        use_softmax=False,
        zero_embeddings=False,
        y_update=config.y_update,
        sum_norm=False,
        input_mlp=config.input_mlp,
        input_mlp_out_dim=config.input_mlp_out_dim,
        gd_mlp_config=True,
        init_scale=0.02,
        dampening=config.gd_dampening,
        clip=config.clip,
        name='Transformer_gd'
    )

    if not gd:
        return tr(tokens, is_training=is_training, predict_test=False)
    else:
        return tr_gd(tokens, is_training=is_training, predict_test=False)

def compute_loss(preds, targets):
    assert preds.shape == targets.shape
    return 0.5 * tf.reduce_sum((targets - preds)**2) / targets.shape[0]

@tf.function
def loss_fn(data: tf.Tensor, gd) -> tf.Tensor:
    """Computes the MSE loss between targets and predictions."""
    preds, _, _ = forward(data[0], True, gd)
    targets = data[1][:, -1]
    preds = preds[:, -1, -1] * (-1.0)
    return compute_loss(preds, targets)

@tf.function
def predict(data: tf.Tensor, gd) -> tf.Tensor:
    """Predict."""
    preds, _, _ = forward(data, False, gd)
    return preds

@tf.function
def predict_stack(data: tf.Tensor, gd) -> tf.Tensor:
    """Predict and return stack."""
    _, stack, _ = forward(data, False, gd)
    return stack

@tf.function
def predict_attn(data: tf.Tensor, gd) -> tf.Tensor:
    """Predict and return stack."""
    _, _, attn = forward(data, False, gd)
    return attn

@tf.function
def predict_test(data: tf.Tensor, gd) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Predict test data used for analyses as well as metrics computation."""
    preds, pred_stack, _ = forward(data[0], False, gd)
    targets = data[1][:, -1]
    preds = preds[:, -1, -1] * (-1.0)
    loss_final = compute_loss(preds, targets)
    loss_f = lambda x: compute_loss(x, targets)
    if not config.ana_copy:
        losses = tf.vectorized_map(loss_f, tf.convert_to_tensor(pred_stack))
    else:
        losses = []
    return loss_final, pred_stack, losses

@tf.function
def update(state: TrainState, data, optimiser, gd=False) -> Tuple[TrainState, _Metrics]:
    """Does an SGD step and returns training state as well as metrics."""
    rng, new_rng = tf.split(state.rng, num_or_size_splits=2)
    loss, gradients = tf.gradients(loss_fn(data, gd), state.params)
    updates, new_opt_state = optimiser.apply_gradients(zip(gradients, state.params), state.opt_state)

    new_state = TrainState(
        params=state.params,
        opt_state=new_opt_state,
        rng=new_rng,
        step=state.step + 1,
    )

    metrics = {
        'step': state.step,
        'train_loss': loss,
    }
    return new_state, metrics

@tf.function
def evaluation(train_state: TrainState, test_state: TestState, data, gd) -> TestState:
    """Compute predictions from model."""
    rng, new_rng = tf.split(test_state.rng, num_or_size_splits=2)
    loss, preds, inter_losses = predict_test(data, gd)
    new_state = TestState(
        prediction=preds,
        inter_losses=inter_losses,
        test_loss=loss,
        rng=new_rng,
        step=test_state.step + 1,
    )
    return new_state

def init_model(rng, train_data, test_data, optimiser) -> TrainState:
    """Init tensorflow modules to create train and test state."""
    train_rng, test_rng = tf.split(rng, num_or_size_splits=2)
    initial_params = loss_fn(train_data, gd=False)

    if config.analyse:
        initial_params_gd = loss_fn(train_data, gd=True)
        _, _, _ = predict_test(test_data, True)

    initial_test_loss, initial_preds, i_inter_losses = predict_test(test_data, False)
    _ = predict(test_data[0], False)
    _ = predict_stack(test_data[0], False)

    initial_opt_state = optimiser.init()

    return TrainState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng=train_rng,
        step=tf.constant(0)), TestState(
        prediction=initial_preds,
        inter_losses=i_inter_losses,
        test_loss=initial_test_loss,
        rng=test_rng,
        step=tf.constant(0))

def init():
    """Init data creator, model, optimizer, etc."""
    rng = tf.random.uniform([2], dtype=tf.int32)
    rng, train_rng = tf.split(rng, num_or_size_splits=2)

    train_data = data_creator(tf.random.uniform([config.bs, 2], dtype=tf.int32),
                              config.input_size,
                              config.dataset_size,
                              config.size_distract,
                              config.input_range,
                              config.weight_scale)

    lr = config.lr
    if config.adam:
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimiser = tf.keras.optimizers.SGD(learning_rate=lr)

    train_state, test_state = init_model(rng, train_data, train_data, optimiser)
    return optimiser, train_state, test_state, rng

@tf.function
def analyse_copy(data, state, rng):
    """Analyse copying behaviour of the first layer of Transformer."""
    own, own_plus_1, other = 0, 0, 0
    len_ana = config.dataset_size*2 -1
    for k in range(0, len_ana):
        sum_over_od = lambda x: tf.reduce_sum(predict_stack(state.params, rng, tf.expand_dims(x, axis=0), False)[0][0, k, :])
        grads = tf.gradients(sum_over_od, data[0])
        grads_wrt_inputs = tf.norm(tf.reduce_mean(grads, axis=0), axis=-1)
        own += grads_wrt_inputs[k]
        own_plus_1 += grads_wrt_inputs[k+1]
        other += (tf.reduce_sum(grads_wrt_inputs[:k]) +
                  tf.reduce_sum(grads_wrt_inputs[k+1:])) / tf.cast(grads_wrt_inputs[:-2].shape[0], tf.float32)
    return own / len_ana, own_plus_1 / len_ana, other / len_ana

@tf.function
def analyse_gd(data, state, rng):
    """Analyse prediction sensitivity wrt output."""
    loss_grad = lambda x: predict_test(state.params, rng, x, False)
    grads_wrt_loss = tf.vectorized_map(tf.gradients, data[0])

    out_grad = lambda x: tf.reduce_sum(predict(state.params, rng, x, False))
    grads_wrt_out = tf.vectorized_map(tf.gradients, data[0])

    return grads_wrt_loss, grads_wrt_out

@tf.function
def analyse(data, state, rng, params_constructed):
    """Analyse alignment between GD and trained Transformer."""
    pred = lambda z: predict(state.params, rng, tf.expand_dims(z, axis=0), False)[0, -1, -1]
    grads = tf.vectorized_map(tf.gradients, data[0])[:, -1, :-1]
    predictions = tf.vectorized_map(pred, data[0])

    grads_norm = tf.norm(grads, axis=1)

    pred_c = lambda z: predict(params_constructed, rng, tf.expand_dims(z, axis=0), True)[0, -1, -1]
    grads_c = tf.vectorized_map(tf.gradients, data[0])[:, -1, :-1]
    predictions_c = tf.vectorized_map(pred_c, data[0])
    grads_c_norm = tf.norm(grads_c, axis=1)

    dot_products = tf.reduce_mean(tf.einsum('ij,ij->i', grads / (grads_norm[..., None] + 1e-8),
                                            grads_c / (grads_c_norm[..., None] + 1e-8)))
    norm = tf.reduce_mean(tf.norm(grads - grads_c, axis=1))
    pred_norm = tf.reduce_mean(tf.norm(predictions[..., None] - predictions_c[..., None], axis=1))
    return dot_products, norm, pred_norm

@tf.function
def interpolate(data, state, rng, params_constructed):
    """Analyse alignment between GD and trained Transformer."""
    pred = lambda z: predict(state.params, rng, tf.expand_dims(z, axis=0), False)[0, -1, -1]
    grads = tf.vectorized_map(tf.gradients, data[0])[:, -1, :]
    predictions = tf.vectorized_map(pred, data[0])
    grads_norm = tf.norm(grads, axis=1)

    pred_c = lambda z: predict(params_constructed, rng, tf.expand_dims(z, axis=0), True)[0, -1, -1]
    grads_c = tf.vectorized_map(tf.gradients, data[0])[:, -1, :]
    predictions_c = tf.vectorized_map(pred_c, data[0])
    grads_c_norm = tf.norm(grads_c, axis=1)

    dot_products = tf.reduce_mean(tf.einsum('ij,ij->i', grads / (grads_norm[..., None] + 1e-8),
                                            grads_c / (grads_c_norm[..., None] + 1e-8)))
    norm = tf.reduce_mean(tf.norm(grads - grads_c, axis=1))
    pred_norm = tf.reduce_mean(tf.norm(predictions[..., None] - predictions_c[..., None], axis=1))
    return dot_products, norm, pred_norm

def compute_other_d_loss(ir, ws, rng, params, gd, bs_size=500):
    """Compute loss on large OOD dataset."""
    data_ood = data_creator_ood(tf.random.uniform([bs_size, 2], dtype=tf.int32),
                                config.input_size,
                                config.dataset_size,
                                ir, ws)

    loss_ood, _, _ = predict_test(params, rng, data_ood, gd)
    return loss_ood

def compute_ood_loss(ir, ws, rng, params, gd, bs_size=10000):
    """Compute loss on large dataset with potential scaling."""
    data = data_creator(tf.random.uniform([bs_size, 2], dtype=tf.int32),
                        config.input_size,
                        config.dataset_size,
                        config.size_distract,
                        ir,
                        ws)
    loss, _, _ = predict_test(params, rng, data, gd)
    return loss

def noisy_data_ana(state, rng, params_c, bs_size=10000):
    """Analyse alignment between GD and trained Transformer on OOD settings."""
    loss = []
    loss_gd = []
    for num_dis in range(0, config.dataset_size, 2):
        disturb_data = data_creator(tf.random.uniform([bs_size, 2], dtype=tf.int32),
                                    config.input_size,
                                    config.dataset_size,
                                    num_dis,
                                    config.input_range,
                                    config.weight_scale)
        loss.append(predict_test(state.params, rng, disturb_data, False)[0])
        loss_gd.append(predict_test(params_c, rng, disturb_data, True)[0])
    return loss, loss_gd

@tf.function
def ood(state, rng, params_c, bs_size):
    """Analyse alignment between GD and trained Transformer on OOD settings."""
    stretch = np.arange(0.5, 5+0.1, 0.1)
    stretch_i = np.arange(0.5, 2+0.03, 0.03)
    eval_ir = lambda ir: compute_ood_loss(ir, config.weight_scale, rng,
                                          state.params, False, bs_size)
    eval_ws = lambda ws: compute_ood_loss(config.input_range, ws, rng, 
                                          state.params, False, bs_size)
    eval_ir_c = lambda ir: compute_ood_loss(ir, config.weight_scale, rng,
                                            params_c, True, bs_size)
    eval_ws_c = lambda ws: compute_ood_loss(config.input_range, ws, rng,
                                            params_c, True, bs_size)

    return (tf.vectorized_map(eval_ir, stretch_i), tf.vectorized_map(eval_ws, stretch),
            tf.vectorized_map(eval_ir_c, stretch_i), tf.vectorized_map(eval_ws_c, stretch), stretch)

@tf.function
def ood_other_d(state, rng, params_c):
    """Analyse alignment between GD and trained Transformer on more OOD."""
    stretch = np.arange(0.5, 5+0.1, 0.1)
    stretch_i = np.arange(0.5, 5+0.05, 0.05)
    eval_ir = lambda ir: compute_other_d_loss(ir, config.weight_scale, rng,
                                              state.params, False)
    eval_ws = lambda ws: compute_other_d_loss(config.input_range, ws, rng, 
                                              state.params, False)
    eval_ir_c = lambda ir: compute_other_d_loss(ir, config.weight_scale, rng,
                                                params_c, True)
    eval_ws_c = lambda ws: compute_other_d_loss(config.input_range, ws, rng,
                                                params_c, True)

    return (tf.vectorized_map(eval_ir, stretch_i), tf.vectorized_map(eval_ws, stretch),
            tf.vectorized_map(eval_ir_c, stretch_i), tf.vectorized_map(eval_ws_c, stretch), stretch)

def scan_lrs(rng, lin_diag=False, bs=10000):
    """Simple brute force search for optimal gradient descent lr on 10k tasks."""
    lr_scan_range = np.arange(0.001, 25, 0.1)

    weights = lambda lr: create_weights(config.input_size, 1,
                                        config.dataset_size, lr,
                                        tf.ones([1, 1, config.input_size]) * 0.0,
                                        lin_diag=lin_diag,
                                        gd_deq=config.gd_deq,
                                        num_layers=config.num_layers,
                                        input_mlp_rnd=rng if (config.input_mlp or config.in_proj) else None,
                                        in_proj=config.in_proj)
    eval_lr = lambda lr: compute_ood_loss(config.input_range,
                                          config.weight_scale, rng,
                                          weights(lr), True, bs)

    losses_lr = []
    for lr in lr_scan_range:
        losses_lr.append(eval_lr(lr))
    losses_lr = tf.convert_to_tensor(losses_lr)
    lr_min_i = tf.argmin(losses_lr)
    min_loss = tf.reduce_min(losses_lr)
    return lr_scan_range[lr_min_i], min_loss

def test_sin(params, rng, gd):
    rng, test_rng = tf.split(rng, num_or_size_splits=2)
    eval_data = data_creator_sin_test(tf.random.uniform([100, 2], dtype=tf.int32),
                                      rng,
                                      config.dataset_size,
                                      config.input_range,
                                      config.weight_scale)
    _, preds, _ = predict_test(params, rng, eval_data, gd)
    return preds, eval_data

def xm_metric_tracking(writer, metric_name, metric_value, step):
    """Metric tracking."""
    if not config.local_usage:
        measurements = work_unit.get_measurement_series(label=metric_name)
        measurements.create_measurement(objective_value=metric_value, step=step)
        writer.write_scalars(step, {metric_name: metric_value})

@tf.function
def gradient_manipulation(gradients, ndim):
    """Manipulates gradients of gradient descent."""
    update_matrix = np.eye(ndim, dtype=bool)
    indx = np.where(~update_matrix)
    aug_gradients = {}
    for param in gradients:
        if config.input_mlp and 'mlp' in param:
            aug_gradients[param] = gradients[param]
        elif (config.input_mlp or config.in_proj) and 'emb' in param:
            aug_gradients[param] = gradients[param]
        else:
            if config.train_gd_whitening and ('linear' in param or 'value' in param):
                gradients[param]['w'] = tf.tensor_scatter_nd_update(gradients[param]['w'], indx, 0)
                sca = tf.eye(ndim-1) * tf.reduce_mean(gradients[param]['w'][:-1, :-1])
                gradients[param]['w'] = tf.tensor_scatter_nd_update(gradients[param]['w'], tf.slice(None, None, [-1, -1]), sca)
                aug_gradients[param] = gradients[param]
            else:
                aug_gradients[param] = {'w': tf.zeros_like(gradients[param]['w'])}

    return aug_gradients

def pre_train_gd_hps(eval_rng, params_gd):
    """Pre training gd hps such as P i.e. gradient modulation matrix."""
    optimiser = tf.keras.optimizers.Adam(learning_rate=config.gd_lr)
    opt_state = optimiser.init()

    eval_data = data_creator(tf.random.uniform([config.bs, 2], dtype=tf.int32),
                             config.input_size,
                             config.dataset_size,
                             config.size_distract,
                             config.input_range,
                             config.weight_scale)
    data_rng, rng = tf.split(eval_rng, num_or_size_splits=2)
    gd_losses = []
    for step in range(config.training_steps_gd):
        data_rng, rng = tf.split(rng, num_or_size_splits=2)
        data = data_creator(tf.random.uniform([config.bs_gd_train, 2], dtype=tf.int32),
                            config.input_size,
                            config.dataset_size,
                            config.size_distract,
                            config.input_range,
                            config.weight_scale)
        loss, gradients = tf.gradients(loss_fn(data, True), params_gd)
        if step % 100 == 0:
            losses_gd, _, _ = predict_test(params_gd, eval_rng, eval_data, True)
            if not config.non_linear_reg_task:
                print('Loss of GD++ (we learn eta and gamma): ', step, losses_gd)
            else:
                print('Loss of trained MLP + GD (on the output head): ', step, losses_gd)
        aug_gradients = gradient_manipulation(gradients, config.key_size)
        updates, opt_state = optimiser.apply_gradients(zip(aug_gradients, params_gd), opt_state)
        params_gd = optax.apply_updates(params_gd, updates)
    return params_gd, data_rng

def train(_):
    """Train loop."""
    print("Use notebook to run the code")

if __name__ == '__main__':
    app.run()
