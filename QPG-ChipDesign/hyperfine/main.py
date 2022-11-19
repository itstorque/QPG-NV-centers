import pandas as pd
import tensorflow as tf
import numpy as np
import itertools
import scipy.linalg as la
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import json

def kron(*ops):
    acc = ops[0]
    for o in ops[1:]:
        acc = np.kron(acc,o)
    return acc

def fock_(i,n):
    v = np.zeros(n,dtype=complex)
    v[i] = 1
    return v

def fock(indices,ns):
    acc = fock_(indices[0], ns[0])
    for i,n in zip(indices[1:], ns[1:]): # TODO use kron
        acc = np.kron(acc,fock_(i,n))
    return acc

@tf.function
def normalize(v):
    return v/tf.norm(v)

@tf.function
def batch_normalize(v):
    return v/tf.einsum('ij,ji->j',v,hc(v))

def modes(*ns):
    ms = [
        np.diag(
            np.arange(1,n,dtype=complex)**0.5,
            k=1)
        for n in ns]
    return ms
s1x = np.array([[0,1],
                [1,0]], dtype=complex)
s1y = np.array([[0,1j],
                [-1j,0]], dtype=complex)
s1z = np.array([[1,0],
                [0,-1]], dtype=complex)
s1m = np.array([[0,0],
                [1,0]], dtype=complex)
s1p = np.array([[0,1],
                [0,0]], dtype=complex)



def uplift(ops):
    if len(ops) <= 1:
        return ops
    bigops = []
    for i,o in enumerate(ops):
        #ids = [tf.linalg.LinearOperatorIdentity(n) for n in ns]
        ids = [np.eye(op.shape[0]) for op in ops]
        ids[i] = o
        #kr = tf.linalg.LinearOperatorKronecker(ids)
        acc = np.kron(ids[0],ids[1])
        for oo in ids[2:]: # TODO use kron
            acc = np.kron(acc,oo)
        kr = acc
        bigops.append(kr)
    return bigops


def subdivide_pulses(subdivisions, control_pulses):
    return np.reshape(np.transpose([control_pulses]*subdivisions,(1,2,0)),(2,-1))


def choose_dtype(a):
    if isinstance(a,list):
        return choose_dtype(a[0])
    elif isinstance(a,float):
        return tf.float64
    elif isinstance(a,complex):
        return tf.complex128
    else:
        return tf.complex128 if any(a.dtype==d for d in [np.complex128, np.complex64]) else tf.float64

def to_tf_constant(*arrays):
    return [tf.constant(a,dtype=choose_dtype(a))
            for a in arrays]



def make_projectors(dims):
    projectors = []
    for I in range(len(dims)):
        projector = np.stack([indices[I]**0.5*fock(indices,dims)
                              for indices in itertools.product(*(range(d) for d in dims))])
        projectors.append(projector)
    return projectors

@tf.function
def get_pops(outs,projectors):
    return tf.reduce_sum(tf.abs(tf.einsum('SiH,tHB->SitB',projectors,outs))**2,axis=1)

@tf.function
def get_pops_dm(outs,projectors):
    return tf.math.real(tf.einsum('SiH,Sih,tBHh->StB',projectors,tf.math.conj(projectors),outs))

@tf.function
def get_purity_dm(outs):
    return tf.map_fn( fn=lambda t: tf.linalg.trace(t@t), elems=outs )

@tf.function
def evolve(control_pulses,control_hs,drift_params,drift_hs,in_states,dt):

    z = tf.constant(0.,dtype=tf.dtypes.float64)
    h_THH = (tf.einsum('D,DHh->Hh',tf.complex(drift_params,z),drift_hs)
            +tf.einsum('CT,CHh->THh',tf.complex(control_pulses,z),control_hs))

    u_THH = tf.linalg.expm((-1j*dt)*h_THH)
    v_TH = tf.scan(lambda a, u: tf.linalg.matmul(u, a), u_THH, initializer=tf.transpose(in_states))

    return v_TH

@tf.function
def step_Lindblad_lhs(a, h, lindblad_ops, l_ops_sq, l_params, dt):
    return (- 1j*(tf.linalg.matmul(h, a)-tf.linalg.matmul(a, h))
            + tf.einsum('Lij,Bjk,Llk,L->Bil',
                        lindblad_ops,a,tf.math.conj(lindblad_ops),l_params)
            - 0.5*(tf.einsum('Lij,Bjk,L->Bik',l_ops_sq,a,l_params)
                  +tf.einsum('Lji,Bkj,L->Bki',l_ops_sq,a,l_params))
           )*dt

@tf.function
def step_Lindblad_rk4(a, h, lindblad_ops, l_ops_sq, l_params, dt):
    k1 = step_Lindblad_lhs(a, h, lindblad_ops, l_ops_sq, l_params, dt)
    k2 = step_Lindblad_lhs(a+k1/2., h, lindblad_ops, l_ops_sq, l_params, dt)
    k3 = step_Lindblad_lhs(a+k2/2., h, lindblad_ops, l_ops_sq, l_params, dt)
    k4 = step_Lindblad_lhs(a+k3, h, lindblad_ops, l_ops_sq, l_params, dt)
    return (k1+2*k2+2*k3+k4)/6.

@tf.function
def step_0Lindblad_lhs(a, h, lindblad_ops, l_ops_sq, l_params, dt):
    return (  tf.einsum('Lij,Bjk,Llk,L->Bil',
                        lindblad_ops,a,tf.math.conj(lindblad_ops),l_params)
            - 0.5*(tf.einsum('Lij,Bjk,L->Bik',l_ops_sq,a,l_params)
                  +tf.einsum('Lji,Bkj,L->Bki',l_ops_sq,a,l_params))
           )*dt

@tf.function
def step_0Lindblad_rk4(a, h, lindblad_ops, l_ops_sq, l_params, dt):
    k1 = step_0Lindblad_lhs(a, h, lindblad_ops, l_ops_sq, l_params, dt)
    k2 = step_0Lindblad_lhs(a+k1/2., h, lindblad_ops, l_ops_sq, l_params, dt)
    k3 = step_0Lindblad_lhs(a+k2/2., h, lindblad_ops, l_ops_sq, l_params, dt)
    k4 = step_0Lindblad_lhs(a+k3, h, lindblad_ops, l_ops_sq, l_params, dt)
    return (k1+2*k2+2*k3+k4)/6.

#method = 'mixed_with_RK4'
@tf.function
def evolve_Lindblad(control_pulses,control_hs,drift_params,drift_hs,lindblad_params,lindblad_ops,in_states,dt,method='mixed_with_RK4'):
    z = tf.constant(0.,dtype=tf.dtypes.float64)
    h_THH = (tf.einsum('D,DHh->Hh',tf.complex(drift_params,z),drift_hs)
            +tf.einsum('CT,CHh->THh',tf.complex(control_pulses,z),control_hs))
    l_params = tf.complex(lindblad_params,z)
    l_ops_sq = tf.einsum('Lji,Ljk->Lik',tf.math.conj(lindblad_ops),lindblad_ops)

    if method=='mixed_with_Euler':
        v_THH = tf.scan(lambda a, u: a + step_Lindblad_lhs(a, u, lindblad_ops, l_ops_sq, l_params, dt),
                       h_THH, initializer=in_states)
    elif method=='mixed_with_RK4':
        v_THH = tf.scan(lambda a, u: a + step_Lindblad_rk4(a, u, lindblad_ops, l_ops_sq, l_params, dt),
                       h_THH, initializer=in_states)
    else:
        raise NotImplementedError('Unknown method')
    return v_THH

@tf.function
def run(h_drifts, h_drives, params, drives, initial_states, projectors, DT, METHOD, data):

    if METHOD == "S":

      outs = evolve(drives,h_drives,params,h_drifts,initial_states, DT)
      pop_func = get_pops

    elif METHOD == "L":

      outs = \
      evolve_Lindblad(drives,h_drives,
                      params,h_drifts,
                      np.array([1/np.sqrt(T2), 0.]),lindblad_ops,
                      initial_states,DT)
      pop_func = get_pops_dm

    if data:
        pops = pop_func(outs, projectors)
        if METHOD == "L": purity = get_purity_dm(outs)
        else: purity = tf.constant(np.array([])) # not implemented yet
        excitation_prob = tf.abs(pops[0, -1])
        return excitation_prob, pops, purity, outs

    pops = pop_func(outs[-1:], projectors)
    excitation_prob = tf.abs(pops[0, -1])

    return excitation_prob, pops

def generate_constants(omega0p, omega_drive, Bxpt, Bypt, METHOD="S"):

    if METHOD=="S": initial_states = [fock([0],[2])]

    elif METHOD=="L":
      initial_states_q = fock([0],[2])
      initial_states_q = np.outer(initial_states_q,initial_states_q)
      initial_states = [initial_states_q]

    else: print("INVALID METHOD?")

    projectors = np.array(make_projectors([2]))

    initial_states, projectors = to_tf_constant(initial_states, projectors)

    srvI = [ 0.5*s1x, 0.5*s1y, 0]
    srvQ = [-0.5*s1y, 0.5*s1x, 0]

    h_drift = np.array([ 0.5 * hbar * (omega0p - omega_drive) * s1z ])
    h_drive = np.array([ 0.5 * hbar * gamma * (Bxpt * srvI[0] + Bypt * srvI[1]),
                         0.5 * hbar * gamma * (Bxpt * srvQ[0] + Bypt * srvQ[1])])

    h_drift = tf.constant(h_drift, dtype=choose_dtype(h_drift))
    h_drive = tf.constant(h_drive, dtype=choose_dtype(h_drive))

    params_drift = np.ones(1)

    params_drift = tf.constant(params_drift, dtype=choose_dtype(params_drift))

    return initial_states, projectors, h_drift, h_drive, params_drift

@tf.function
def loss_func(initial_states, projectors, h_drift, h_drive, params_drift, I, Q, DT, resolution=1, METHOD="S", fft=False, data=False):

  if not fft:
      params_drives = tf.stack([tf.repeat(I, resolution), tf.repeat(Q, resolution)], axis=0)
  else:
      # upsampling should happen BEFORE ifft by adding zero freq bins
      res_pad = (resolution-1)*tf.size(I)/2 # half affixed and half prefixed
      freqs = tf.signal.ifftshift( tf.pad( tf.signal.fftshift(tf.dtypes.complex(1/tf.size(I)*I, 1/tf.size(Q)*Q)) , [[res_pad, res_pad]]) )
      encoded = tf.signal.ifft( freqs )
      params_drives = tf.stack([tf.math.real(encoded) * fft_window_cnst, tf.math.imag(encoded) * fft_window_cnst], axis=0)
      # if fft_window and tf.size(encoded)==tf.size(fft_window_cnst):
      #     params_drives = tf.stack([tf.math.real(encoded) * fft_window_cnst, tf.math.imag(encoded) * fft_window_cnst], axis=0)
      # else:
      #     fft_window_cnst = get_window_function(tf.size(encoded))
      #     params_drives = tf.stack([tf.math.real(encoded) * fft_window_cnst, tf.math.imag(encoded) * fft_window_cnst], axis=0)

  if data: return run(h_drift, h_drive, params_drift, params_drives, initial_states, projectors, DT, METHOD, data=True)

  loss = 1 - run(h_drift, h_drive, params_drift, params_drives, initial_states, projectors, DT, METHOD, data=False)[0]

  return loss

def IQ_initial_guess(size, guess_size=None, opt_steps=None, LR=None, loss_metric=None, bounds_at_zero=False, df_data={}):

    if (None in [LR, loss_metric, opt_steps, guess_size]) or guess_size > size:
        guess_size = size
        opt_steps = 0

    if guess_size < 3: 
        Ig, Qg = tf.constant([1.], name='I', dtype=choose_dtype(np.array([1.0]))), tf.constant([0.], name='Q', dtype=choose_dtype(np.array([1.0])))
        df_data["guess:I"].append(Ig.numpy())
        df_data["guess:Q"].append(Qg.numpy())
        df_data["guess:loss"].append(None)
        return tf.Variable(Ig), tf.Variable(Qg)

    Ig = np.random.normal(loc=0.0, scale=0.1, size=guess_size-2 if bounds_at_zero else guess_size)
    Qg = np.random.normal(loc=0.0, scale=0.1, size=guess_size-2 if bounds_at_zero else guess_size)

    if bounds_at_zero:
        Ig, Qg = np.insert(np.insert(Ig, 0, 0), guess_size-1, 0), np.insert(np.insert(Qg, 0, 0), guess_size-1, 0)

    Ig, Qg = tf.constant(Ig, dtype=choose_dtype(Ig)), tf.constant(Qg, dtype=choose_dtype(Qg))

    Ig, Qg = tf.Variable(Ig), tf.Variable(Qg)

    variables = [Ig, Qg]

    opt = tf.keras.optimizers.Nadam(learning_rate=LR)

    loss = tf.constant(0.0)

    for i in range(int(opt_steps)):

      with tf.GradientTape() as tape:
          loss = loss_metric(Ig, Qg)

      grads = tape.gradient(loss, variables)
      opt.apply_gradients(zip(grads, variables))

    if guess_size != size:
        # eventually we want to remove the [*guess_size/size] term when we incorporate T2* etc.
        I, Q = tf.repeat(Ig, np.ceil(size/guess_size))[:size], tf.repeat(Qg, np.ceil(size/guess_size))[:size]
        I, Q = I*guess_size/size, Q*guess_size/size
    else:
        I, Q = Ig, Qg

    I, Q = tf.constant(I, name='I', dtype=choose_dtype(I)), tf.constant(Q, name='Q', dtype=choose_dtype(Q))

    I, Q = tf.Variable(I), tf.Variable(Q)

    df_data["guess:I"].append(I.numpy())
    df_data["guess:Q"].append(Q.numpy())
    df_data["guess:loss"].append(loss.numpy().flatten())

    return I, Q

def get_window_function(size):

    if fft_window == None:
        fft_window_cnst = tf.constant(1., dtype=tf.complex128)
    elif fft_window == "hann":
        fft_window_cnst = tf.signal.hann_window(size, dtype=tf.float64)
        fft_window_cnst = tf.constant(fft_window_cnst, dtype=tf.float64)
    else:
        print('fft_window input invalid')
        raise TypeError

    return fft_window_cnst

if __name__ == "__main__":

    print("main.py > ")

    sys.stdout.flush()

    # GLOBALLY REUSED

    lindblad_ops = np.array(uplift([s1z])+uplift([s1m]))
    lindblad_ops = tf.constant(lindblad_ops)

    #

    tf.get_logger().setLevel('INFO') # 'ERROR'

    hbar = 1
    gamma = 1. #to make math easier, make 1 #2*np.pi*28 # unit GHz/T

    # qubits = 6
    target_qubit = 1
    diff_factor, diff_target = 200, None
    plot_res = 200
    METHOD = 'S'
    FFT = False

    OPT_LIM = 1e-4
    OPT_STEPS = 0

    GUESS_OPT_STEPS = None
    GUESS_SIZE = None

    REG_PULSE = False
    MAX_SLOPE = 0.05

    omega0p, omega_drive = 1, 2.2 #2.2

    data = json.loads(sys.argv[1])

    for key, value in data.items():
      vars()[key] = value

    df_data = {"final_pops": [], "pops": [], "T": [], "LR": [], "loss": [], "I": [], "Q": [], "OPT_STEPS": [], "distance": [], "loss_hist": [], "iter_step": [], "STEPS": [], "purity": [], "fft:I": [], "fft:Q": [], "guess:I": [], "guess:Q": [], "guess:loss": [], "pulse_loss_hist": [], "target_loss_hist": []}

    if qubits > 1: df_data["final_pops2"] = []; df_data["pops2"] = []; df_data["purity2"] = [];
    if qubits > 2: df_data["final_pops3"] = []; df_data["pops3"] = []; df_data["purity3"] = [];
    if qubits > 3: df_data["final_pops4"] = []; df_data["pops4"] = []; df_data["purity4"] = [];
    if qubits > 4: df_data["final_pops5"] = []; df_data["pops5"] = []; df_data["purity5"] = [];
    if qubits > 5: df_data["final_pops6"] = []; df_data["pops6"] = []; df_data["purity6"] = [];
    if qubits > 6: df_data["final_pops7"] = []; df_data["pops7"] = []; df_data["purity7"] = [];

    df_data['data'] = sys.argv[1]

    df_data['odmra'] = []
    df_data['odmrb'] = []
    df_data['odmr_domain'] = []

    df_data['odmra1'] = []
    df_data['odmra2'] = []
    df_data['odmra3'] = []

    df_data['units'] = []

    folder = sys.argv[-2]

    sim_id = "sim_" + sys.argv[-1]

    step_space = [steps]
    d_space = [d]

    t_space = [T]
    lr_array = [LR]

    # B = 1./gamma

    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE
    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE
    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE

    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE
    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE
    # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE
    B = CUSTOM_B/gamma # TEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPETEMPTMEPMTEPMTPEMTPE


    print(B, gamma)

    if FFT: fft_window_cnst = get_window_function(steps*diff_factor)

    print(f'began running {sim_id}')

    sys.stdout.flush()

    STEPS = steps

    if diff_target != None:
        # diff_factor = round(T /STEPS /diff_target)
        diff_factor = round(1 /STEPS *diff_target)
        if diff_factor < 1: diff_factor = 1

    DT = T /STEPS /diff_factor

    # omega_drive = omega0p/((target_qubit-1)*d+1)

    # omega_drive = CUSTOM_splitting+2.2

    print(CUSTOM_ground_splitting, CUSTOM_fine_splitting)

    # def gen_sim_params(omega_drive):

    #     sim_params                 = generate_constants(CUSTOM_ground_splitting+0 , omega_drive, B, 0, METHOD)
    #     sim_params_2               = generate_constants(CUSTOM_ground_splitting+CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)
    #     sim_params_3               = generate_constants(CUSTOM_ground_splitting+2*CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)

    #     sim_params_4               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+0. , omega_drive, B, 0, METHOD)
    #     sim_params_5               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)
    #     sim_params_6               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+2*CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)

    #     return sim_params, sim_params_2, sim_params_3, sim_params_4, sim_params_5, sim_params_6

    # sim_params, sim_params_2, sim_params_3, sim_params_4, sim_params_5, sim_params_6 = gen_sim_params(omega_drive)

    sim_params                 = generate_constants(CUSTOM_ground_splitting+0 , omega_drive, B, 0, METHOD)
    sim_params_2               = generate_constants(CUSTOM_ground_splitting+CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)
    sim_params_3               = generate_constants(CUSTOM_ground_splitting+2*CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)

    sim_params_4               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+0. , omega_drive, B, 0, METHOD)
    sim_params_5               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)
    sim_params_6               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+2*CUSTOM_fine_splitting, omega_drive, B, 0, METHOD)


    loss_func_params = {"DT": DT, "resolution": diff_factor, "METHOD": METHOD, "fft": FFT}

    all_sim_params = [None, sim_params, sim_params_2, sim_params_3,
                        sim_params_4, sim_params_5, sim_params_6]

    @tf.function
    def total_loss(I, Q):

        # target_qubits = {4, 5, 6}

        flf = ( lambda x, y: (1 - x) if y in target_qubits else x )

        # loss = 0

        # for i in range(1, qubits+1):
            # print(loss)
            # print(flf(0, i), i)
            # loss += flf(loss_func(*all_sim_params[i], I, Q, **loss_func_params), i)

        # loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2) + flf(loss_func(*sim_params_3, I=I, Q=Q, **loss_func_params), 3) + flf(loss_func(*sim_params_4, I=I, Q=Q, **loss_func_params), 4) + flf(loss_func(*sim_params_5, I=I, Q=Q, **loss_func_params), 5) + flf(loss_func(*sim_params_6, I=I, Q=Q, **loss_func_params), 6)

        # return flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2)
        # return flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2) + flf(loss_func(*sim_params_3, I=I, Q=Q, **loss_func_params), 3) + flf(loss_func(*sim_params_4, I=I, Q=Q, **loss_func_params), 4) + flf(loss_func(*sim_params_5, I=I, Q=Q, **loss_func_params), 5) + flf(loss_func(*sim_params_6, I=I, Q=Q, **loss_func_params), 6)
        # return flf(loss_func(*sim_params_2, I, Q, **loss_func_params), 2)

        # return loss

        return loss_func(*sim_params, I, Q, **loss_func_params) + 1 - loss_func(*sim_params_2, I, Q, **loss_func_params)

    @tf.function
    def reg_loss(I, Q):

        loss = tf.constant(0., dtype=tf.double)

        if REG_PULSE == False or tf.size(I) < 3: return loss

        if FFT: raise NotImplementedError # mean to 0th freq and some other stuff

        # check magnitude by avg of norm
        # norm = tf.reduce_mean( ( abs(I)**2 + abs(Q)**2 )**0.5 )
        # loss += norm
        # print("norm =", norm)

        shift_from_zero = (abs(I[0]) + abs(I[-1]) + abs(Q[0]) + abs(Q[-1])) #/ norm
        loss += shift_from_zero

        # # linear w/ boundaries at 0
        # # loss += shift_from_zero if norm != 0 else 0
        # # print("shift_from_zero =", shift_from_zero)

        # # additional factor if boundary not 0 # THIS CAUSES NAN, change with log curve or something
        # # if I[0] != 0 or I[-1] != 0 or Q[0] != 0 or Q[-1] != 0:
        # #     loss += tf.math.sigmoid(-10 * (shift_from_zero) + 5)
        # #     print("sigmoid_component =", tf.math.sigmoid(-10 * (shift_from_zero) + 5))

        # # check slope to ensure smoothness
        slope = tf.reduce_mean( abs(I[1:]-I[:-1])**2 ) + tf.reduce_mean( abs(Q[1:]-Q[:-1])**2 ) #max(abs(I[1:]-I[:-1]))
        # # if slope > MAX_SLOPE: loss += slope
        loss += slope
        # print('   total reg >', loss)
        return loss

    I, Q = IQ_initial_guess(size=STEPS, opt_steps=GUESS_OPT_STEPS, guess_size=GUESS_SIZE, LR=LR, loss_metric=total_loss, bounds_at_zero=REG_PULSE, df_data=df_data)
    # loss_metric = lambda I, Q: total_loss(I, Q) + reg_loss(I, Q) if reg wanted for guess

#     I = tf.constant(
#         [ 6.37284246e-06, 7.93067836e-01, 1.22940721e+00, 1.43343548e+00
# , 1.51674370e+00, 1.53407715e+00, 1.52114473e+00, 1.49453489e+00
# , 1.46622591e+00, 1.43656845e+00, 1.40669226e+00, 1.38344798e+00
# , 1.37940418e+00, 1.38548872e+00, 1.38848234e+00, 1.38404596e+00
# , 1.37620908e+00, 1.37028732e+00, 1.36602678e+00, 1.35742168e+00
# , 1.33850255e+00, 1.31010785e+00, 1.28008601e+00, 1.25773438e+00
# , 1.24543518e+00, 1.23888824e+00, 1.23654201e+00, 1.23232745e+00
# , 1.22229484e+00, 1.20448627e+00, 1.17915910e+00, 1.14828869e+00
# , 1.11469559e+00, 1.08036121e+00, 1.04899193e+00, 1.02025726e+00
# , 9.92943697e-01, 9.65665633e-01, 9.37678677e-01, 9.08941350e-01
# , 8.79721440e-01, 8.50214385e-01, 8.20400552e-01, 7.90093121e-01
# , 7.59012187e-01, 7.26789315e-01, 6.92688758e-01, 6.57257700e-01
# , 6.20059700e-01, 5.80814360e-01, 5.40083214e-01, 4.98154001e-01
# , 4.56154563e-01, 4.15229247e-01, 3.77057326e-01, 3.43565179e-01
# , 3.16127044e-01, 2.90809306e-01, 2.62983445e-01, 2.32088286e-01
# , 2.00027327e-01, 1.69768710e-01, 1.44124862e-01, 1.24918238e-01
# , 1.12537849e-01, 1.05765649e-01, 1.02065653e-01, 9.86047872e-02
# , 9.36206307e-02, 8.71515891e-02, 8.06011933e-02, 7.57186275e-02
# , 7.37508214e-02, 7.49243202e-02, 7.81276727e-02, 8.08256089e-02
# , 7.96272200e-02, 7.20260191e-02, 5.83269387e-02, 4.11746957e-02
# , 2.34285336e-02, 6.70610787e-03,-8.93593075e-03,-2.44857731e-02
# ,-4.08037052e-02,-5.72241120e-02,-7.11271921e-02,-7.92190160e-02
# ,-7.93304620e-02,-7.14721210e-02,-5.80838769e-02,-4.36419060e-02
# ,-3.35643712e-02,-3.23335827e-02,-4.12295832e-02,-5.73707597e-02
# ,-7.56198786e-02,-9.15115083e-02,-1.02717693e-01,-1.08918783e-01
# ,-1.10823974e-01,-1.09267630e-01,-1.05089782e-01,-9.93799337e-02
# ,-9.34061959e-02,-8.82886824e-02,-8.49563391e-02,-8.44040768e-02
# ,-8.76369318e-02,-9.49270822e-02,-1.04932268e-01,-1.14880364e-01
# ,-1.21899170e-01,-1.24248040e-01,-1.21745123e-01,-1.15471250e-01
# ,-1.07046688e-01,-9.78884703e-02,-8.88117818e-02,-8.00330336e-02
# ,-7.14646110e-02,-6.31971936e-02,-5.59462479e-02,-5.11285396e-02
# ,-5.03287232e-02,-5.42717286e-02,-6.20561023e-02,-7.15050328e-02
# ,-8.03434665e-02,-8.71695798e-02,-9.17609768e-02,-9.48350310e-02
# ,-9.75542218e-02,-1.01101651e-01,-1.06467131e-01,-1.14252217e-01
# ,-1.24359749e-01,-1.35835076e-01,-1.47168451e-01,-1.56851328e-01
# ,-1.63755211e-01,-1.67194211e-01,-1.66813467e-01,-1.62502994e-01
# ,-1.54420924e-01,-1.43060376e-01,-1.29258945e-01,-1.14123618e-01
# ,-9.89210085e-02,-8.49927862e-02,-7.37016043e-02,-6.63428048e-02
# ,-6.39212734e-02,-6.67799205e-02,-7.43082968e-02,-8.50508247e-02
# ,-9.71692122e-02,-1.08886248e-01,-1.18695083e-01,-1.25406760e-01
# ,-1.28191895e-01,-1.26673153e-01,-1.21012500e-01,-1.11920839e-01
# ,-1.00548212e-01,-8.82468418e-02,-7.62731909e-02,-6.55683054e-02
# ,-5.66955073e-02,-4.98611387e-02,-4.49034799e-02,-4.12416280e-02
# ,-3.78970135e-02,-3.37054819e-02,-2.76808727e-02,-1.93361636e-02
# ,-8.83575518e-03, 3.00208614e-03, 1.48580522e-02, 2.53104745e-02
# , 3.34148602e-02, 3.90554946e-02, 4.27657613e-02, 4.52730516e-02
# , 4.71678995e-02, 4.88121063e-02, 5.04043116e-02, 5.20865813e-02
# , 5.39947055e-02, 5.62194708e-02, 5.87303881e-02, 6.13352257e-02
# , 6.37162293e-02, 6.55524689e-02, 6.66959351e-02, 6.73046725e-02
# , 6.78275594e-02, 6.88432356e-02, 7.08709626e-02, 7.42598244e-02
# , 7.91676415e-02, 8.55851945e-02, 9.33712758e-02, 1.02290099e-01
# , 1.12052246e-01, 1.22353338e-01, 1.32897800e-01, 1.43402870e-01
# , 1.53592641e-01, 1.63197285e-01, 1.71966465e-01, 1.79694841e-01
# , 1.86248388e-01, 1.91578270e-01, 1.95716056e-01, 1.98754696e-01
# , 2.00826450e-01, 2.02087951e-01, 2.02715697e-01, 2.02907357e-01
# , 2.02879748e-01, 2.02856833e-01, 2.03049380e-01, 2.03634845e-01
# , 2.04744995e-01, 2.06462729e-01, 2.08825343e-01, 2.11830950e-01
# , 2.15445876e-01, 2.19611966e-01, 2.24253465e-01, 2.29283823e-01
# , 2.34613148e-01, 2.40156812e-01, 2.45844449e-01, 2.51626960e-01
# , 2.57478690e-01, 2.63393758e-01, 2.69378467e-01, 2.75443377e-01
# , 2.81598137e-01, 2.87850423e-01, 2.94208685e-01, 3.00687258e-01
# , 3.07311596e-01, 3.14121048e-01, 3.21166964e-01, 3.28505474e-01
# , 3.36186547e-01, 3.44242454e-01, 3.52678184e-01, 3.61464371e-01
# , 3.70531567e-01, 3.79764523e-01, 3.88996434e-01, 3.98005963e-01
# , 4.06523310e-01, 4.14255129e-01, 4.20938815e-01, 4.26427197e-01
# , 4.30778141e-01, 4.34285949e-01, 4.37397383e-01, 4.40554579e-01
# , 4.44083034e-01, 4.48168669e-01, 4.52881096e-01, 4.58201678e-01
# , 4.64043779e-01, 4.70266915e-01, 4.76689291e-01, 4.83103564e-01
# , 4.89301119e-01, 4.95107262e-01, 5.00418414e-01, 5.05217376e-01
# , 5.09548890e-01, 5.13472681e-01, 5.17037748e-01, 5.20292897e-01
# , 5.23304789e-01, 5.26233323e-01, 5.29599577e-01, 5.33841489e-01
# , 5.38956549e-01, 5.44547546e-01, 5.50124035e-01, 5.55220382e-01
# , 5.59504773e-01, 5.62742463e-01, 5.64978704e-01, 5.66567090e-01
# , 5.67746580e-01, 5.68201336e-01, 5.67084410e-01, 5.63450200e-01
# , 5.56612386e-01, 5.46325432e-01, 5.32942543e-01, 5.17502774e-01
# , 5.01484635e-01, 4.86255180e-01, 4.72687509e-01, 4.61202942e-01
# , 4.52060120e-01, 4.45643937e-01, 4.42668966e-01, 4.42916724e-01
# , 4.46690480e-01, 4.53550672e-01, 4.62032925e-01, 4.70745699e-01
# , 4.78097658e-01, 4.82706948e-01, 4.84195395e-01, 4.83238620e-01
# , 4.80108435e-01, 4.76494993e-01, 4.73574540e-01, 4.71806846e-01
# , 4.71053995e-01, 4.70949629e-01, 4.71239235e-01, 4.71911104e-01
# , 4.73095276e-01, 4.74827531e-01, 4.76848892e-01, 4.78584837e-01
# , 4.79318195e-01, 4.78447512e-01, 4.75695170e-01, 4.71172961e-01
# , 4.65275467e-01, 4.58445773e-01, 4.50942120e-01, 4.42761543e-01
# , 4.33773135e-01, 4.23957428e-01, 4.13593980e-01, 4.03296558e-01
# , 3.93878581e-01, 3.86100293e-01, 3.80400820e-01, 3.76737004e-01
# , 3.74611163e-01, 3.73270903e-01, 3.71966338e-01, 3.70134214e-01
# , 3.67447860e-01, 3.63763211e-01, 3.59048115e-01, 3.53368656e-01
# , 3.46928909e-01, 3.40092426e-01, 3.33319752e-01, 3.27026803e-01
# , 3.21445563e-01, 3.16582290e-01, 3.12296688e-01, 3.08437213e-01
# , 3.04943813e-01, 3.01868773e-01, 2.99318357e-01, 2.97347200e-01
# , 2.95841129e-01, 2.94422344e-01, 2.92421028e-01, 2.88962808e-01
# , 2.83186647e-01, 2.74538625e-01, 2.63039724e-01, 2.49429576e-01
# , 2.35115148e-01, 2.21882493e-01, 2.11392841e-01, 2.04622266e-01
# , 2.01541302e-01, 2.01251827e-01, 2.02479959e-01, 2.04105230e-01
# , 2.05476648e-01, 2.06443244e-01, 2.07150857e-01, 2.07735313e-01
# , 2.08087677e-01, 2.07823581e-01, 2.06447790e-01, 2.03586614e-01
# , 1.99156536e-01, 1.93405984e-01, 1.86836480e-01, 1.80055040e-01
# , 1.73630150e-01, 1.68012845e-01, 1.63542730e-01, 1.60506587e-01
# , 1.59185464e-01, 1.59832854e-01, 1.62569499e-01, 1.67242543e-01
# , 1.73344629e-01, 1.80075852e-01, 1.86548042e-01, 1.92041655e-01
# , 1.96203317e-01, 1.99111894e-01, 2.01195696e-01, 2.03029476e-01
# , 2.05078529e-01, 2.07469837e-01, 2.09370466e-01, 2.11607363e-01
# , 2.13477486e-01, 2.14602696e-01, 2.15171872e-01, 2.15810181e-01
# , 2.17445676e-01, 2.21064255e-01, 2.27407616e-01, 2.36714953e-01
# , 2.48610883e-01, 2.62183295e-01, 2.76208348e-01, 2.89429641e-01
# , 3.00806256e-01, 3.09681520e-01, 3.15858418e-01, 3.19586820e-01
# , 3.21477020e-01, 3.22362088e-01, 3.23142180e-01, 3.24651463e-01
# , 3.27580227e-01, 3.32456309e-01, 3.39656992e-01, 3.49407057e-01
# , 3.61645875e-01, 3.75908035e-01, 3.92748344e-01, 4.10957592e-01
# , 4.29654002e-01, 4.47920069e-01, 4.64927167e-01, 4.80046992e-01
# , 4.92911345e-01, 5.03416163e-01, 5.11684947e-01, 5.18015502e-01
# , 5.22834500e-01, 5.26675424e-01, 5.30176622e-01, 5.34076022e-01
# , 5.39169101e-01, 5.46202251e-01, 5.55701489e-01, 5.67792559e-01
# , 5.82120989e-01, 5.97952445e-01, 6.14413278e-01, 6.30738225e-01
# , 6.46427699e-01, 6.61299963e-01, 6.75455563e-01, 6.89168619e-01
# , 7.02725250e-01, 7.16263752e-01, 7.29695587e-01, 7.42749078e-01
# , 7.55104032e-01, 7.66550186e-01, 7.77119769e-01, 7.87167860e-01
# , 7.97373098e-01, 8.08625509e-01, 8.21802666e-01, 8.37514652e-01
# , 8.55948218e-01, 8.76887413e-01, 8.99876419e-01, 9.24428724e-01
# , 9.50201504e-01, 9.77091033e-01, 1.00521222e+00, 1.03475639e+00
# , 1.06578938e+00, 1.09808708e+00, 1.13108337e+00, 1.16392288e+00
# , 1.19553415e+00, 1.22506989e+00, 1.24942034e+00, 1.26496506e+00
# , 1.26746879e+00, 1.24810409e+00, 1.19667590e+00, 1.10264807e+00
# , 9.55977957e-01, 7.44833513e-01, 4.40922339e-01,-1.91248981e-06]
#         , name='I', dtype=choose_dtype(I))

#     Q = tf.constant(
#         [-1.69394490e-06,-4.95854195e-01,-8.34305002e-01,-1.04740755e+00
# ,-1.19131753e+00,-1.28490016e+00,-1.33930665e+00,-1.36492117e+00
# ,-1.36825911e+00,-1.35724868e+00,-1.33728383e+00,-1.31141612e+00
# ,-1.28430798e+00,-1.25711791e+00,-1.23092114e+00,-1.20631994e+00
# ,-1.18343362e+00,-1.16198492e+00,-1.14141981e+00,-1.12102881e+00
# ,-1.10005276e+00,-1.07776169e+00,-1.05347408e+00,-1.02636323e+00
# ,-9.97378499e-01,-9.66088591e-01,-9.32604985e-01,-8.97262298e-01
# ,-8.60535584e-01,-8.22984920e-01,-7.85208696e-01,-7.47803931e-01
# ,-7.11334081e-01,-6.76303534e-01,-6.43136404e-01,-6.12155348e-01
# ,-5.83558538e-01,-5.57397576e-01,-5.33561332e-01,-5.11771674e-01
# ,-4.91594512e-01,-4.72441485e-01,-4.53937683e-01,-4.35582330e-01
# ,-4.16857036e-01,-3.97505794e-01,-3.77364094e-01,-3.56391547e-01
# ,-3.34557855e-01,-3.11917025e-01,-2.88564122e-01,-2.64617501e-01
# ,-2.40207987e-01,-2.15471760e-01,-1.90544791e-01,-1.65557697e-01
# ,-1.40630368e-01,-1.15866740e-01,-9.13528709e-02,-6.71608604e-02
# ,-4.33563159e-02,-2.00062244e-02, 2.81382478e-03, 2.50099421e-02
# , 4.65095913e-02, 6.72149000e-02, 8.69810541e-02, 1.05721598e-01
# , 1.23290769e-01, 1.39519409e-01, 1.54250819e-01, 1.67361007e-01
# , 1.78782156e-01, 1.88521902e-01, 1.96670404e-01, 2.03390709e-01
# , 2.08893577e-01, 2.13402575e-01, 2.17117244e-01, 2.20182248e-01
# , 2.22668756e-01, 2.24571388e-01, 2.25821576e-01, 2.26316753e-01
# , 2.25964878e-01, 2.24742408e-01, 2.22708539e-01, 2.19938760e-01
# , 2.16571028e-01, 2.12819006e-01, 2.08921768e-01, 2.05103901e-01
# , 2.01548021e-01, 1.98374660e-01, 1.95627554e-01, 1.93265446e-01
# , 1.91163929e-01, 1.89130426e-01, 1.86932842e-01, 1.84340385e-01
# , 1.81161273e-01, 1.77253063e-01, 1.72556077e-01, 1.67072125e-01
# , 1.60969939e-01, 1.54500995e-01, 1.47950224e-01, 1.41598720e-01
# , 1.35694782e-01, 1.30432860e-01, 1.25939570e-01, 1.22266316e-01
# , 1.19388614e-01, 1.17211954e-01, 1.15583191e-01, 1.14305996e-01
# , 1.13159069e-01, 1.11916232e-01, 1.10367724e-01, 1.08341579e-01
# , 1.05722910e-01, 1.02467904e-01, 9.86092441e-02, 9.42510019e-02
# , 8.95534093e-02, 8.47103740e-02, 7.99241677e-02, 7.53820538e-02
# , 7.12385743e-02, 6.76051262e-02, 6.45442840e-02, 6.20232434e-02
# , 6.00052452e-02, 5.84309794e-02, 5.72128313e-02, 5.62486525e-02
# , 5.54348684e-02, 5.46784428e-02, 5.39063286e-02, 5.30706471e-02
# , 5.21483528e-02, 5.11360543e-02, 5.00433139e-02, 4.88895227e-02
# , 4.77115374e-02, 4.65260123e-02, 4.53787272e-02, 4.43797011e-02
# , 4.36572788e-02, 4.33479602e-02, 4.35781281e-02, 4.44505756e-02
# , 4.60387326e-02, 4.83861282e-02, 5.15080401e-02, 5.53927143e-02
# , 6.00008876e-02, 6.52642525e-02, 7.10841136e-02, 7.73291448e-02
# , 8.38315308e-02, 9.03889134e-02, 9.67748409e-02, 1.02751865e-01
# , 1.08083520e-01, 1.12544449e-01, 1.15928401e-01, 1.18053989e-01
# , 1.18769415e-01, 1.17961149e-01, 1.15580111e-01, 1.11699062e-01
# , 1.06558964e-01, 1.00510973e-01, 9.38804485e-02, 8.68837826e-02
# , 7.96331958e-02, 7.21829219e-02, 6.45795896e-02, 5.69035946e-02
# , 4.92986121e-02, 4.19879572e-02, 3.52743654e-02, 2.95167994e-02
# , 2.50777775e-02, 2.22420579e-02, 2.11237589e-02, 2.15966695e-02
# , 2.32852210e-02, 2.56300907e-02, 2.80049241e-02, 2.98383163e-02
# , 3.07022148e-02, 3.03499414e-02, 2.87041933e-02, 2.58055870e-02
# , 2.17445689e-02, 1.66115167e-02, 1.04951587e-02, 3.53252873e-03
# ,-4.01657167e-03,-1.17114503e-02,-1.89266763e-02,-2.49269729e-02
# ,-2.90186786e-02,-3.07357155e-02,-2.99861779e-02,-2.70934813e-02
# ,-2.27164000e-02,-1.76879692e-02,-1.28365457e-02,-8.84006219e-03
# ,-6.13773605e-03,-4.90145535e-03,-5.05762540e-03,-6.34520165e-03
# ,-8.39225488e-03,-1.07929876e-02,-1.31735143e-02,-1.52457191e-02
# ,-1.68547632e-02,-1.80199223e-02,-1.89534520e-02,-2.00317410e-02
# ,-2.17018592e-02,-2.43349480e-02,-2.80662760e-02,-3.26712076e-02
# ,-3.75201563e-02,-4.16475643e-02,-4.39545135e-02,-4.35173381e-02
# ,-3.98960659e-02,-3.32936935e-02,-2.44875194e-02,-1.45919951e-02
# ,-4.78292099e-03, 3.92406425e-03, 1.08166547e-02, 1.55643229e-02
# , 1.82454287e-02, 1.93164315e-02, 1.94963748e-02, 1.95408385e-02
# , 1.99296216e-02, 2.05939088e-02, 2.08706351e-02, 1.97606430e-02
# , 1.63515739e-02, 1.01909998e-02, 1.49403340e-03,-8.83561841e-03
# ,-1.93789925e-02,-2.85657444e-02,-3.51342291e-02,-3.84835845e-02
# ,-3.87406646e-02,-3.65473999e-02,-3.27320474e-02,-2.80352085e-02
# ,-2.29743273e-02,-1.78455581e-02,-1.28102883e-02,-7.98949568e-03
# ,-3.50178153e-03, 5.61788386e-04, 4.17220588e-03, 7.34328224e-03
# , 1.00421591e-02, 1.20828515e-02, 1.30976038e-02, 1.26411893e-02
# , 1.03863501e-02, 6.31274893e-03, 8.11012500e-04,-5.33857356e-03
# ,-1.11321222e-02,-1.56370908e-02,-1.83144020e-02,-1.91860902e-02
# ,-1.87317468e-02,-1.75967405e-02,-1.63045927e-02,-1.50983146e-02
# ,-1.39269326e-02,-1.25377736e-02,-1.06201034e-02,-7.94839643e-03
# ,-4.48892746e-03,-4.48506020e-04, 3.75115236e-03, 7.57917719e-03
# , 1.05553061e-02, 1.24081062e-02, 1.31812737e-02, 1.32664633e-02
# , 1.33563866e-02, 1.43078143e-02, 1.69155084e-02, 2.16520844e-02
# , 2.84939312e-02, 3.69426938e-02, 4.62249672e-02, 5.55262290e-02
# , 6.41278713e-02, 7.14254336e-02, 7.68914642e-02, 8.00638819e-02
# , 8.05916017e-02, 7.82982272e-02, 7.32217590e-02, 6.56443502e-02
# , 5.61499364e-02, 4.57117139e-02, 3.57338627e-02, 2.79251813e-02
# , 2.39471091e-02, 2.48990930e-02, 3.08592430e-02, 4.07761610e-02
# , 5.27345290e-02, 6.44957033e-02, 7.44171168e-02, 8.18188540e-02
# , 8.64444415e-02, 8.81679453e-02, 8.69646539e-02, 8.29305462e-02
# , 7.62843752e-02, 6.73446793e-02, 5.65272405e-02, 4.44400761e-02
# , 3.21632658e-02, 2.11346065e-02, 1.26428651e-02, 7.55588164e-03
# , 6.18435576e-03, 8.13354472e-03, 1.24069947e-02, 1.78917408e-02
# , 2.37858507e-02, 2.96358515e-02, 3.52084385e-02, 4.03510767e-02
# , 4.49717173e-02, 4.90103358e-02, 5.24287201e-02, 5.52163517e-02
# , 5.74039277e-02, 5.90761993e-02, 6.03883206e-02, 6.15998553e-02
# , 6.30600534e-02, 6.52229181e-02, 6.82058077e-02, 7.15609708e-02
# , 7.47604298e-02, 7.73875145e-02, 7.91620058e-02, 7.98943711e-02
# , 7.94651676e-02, 7.78204778e-02, 7.49742195e-02, 7.10192990e-02
# , 6.61593186e-02, 6.07712330e-02, 5.54292438e-02, 5.10539932e-02
# , 4.84813132e-02, 4.79957205e-02, 4.92442139e-02, 5.15196225e-02
# , 5.40648981e-02, 5.62400632e-02, 5.75768432e-02, 5.77861232e-02
# , 5.66715805e-02, 5.41552910e-02, 5.02700344e-02, 4.52246615e-02
# , 3.93224204e-02, 3.28318968e-02, 2.59023241e-02, 1.86261319e-02
# , 1.11663127e-02, 3.86188837e-03,-2.71916440e-03,-7.85772417e-03
# ,-1.10198070e-02,-1.23322907e-02,-1.26236879e-02,-1.28534567e-02
# ,-1.36721099e-02,-1.53927493e-02,-1.80724136e-02,-2.15702845e-02
# ,-2.56128271e-02,-2.99605382e-02,-3.46868119e-02,-4.02581665e-02
# ,-4.71919935e-02,-5.58576100e-02,-6.64965764e-02,-7.92146848e-02
# ,-9.40136627e-02,-1.10795416e-01,-1.29407114e-01,-1.49650843e-01
# ,-1.71283014e-01,-1.94027330e-01,-2.17586977e-01,-2.41657775e-01
# ,-2.65942351e-01,-2.90165337e-01,-3.14088171e-01,-3.37513206e-01
# ,-3.60258760e-01,-3.82110690e-01,-4.02800817e-01,-4.22011616e-01
# ,-4.39369725e-01,-4.54523567e-01,-4.67550510e-01,-4.79298190e-01
# ,-4.90933144e-01,-5.03462796e-01,-5.17586337e-01,-5.33651412e-01
# ,-5.51651323e-01,-5.71278383e-01,-5.92047597e-01,-6.13470636e-01
# ,-6.35229453e-01,-6.57289939e-01,-6.79896175e-01,-7.03397542e-01
# ,-7.27931950e-01,-7.53146502e-01,-7.78246910e-01,-8.02737483e-01
# ,-8.25382295e-01,-8.44872339e-01,-8.62231974e-01,-8.77818234e-01
# ,-8.91567510e-01,-9.03059375e-01,-9.11966582e-01,-9.18543656e-01
# ,-9.23822939e-01,-9.29417228e-01,-9.37029670e-01,-9.47842899e-01
# ,-9.61992005e-01,-9.78463312e-01,-9.95637573e-01,-1.01218660e+00
# ,-1.02785436e+00,-1.04480197e+00,-1.06176856e+00,-1.08049678e+00
# ,-1.10018299e+00,-1.11883652e+00,-1.13506517e+00,-1.14866730e+00
# ,-1.15969674e+00,-1.16742132e+00,-1.17101045e+00,-1.17150131e+00
# ,-1.17312327e+00,-1.18268645e+00,-1.20638832e+00,-1.24398707e+00
# ,-1.28691302e+00,-1.32469092e+00,-1.35026071e+00,-1.36102946e+00
# ,-1.35719256e+00,-1.34043627e+00,-1.31713783e+00,-1.30273377e+00
# ,-1.31894767e+00,-1.35501427e+00,-1.39551369e+00,-1.43052065e+00
# ,-1.46041815e+00,-1.49149846e+00,-1.51739473e+00,-1.52974879e+00
# ,-1.52797497e+00,-1.52166290e+00,-1.52413008e+00,-1.53371494e+00
# ,-1.54457987e+00,-1.55654581e+00,-1.56173755e+00,-1.53650131e+00
# ,-1.45442243e+00,-1.26511854e+00,-8.35187352e-01,-5.31314906e-06]
#         , name='Q', dtype=choose_dtype(I))

    # I = tf.constant([np.pi*2/T*CUSTOM_n], name='I', dtype=choose_dtype(I))
    # if tf.size(I) == 1: 
    #     I = tf.constant([np.pi*2/T*CUSTOM_n], name='I', dtype=choose_dtype(I))
    #     Q = tf.constant([0.], name='Q', dtype=choose_dtype(I))

    # I = tf.constant([np.pi*2/T*CUSTOM_n/np.sqrt(2)]*STEPS, name='I', dtype=choose_dtype(I))
    # Q = tf.constant([np.pi*2/T*CUSTOM_n/np.sqrt(2)]*STEPS, name='Q', dtype=choose_dtype(I))

    # I, Q = tf.Variable(I), tf.Variable(Q)

    ## OPTIMIZATION ###

    loss_hist = []
    if REG_PULSE:
        pulse_loss_hist = []
        target_loss_hist = []
    else:
        pulse_loss_hist = None
        target_loss_hist = None

    variables = [I, Q]

    OPT_STEPS_USED = OPT_STEPS

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #   LR,
    #   decay_steps=int(OPT_STEPS/100),
    #   decay_rate=0.81,
    #   staircase=True)

    # opt = tf.keras.optimizers.Nadam(learning_rate=LR) #(learning_rate=lr_schedule)
    # for i in range(int(OPT_STEPS)): #/10)):

    loss = total_loss(I, Q)

    for o in range(0,11):
        opt = tf.keras.optimizers.Nadam(learning_rate=LR/((100**0.1)**o))

        for i in range(int(OPT_STEPS / 10)):

            with tf.GradientTape() as tape:
                tape.watch(variables)

                loss = total_loss(I, Q)
                pulse_loss = reg_loss(I, Q)*CUSTOM_pulse_reg # not larger than 0.1

                # print(loss, '+', pulse_loss, end=' | ')

                loss_grads = loss + pulse_loss

            grads = tape.gradient(loss_grads, variables)
            opt.apply_gradients(zip(grads, variables))
            loss_hist.append(loss_grads.numpy())
            if REG_PULSE:
                pulse_loss_hist.append(pulse_loss.numpy())
                target_loss_hist.append(loss.numpy())

            try:
                if loss_hist[-1] < OPT_LIM:
                    print("\nReached loss of OPT_LIM =", OPT_LIM)
                    OPT_STEPS_USED = i
                    break
            except: print("Did not optimize")
         
        try:       
            if loss_hist[-1] < OPT_LIM:
                        print("\nReached loss of OPT_LIM =", OPT_LIM)
                        OPT_STEPS_USED = i
                        break
        except: print("Did not optimize")

    if plot_res!=None: higherRes = int(np.ceil(plot_res/STEPS/diff_factor))
    else: higherRes = 1

    if FFT:
        # whenever in FFT mode, hardwire a fft=False and apply transforms outside of upsampled
        # sim, should be fine without tf sugar since its called only once...
        res_pad = (higherRes*diff_factor-1)*tf.size(I)/2 # copied from inside loss_func
        freqs = tf.signal.ifftshift( tf.pad( tf.signal.fftshift(tf.dtypes.complex(1/tf.size(I)*I, 1/tf.size(Q)*Q)) , [[res_pad, res_pad]]) )
        encoded = tf.signal.ifft( freqs )

        fft_window_cnst = get_window_function(tf.size(encoded))

        highResI = tf.math.real(encoded) * fft_window_cnst
        highResQ = tf.math.imag(encoded) * fft_window_cnst

        HR_loss_func_params = {"DT": DT/higherRes, "resolution": 1, "METHOD": METHOD, "fft": False, "data": True}

    else:

        highResI = tf.repeat(I, higherRes)
        highResQ = tf.repeat(Q, higherRes)

        HR_loss_func_params = {"DT": DT/higherRes, "resolution": diff_factor, "METHOD": METHOD, "fft": False, "data": True}

    pops, purity, outs  = loss_func(*sim_params, highResI, highResQ, **HR_loss_func_params)[1:4]
    pops2, purity2, outs2 = loss_func(*sim_params_2, highResI, highResQ, **HR_loss_func_params)[1:4]
    if qubits>2: pops3, purity3, outs3 = loss_func(*sim_params_3, highResI, highResQ, **HR_loss_func_params)[1:4]
    if qubits>3: pops4, purity4, outs4 = loss_func(*sim_params_4, highResI, highResQ, **HR_loss_func_params)[1:4]
    if qubits>4: pops5, purity5, outs5 = loss_func(*sim_params_5, highResI, highResQ, **HR_loss_func_params)[1:4]
    if qubits>5: pops6, purity6, outs6 = loss_func(*sim_params_6, highResI, highResQ, **HR_loss_func_params)[1:4]

    if diff_factor != 1 and plot_res!=None: resamp = lambda x: x[::int( np.ceil(tf.size(x) / plot_res) )]
    else: resamp = lambda x: x

    np.set_printoptions(threshold=sys.maxsize)

    calcT = HR_loss_func_params["DT"] * tf.size(highResI).numpy()

    print("sanity check, T=DT*steps:", True if calcT==T else str(T) + " != " + str(calcT) + "="+str(HR_loss_func_params["DT"])+"*"+str(tf.size(highResI).numpy()))

    # # testing purposes
    # for k in np.linspace(0, 6.28, 50):

    #     mag, phase = tf.sqrt(highResI**2 + highResQ**2), np.arctan( highResQ / highResI )

    #     phase += k

    #     newI, newQ = mag * (tf.cos(phase), tf.sin(phase))

    #     pops, purity, outs  = loss_func(*sim_params, highResI, highResQ, **HR_loss_func_params)[1:4]
    #     pops2, purity2, outs2 = loss_func(*sim_params_2, highResI, highResQ, **HR_loss_func_params)[1:4]

    #     print(k, pops.numpy().flatten()[-1], pops2.numpy().flatten()[-1])
    # # testing done

    # odmr

    print("ODMR INIT")

    try:

        odmr_domain = np.linspace(-25*2*np.pi, 25*2*np.pi, 500)
        odmra = []
        odmrb = []

        odmra1, odmra2, odmra3 = [], [], []

        for i in odmr_domain:

            sim_params                 = generate_constants(CUSTOM_ground_splitting+0 , omega_drive+i, B, 0, METHOD)
            sim_params_2               = generate_constants(CUSTOM_ground_splitting+CUSTOM_fine_splitting, omega_drive+i, B, 0, METHOD)
            sim_params_3               = generate_constants(CUSTOM_ground_splitting+2*CUSTOM_fine_splitting, omega_drive+i, B, 0, METHOD)

            sim_params_4               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+0. , omega_drive+i, B, 0, METHOD)
            sim_params_5               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+CUSTOM_fine_splitting, omega_drive+i, B, 0, METHOD)
            sim_params_6               = generate_constants(CUSTOM_ground_splitting+CUSTOM_splitting+2*CUSTOM_fine_splitting, omega_drive+i, B, 0, METHOD)


            a1, a2, a3 = loss_func(*sim_params, I=I, Q=Q, **loss_func_params), loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), loss_func(*sim_params_3, I=I, Q=Q, **loss_func_params)

            a = a1+a2+a3
            b = loss_func(*sim_params_4, I=I, Q=Q, **loss_func_params) + loss_func(*sim_params_5, I=I, Q=Q, **loss_func_params) + loss_func(*sim_params_6, I=I, Q=Q, **loss_func_params)
            
            odmra.append(a.numpy().flatten()[-1])
            odmrb.append(b.numpy().flatten()[-1])

            odmra1.append(a1.numpy().flatten()[-1])
            odmra2.append(a2.numpy().flatten()[-1])
            odmra3.append(a3.numpy().flatten()[-1])

        df_data["odmra"].append(np.array(odmra))
        df_data["odmrb"].append(np.array(odmrb))
        df_data["odmr_domain"].append(odmr_domain/2/np.pi)

        df_data["odmra1"].append(np.array(odmra1))
        df_data["odmra2"].append(np.array(odmra2))
        df_data["odmra3"].append(np.array(odmra3))

    except Exception as e:
        print('odmr err', e)

    # odmr done

    # units

    print("UNIT INIT")

    try:

        sim_params = generate_constants(1., 1., B, 0, METHOD)

        temp_loss = {"DT": T/diff_factor/200, "resolution": 1, "METHOD": METHOD, "fft": FFT}

        a = loss_func(*sim_params, I=tf.constant(np.array([np.pi*2/T]*200), dtype=choose_dtype(I)), Q=tf.constant(np.array([0]*200), dtype=choose_dtype(I)), **temp_loss, data=True)[1]

        df_data["units"].append(a.numpy().flatten())

    except Exception as e:
        print(e)

    # units done

    df_data["loss"].append(loss.numpy()[0])
    df_data["pops"].append(resamp(pops.numpy()[0].flatten()))
    if qubits>1: df_data["pops2"].append(resamp(pops2.numpy()[0].flatten()))
    if qubits>2: df_data["pops3"].append(resamp(pops3.numpy()[0].flatten()))
    if qubits>3: df_data["pops4"].append(resamp(pops4.numpy()[0].flatten()))
    if qubits>4: df_data["pops5"].append(resamp(pops5.numpy()[0].flatten()))
    if qubits>5: df_data["pops6"].append(resamp(pops6.numpy()[0].flatten()))
    df_data["loss_hist"].append(loss_hist)
    df_data["pulse_loss_hist"].append(pulse_loss_hist)
    df_data["target_loss_hist"].append(target_loss_hist)
    df_data["iter_step"].append(range(len(loss_hist)))

    if not FFT:

        df_data["I"].append(I.numpy())
        df_data["Q"].append(Q.numpy())

        df_data["fft:I"].append(None)
        df_data["fft:Q"].append(None)

    else:

        df_data["I"].append(highResI.numpy())
        df_data["Q"].append(highResQ.numpy())

        df_data["fft:I"].append(I.numpy())
        df_data["fft:Q"].append(Q.numpy())

    df_data["LR"].append(LR)
    df_data["OPT_STEPS"].append(OPT_STEPS_USED)
    df_data["distance"].append(d)
    df_data["STEPS"].append(STEPS)
    df_data["T"].append(T)
    df_data["purity"].append(resamp(purity.numpy().flatten()))
    if qubits>1: df_data["purity2"].append(resamp(purity2.numpy().flatten()))
    if qubits>2: df_data["purity3"].append(resamp(purity3.numpy().flatten()))
    if qubits>3: df_data["purity4"].append(resamp(purity4.numpy().flatten()))
    if qubits>4: df_data["purity5"].append(resamp(purity5.numpy().flatten()))
    if qubits>5: df_data["purity6"].append(resamp(purity6.numpy().flatten()))

    df_data["final_pops"].append(pops.numpy().flatten()[-1])
    if qubits>1: df_data["final_pops2"].append(pops2.numpy().flatten()[-1])
    if qubits>2: df_data["final_pops3"].append(pops3.numpy().flatten()[-1])
    if qubits>3: df_data["final_pops4"].append(pops4.numpy().flatten()[-1])
    if qubits>4: df_data["final_pops5"].append(pops5.numpy().flatten()[-1])
    if qubits>5: df_data["final_pops6"].append(pops6.numpy().flatten()[-1])

    current_time = datetime.now().strftime("%H:%M:%S")
    print("Done w/", sim_id, "@", current_time)

    print('LOSS:', loss.numpy())
    sys.stdout.flush()

    try:
        df = pd.DataFrame(df_data)
        df.to_csv(folder+'/'+sim_id+'.csv')
    except:
        print('backup df_data:', df_data)
        print("Saving DF failed")
        print([(i, len(df_data[i])) for i in df_data.keys()])
        raise Saving_DF_Failed

    print("Saved", sim_id, "in", folder)
    sys.stdout.flush()
