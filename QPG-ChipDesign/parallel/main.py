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
    gamma = 2*np.pi*28 # unit GHz/T

    qubits = 3
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

    data = json.loads(sys.argv[1])

    for key, value in data.items():
      vars()[key] = value

    folder = sys.argv[-2]

    sim_id = "sim_" + sys.argv[-1]

    step_space = [steps]
    d_space = [d]

    t_space = [T]
    lr_array = [LR]

    df_data = {"pops": [], "T": [], "LR": [], "loss": [], "I": [], "Q": [], "OPT_STEPS": [], "distance": [], "loss_hist": [], "iter_step": [], "STEPS": [], "purity": [], "fft:I": [], "fft:Q": [], "guess:I": [], "guess:Q": [], "guess:loss": [], "pulse_loss_hist": [], "target_loss_hist": []}

    df_data['data'] = sys.argv[1]

    if qubits > 1: df_data["pops2"] = []; df_data["purity2"] = [];
    if qubits > 2: df_data["pops3"] = []; df_data["purity3"] = [];
    if qubits > 3: df_data["pops4"] = []; df_data["purity4"] = [];

    B = 1.055*2/gamma*10

    if FFT: fft_window_cnst = get_window_function(steps*diff_factor)

    print(f'began running {sim_id}')

    sys.stdout.flush()

    STEPS = steps

    if diff_target != None:
        # diff_factor = round(T /STEPS /diff_target)
        diff_factor = round(1 /STEPS /diff_target)
        if diff_factor < 1: diff_factor = 1

    DT = T /STEPS /diff_factor
    omega0p, omega_drive = 1, 1

    omega_drive = omega0p/((target_qubit-1)*d+1)

    sim_params                 = generate_constants(omega0p, omega_drive, B, B, METHOD)
    sim_params_2               = generate_constants(omega0p/(1*d+1), omega_drive, B/(1*d+1), B/(1*d+1), METHOD)
    sim_params_3, sim_params_4 = (), ()
    if qubits>2: sim_params_3  = generate_constants(omega0p/(2*d+1), omega_drive, B/(2*d+1), B/(2*d+1), METHOD)
    if qubits>3: sim_params_4  = generate_constants(omega0p/(3*d+1), omega_drive, B/(3*d+1), B/(3*d+1), METHOD)

    loss_func_params = {"DT": DT, "resolution": diff_factor, "METHOD": METHOD, "fft": FFT}

    @tf.function
    def total_loss(I, Q):
        flf = lambda x, y: (1 - x) if y!=target_qubit else x

        if qubits==4: loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2) + flf(loss_func(*sim_params_3, I=I, Q=Q, **loss_func_params), 3) + flf(loss_func(*sim_params_4, I=I, Q=Q, **loss_func_params), 4)
        if qubits==3: loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2) + flf(loss_func(*sim_params_3, I=I, Q=Q, **loss_func_params), 3)
        if qubits==2: loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + flf(loss_func(*sim_params_2, I=I, Q=Q, **loss_func_params), 2)
        if qubits==1: loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1)

        # ENABLE THIS WHEN TRYING TO GET UNITS
        loss = flf(loss_func(*sim_params, I, Q, **loss_func_params), 1) + tf.reduce_mean( abs(Q) )*50

        return loss

    @tf.function
    def reg_loss(I, Q):

        loss = tf.constant(0., dtype=tf.double)

        if REG_PULSE == False: return tf.constant(0., dtype=tf.double)

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

    opt = tf.keras.optimizers.Nadam(learning_rate=LR) #(learning_rate=lr_schedule)

    # for o in range(0,11):
      # opt = tf.keras.optimizers.Nadam(learning_rate=LR/((100**0.1)**o))

    for i in range(int(OPT_STEPS)): #/10)):

        with tf.GradientTape() as tape:
            tape.watch(variables)

            loss = total_loss(I, Q)
            pulse_loss = reg_loss(I, Q)*0.1 # not larger than 0.1

            # print(loss, '+', pulse_loss, end=' | ')

            loss_grads = loss + pulse_loss

        grads = tape.gradient(loss_grads, variables)
        opt.apply_gradients(zip(grads, variables))
        loss_hist.append(loss_grads.numpy())
        if REG_PULSE:
            pulse_loss_hist.append(pulse_loss.numpy())
            target_loss_hist.append(loss.numpy())

        if loss_hist[-1] < OPT_LIM:
            print("\nReached loss of OPT_LIM =", OPT_LIM)
            OPT_STEPS_USED = i
            break

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

    if diff_factor != 1 and plot_res!=None: resamp = lambda x: x[::int( np.ceil(tf.size(x) / plot_res) )]
    else: resamp = lambda x: x

    np.set_printoptions(threshold=sys.maxsize)

    calcT = HR_loss_func_params["DT"] * tf.size(highResI).numpy()

    print("sanity check, T=DT*steps:", True if calcT==T else str(T) + " != " + str(calcT) + "="+str(HR_loss_func_params["DT"])+"*"+str(tf.size(highResI).numpy()))

    df_data["loss"].append(loss.numpy()[0])
    df_data["pops"].append(resamp(pops.numpy()[0].flatten()))
    if qubits>1: df_data["pops2"].append(resamp(pops2.numpy()[0].flatten()))
    if qubits>2: df_data["pops3"].append(resamp(pops3.numpy()[0].flatten()))
    if qubits>3: df_data["pops4"].append(resamp(pops4.numpy()[0].flatten()))
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

    print(df_data)

    sys.stdout.flush()

    current_time = datetime.now().strftime("%H:%M:%S")
    print("Done w/", sim_id, "@", current_time)

    df = pd.DataFrame(df_data)
    df.to_csv(folder+'/'+sim_id+'.csv')

    print("Saved", sim_id, "in", folder)
    sys.stdout.flush()
