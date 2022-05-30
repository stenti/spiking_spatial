import matplotlib.pyplot as plt
import nest as sim
import numpy as np
import pandas
from collections import Counter
import time as tm
import scipy.stats
import os

mins = 3.
sim_len = int(mins * 60000)

# params describing number of cells in each population
N_ex = 180
N_in = N_ex
N_cj = N_ex

# params for describing connection weights between populations
sigma = 0.12 
mu = 0.5
delay = 0.1
base_ex = 4000
base_in = 450
base_cj = 169
w_ex_cj = 660

# params describing size , duration and excitatory cell number to initialize the bump
I_init = 300.0 #pA
I_init_dur = 100.0 #ms
I_init_pos = N_ex//2

# connections between excitatory and inhibitory rings described in weight matrices
w_ex = np.empty((N_in,N_ex))
w_in = np.empty((N_ex,N_in))

#loop through each excitatory and inhibitory cell
for e in range(N_ex):
    for i in range(N_in):
        # find the smallest distance between the excitatory and inhibitory cell, looking both ways around the ring
        d1 = abs(e/N_ex - i/N_in)
        d2 = abs(e/N_ex - i/N_in -1)
        d3 = abs(e/N_ex - i/N_in +1)
        d = min(abs(d1),abs(d2),abs(d3))
        
        #gaussian function finds the distance dependent connection strength
        w_gauss = np.exp(-(d)**2/2/sigma**2)
        w_ring = np.exp(-(d - mu)**2/2/sigma**2) #inhibitory connections are ofset by parameter mu
        # scale by base weight and add to matrix
        w_ex[i,e] = base_ex * w_gauss
        w_in[e,i] = base_in * w_ring 

# set all very small weights to zero to reduce total number of connections
w_ex[w_ex<10]=0
w_in[w_in<10]=0

# connections between conjuntive layers and excitatory ring
w_l = np.empty((N_ex,N_cj))
w_r = np.empty((N_ex,N_cj))
for c in range(N_cj):  
    for e in range(N_ex):
        d1 = abs((e-1)/N_cj - c/N_ex)
        d2 = abs((e-1)/N_cj - c/N_ex -1)
        d3 = abs((e-1)/N_cj - c/N_ex +1)
        d = min(abs(d1),abs(d2),abs(d3))
        w_l[e,c] = base_cj * (np.exp(-(d)**2/2/sigma**2))
        
        d1 = abs((e+1)/N_cj - c/N_ex)
        d2 = abs((e+1)/N_cj - c/N_ex -1)
        d3 = abs((e+1)/N_cj - c/N_ex +1)
        d = min(abs(d1),abs(d2),abs(d3))
        w_r[e,c] = base_cj * (np.exp(-(d)**2/2/sigma**2))
        
m = np.amax(w_l)
w_l[w_l<m] = 0
m = np.amax(w_r)
w_r[w_r<m] = 0


directory = os.fsencode('data')
    
for f in os.listdir(directory):
    folder = os.fsdecode(f)
    if (folder.endswith("_PCN") or folder.endswith("_VAE") or folder.endswith("_convnet_refined")) and os.path.isdir(f'data/{folder}'):

          sim.ResetKernel()

          #create and connect populations
          exc = sim.Create("iaf_psc_alpha",N_ex, params={"I_e": 450.})
          inh = sim.Create("iaf_psc_alpha",N_in)

          l = sim.Create("iaf_psc_alpha",N_cj)
          r = sim.Create("iaf_psc_alpha",N_cj)

          sim.Connect(exc,inh,'all_to_all',syn_spec={'weight': w_ex, 'delay': delay})
          sim.Connect(inh,exc,'all_to_all',syn_spec={'weight': -w_in, 'delay': delay})

          sim.Connect(exc,l,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})
          sim.Connect(exc,r,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})

          sim.Connect(l,exc,'all_to_all',syn_spec={'weight': w_l, 'delay': delay})
          sim.Connect(r,exc,'all_to_all',syn_spec={'weight': w_r, 'delay': delay})

          exc_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable
          sim.Connect(exc,exc_spikes)


          posedata = pandas.read_csv(f'data/{folder}.csv')

          f = 50 #Hz
          dt = int(1000/f)
          theta = posedata['field.theta']
          theta = np.array(theta)

          angle_per_cell = (2*np.pi)/N_ex
          I_init_pos = np.around((theta[0]//angle_per_cell)+(N_ex//2)).astype(int)

          t = np.arange(0,(len(theta)*dt),dt*1.) #assume 20ms timestep
          time = [i for i in t if i < sim_len]

          vel = np.diff(np.unwrap(theta))


          #arbitary scaling
          Ivel = vel * 0.35 * 10000

          sh = 150 # add on this threshold as the lowest posible current value
          go_l,go_r = Ivel,-Ivel #separate into clockwise and anticlockwise movements
          go_l = go_l+sh
          go_r = go_r+sh
          go_l[go_l<=sh] = 0 # everything below the threshold set to 0pA
          go_r[go_r<=sh] = 0

          # Connect AV input to conjunctive layers
          l_input = sim.Create('step_current_generator', 1)
          sim.SetStatus(l_input,{'amplitude_times': t[1:],'amplitude_values': go_l})
          r_input = sim.Create('step_current_generator', 1)
          sim.SetStatus(r_input,{'amplitude_times': t[1:],'amplitude_values': go_r})

          sim.Connect(r_input,r,'all_to_all')
          sim.Connect(l_input,l,'all_to_all')

          predtm = pandas.read_csv(f'data/{experiment}_frames.csv')
          predNet = np.load(f'data/{folder}/{folder}_reconstructions.npy')

          predtm = np.array(predtm['%time'])
          predtm = (predtm-predtm[0])/1000000

          # #to use random current inputs:
          # for p in range(len(predNet)):
          #     predNet[p] = np.random.rand(180)

          predNet[predNet<0] = 0
          predNet = np.vstack([predNet,np.zeros([1,180])])

          plt.plot(predNet[0])
          plt.plot(predNet[50])
          plt.plot(predNet[100])

          predicitons = predNet[5:,:]
          prediciton_times = predtm[5:len(predicitons)+5]

          view_input = sim.Create('step_current_generator', N_ex)
          for n in range(N_ex):
              sim.SetStatus([view_input[n]],{'amplitude_times': prediciton_times,'amplitude_values': predicitons[:,n]})
              
          sim.Connect(view_input,exc,'one_to_one')

          sim_len = prediciton_times[-1]
          time = [i*1. for i in t if i < sim_len]


          bump_init = sim.Create('step_current_generator', 1, params = {'amplitude_times':[0.1,0.1+I_init_dur],'amplitude_values':[I_init,0.0]})
          sim.Connect(bump_init,[exc[I_init_pos]])

          print(f'Running ismulation for {folder}...')
          tic = tm.time()
          sim.Simulate(sim_len)
          print(f'Simulation run time: {np.around(tm.time()-tic,2)} s  Simulated time: {np.around(sim_len/1000,2)} s')

          if not os.path.exists(f'data/{folder}'):
               os.makedirs(f'data/{folder}')

          # get the spike times of the cells from the spike recorder
          ev = sim.GetStatus(exc_spikes)[0]['events']
          t = ev['times'] # time of each spike
          sp = ev['senders'] # sender of each spike (cell ID)

          dt = 20
          T = np.arange(0,(len(theta)*dt),dt*2)

          #find the most active cell in each 40ms bin
          modes = np.zeros(len(T))
          modes[:] = np.nan
          rates = np.zeros((N_ex,len(time)))
          for i in range(len(T)-1):
              idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
              lst = sp[np.where(idx)] # get the senders of those spikes
              occurence_count = Counter(lst) 
              mode = occurence_count.most_common(1) # find most common sender
              if len(mode):
                  modes[i] = mode[0][0]
                  
          step = (2*np.pi)/N_ex
          modes = (modes*step) - np.pi

          # fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')
          fig, ax = plt.subplots(1, 1,figsize=(8, 3),facecolor='w')
          ax.set_xlabel('Time (s)')
          ax.set_ylabel('Head angle (deg)')

          ax.plot(T/1000,modes*(180/np.pi),'.',label='corrected',color='hotpink')
          ax.plot(np.array(time)/1000,(theta[:len(time)])*(180/np.pi),'.',markersize=.5,label='ground truth',color='black')
          plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
          plt.xlim([0,sim_len/1000])
          plt.yticks([-180,0,180])
          plt.tight_layout()
          plt.savefig(f'data/{folder}/{folder}_drift_corrected.png')

          nanidx = np.where(~np.isnan(modes[:-1]))
          modes = modes[nanidx]
          T=T[nanidx]

          hd_estimate = np.vstack([T,modes])
          np.save(f'data/{folder}/{folder}_estimate_corrected.npy',hd_estimate)

          est = np.unwrap(modes)
          groundTruth = np.unwrap(theta[:len(time)])
          estimate = np.interp(time, T, est)

          fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')

          d = (groundTruth-estimate)
          plt.plot(np.array(time)/1000,abs(d)*(180/np.pi),color='hotpink')
          plt.xlabel('Time (s)')
          plt.ylabel('Error (deg)')
          plt.xlim([0,sim_len/1000])

          RMSE = np.sqrt(np.sum(d**2)/len(d))
          print(RMSE*(180/np.pi))

          plt.tight_layout()
          plt.savefig(f'data/{folder}/{folder}_error_corrected.png')


          if os.path.exists(f'data/{experiment}/{experiment}_estimate.npy'):
              ideo = np.load(f'data/{experiment}/{experiment}_estimate.npy')
              
              fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')
              ax.set_xlabel('Time (s)')
              ax.set_ylabel('Head angle (deg)')

              ax.plot(ideo[0]/1000,ideo[1]*(180/np.pi),'.',label='ideo only',color='steelblue')
              ax.plot(T/1000,modes*(180/np.pi),'.',label='corrected',color='hotpink')
              ax.plot(np.array(time)/1000,(theta[:len(time)])*(180/np.pi),'.',markersize=.5,label='ground truth',color='black')
              plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
              plt.xlim([0,sim_len/1000])
              plt.yticks([-180,0,180])
              plt.tight_layout()
              plt.savefig(f'data/{folder}/{folder}_drift_comparison.png')

              T = np.array(ideo[0])
              est = np.unwrap(np.array(ideo[1]))
              ideo_estimate = np.interp(time, T, est)

              fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')

              d = (groundTruth-estimate)
              plt.plot(np.array(time)/1000,abs(d)*(180/np.pi),label='corrected',color='hotpink')

              d = (groundTruth-ideo_estimate)
              plt.plot(np.array(time)/1000,abs(d)*(180/np.pi),label='ideo only',color='steelblue')

              plt.xlabel('Time (s)')
              plt.ylabel('Error (deg)')
              plt.xlim([0,sim_len/1000])
              plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
              plt.tight_layout()
              plt.savefig(f'data/{folder}/{folder}_error_comparison.png')