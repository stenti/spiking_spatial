# @author R Stentiford
import matplotlib.pyplot as plt
import nest as sim
import numpy as np
import pandas
from collections import Counter
import time as tm
import scipy.stats
import pandas as pd
from datetime import datetime


mins = 1.
sim_len = int(mins * 60000)


r_base_ex = np.arange(600,2100,100)
r_base_in = np.arange(400,900,50) 
r_gain = np.arange(500,2000,100)
r_base_cj = np.arange(40,150,10)

for base_ex in r_base_ex:
	for base_in in r_base_in:
		for gain in r_gain:
			for base_cj in r_base_cj:

				sim.ResetKernel()

				#params

				sh = 0

				y_dim = (0.5* np.sqrt(3))
				Nx = 20
				Ny = int(np.ceil(Nx * y_dim))
				N = Nx * Ny

				sigma = 0.5/6
				mu = 0.5
				delay = 0.1
				w_ex_cj = 500.

				I_init = 300.0 #pA
				I_init_dur = 100.0 #ms
				I_init_pos = 350


				tic = tm.time()
				txt = str(datetime.now())
				spl = txt.split(' ')
				spl2 = spl[1].split('.')
				txt = f'{spl[0]}_{spl2[0][:-3]}'


				#populations

				exc = sim.Create("iaf_psc_alpha",N, params={"I_e": 400.})
				inh = sim.Create("iaf_psc_alpha",N)

				l = sim.Create("iaf_psc_alpha",N)
				r = sim.Create("iaf_psc_alpha",N)
				u = sim.Create("iaf_psc_alpha",N)
				d = sim.Create("iaf_psc_alpha",N)

				#define connections

				w_ex = np.empty((N,N))
				w_in = np.empty((N,N))
				for e in range(N):
				    x_e = (e%Nx) / Nx
				    y_e = y_dim*(e//Nx)/ Ny
				    for i in range(N): 
				        x_i = (i%Nx) / Nx 
				        y_i = y_dim*(i//Nx) / Ny

				        d1 = np.sqrt(abs(x_e - x_i)**2 + abs(y_e - y_i)**2)
				        d2 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d3 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d4 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d5 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d6 = np.sqrt(abs(x_e - x_i - 1.)**2 + abs(y_e - y_i)**2)
				        d7 = np.sqrt(abs(x_e - x_i + 1.)**2 + abs(y_e - y_i)**2)
				        
				        d_ = min(d1,d2,d3,d4,d5,d6,d7)

				        w_gauss = np.exp(-(d_)**2/2/sigma**2)
				        w_ring = np.exp(-(d_ - mu)**2/2/sigma**2)

				        w_ex[i,e] = base_ex * w_ring
				        w_in[e,i] = base_in * w_gauss

				w_ex[w_ex<10]=0
				w_in[w_in<10]=0

				w_l = np.empty((N,N))
				w_r = np.empty((N,N))
				w_u = np.empty((N,N))
				w_d = np.empty((N,N))
				for e in range(N):
				    x_e = (e%Nx) / Nx
				    y_e = (e//Nx) / Ny * y_dim
				    for i in range(N): 
				        x_i = ((i%Nx) / Nx) - (1/Nx) 
				        y_i = (i//Nx) / Ny * y_dim

				        d1 = np.sqrt(abs(x_e - x_i)**2 + abs(y_e - y_i)**2)
				        d2 = np.sqrt(abs(x_e - x_i - 1.)**2 + abs(y_e - y_i)**2)
				        d3 = np.sqrt(abs(x_e - x_i + 1.)**2 + abs(y_e - y_i)**2)
				        d4 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d5 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d6 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d7 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d_ = min(d1,d2,d3,d4,d5,d6,d7)
				        w_l[i,e] = base_cj * (np.exp(-(d_)**2/2/sigma**2))
				        
				        x_i = ((i%Nx) / Nx) + (1/Nx) 
				        y_i = (i//Nx) / Ny * y_dim

				        d1 = np.sqrt(abs(x_e - x_i)**2 + abs(y_e - y_i)**2)
				        d2 = np.sqrt(abs(x_e - x_i - 1.)**2 + abs(y_e - y_i)**2)
				        d3 = np.sqrt(abs(x_e - x_i + 1.)**2 + abs(y_e - y_i)**2)
				        d4 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d5 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d6 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d7 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d_ = min(d1,d2,d3,d4,d5,d6,d7)
				        w_r[i,e] = base_cj * (np.exp(-(d_)**2/2/sigma**2))        

				        x_i = (i%Nx) / Nx 
				        y_i = ((i//Nx) / Ny * y_dim) + (1 / Ny * y_dim)

				        d1 = np.sqrt(abs(x_e - x_i)**2 + abs(y_e - y_i)**2)
				        d2 = np.sqrt(abs(x_e - x_i - 1.)**2 + abs(y_e - y_i)**2)
				        d3 = np.sqrt(abs(x_e - x_i + 1.)**2 + abs(y_e - y_i)**2)
				        d4 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d5 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d6 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d7 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d_ = min(d1,d2,d3,d4,d5,d6,d7)
				        w_u[i,e] = base_cj * (np.exp(-(d_)**2/2/sigma**2))
				        
				        x_i = (i%Nx) / Nx 
				        y_i = ((i//Nx) / Ny * y_dim) - (1 / Ny * y_dim)

				        d1 = np.sqrt(abs(x_e - x_i)**2 + abs(y_e - y_i)**2)
				        d2 = np.sqrt(abs(x_e - x_i - 1.)**2 + abs(y_e - y_i)**2)
				        d3 = np.sqrt(abs(x_e - x_i + 1.)**2 + abs(y_e - y_i + 1.)**2)
				        d4 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d5 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i - y_dim)**2)
				        d6 = np.sqrt(abs(x_e - x_i + 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d7 = np.sqrt(abs(x_e - x_i - 0.5)**2 + abs(y_e - y_i + y_dim)**2)
				        d_ = min(d1,d2,d3,d4,d5,d6,d7)
				        w_d[i,e] = base_cj * (np.exp(-(d_)**2/2/sigma**2))
				        
				m = np.amax(w_l)
				w_l[w_l<m] = 0
				m = np.amax(w_r)
				w_r[w_r<m] = 0
				m = np.amax(w_u)
				w_u[w_u<m] = 0
				m = np.amax(w_d)
				w_d[w_d<m] = 0


				exc_2_inh = sim.Connect(exc,inh,'all_to_all',syn_spec={'weight': w_ex, 'delay': delay})
				inh_2_exc = sim.Connect(inh,exc,'all_to_all',syn_spec={'weight': -w_in, 'delay': delay})

				exc_2_l = sim.Connect(exc,l,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})
				exc_2_r = sim.Connect(exc,r,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})
				exc_2_u = sim.Connect(exc,u,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})
				exc_2_d = sim.Connect(exc,d,'one_to_one',syn_spec={'weight': w_ex_cj, 'delay': delay})

				l_2_exc = sim.Connect(l,exc,'all_to_all',syn_spec={'weight': w_l, 'delay': delay})
				r_2_exc = sim.Connect(r,exc,'all_to_all',syn_spec={'weight': w_r, 'delay': delay})
				u_2_exc = sim.Connect(u,exc,'all_to_all',syn_spec={'weight': w_u, 'delay': delay})
				d_2_exc = sim.Connect(d,exc,'all_to_all',syn_spec={'weight': w_d, 'delay': delay})

				#network input

				number_of_turns = 20
				numT = number_of_turns * 1000 * np.pi
				dt = 20
				t = np.arange(0,sim_len,dt)*1.
				time = [i * 1. for i in t if i < sim_len]
				ts = np.arange(0,numT,numT/len(t))/1000.
				V = 30
				dr = 5
				ph = np.sqrt(((V * (4*np.pi) * ts) / dr))
				ra =  np.sqrt(((V * dr * ts) / np.pi))

				pos_x = ra * np.cos(ph) 
				pos_y = ra * np.sin(ph)

				spiral = np.vstack((time,pos_x,pos_y))

				vel_x = np.diff(pos_x)
				vel_y = np.diff(pos_y)

				vel_x,vel_y = vel_x*gain, vel_y*gain

				go_l,go_r = vel_x,-vel_x
				go_u,go_d = vel_y,-vel_y
				go_l, go_r, go_u, go_d = go_l+sh, go_r+sh, go_u+sh, go_d+sh
				go_l[go_l<=sh] = 0.
				go_r[go_r<=sh] = 0.
				go_u[go_u<=sh] = 0.
				go_d[go_d<=sh] = 0.

				l_input = sim.Create('step_current_generator', 1)
				sim.SetStatus(l_input,{'amplitude_times': t[1:],'amplitude_values': go_l})
				r_input = sim.Create('step_current_generator', 1)
				sim.SetStatus(r_input,{'amplitude_times': t[1:],'amplitude_values': go_r})
				u_input = sim.Create('step_current_generator', 1)
				sim.SetStatus(u_input,{'amplitude_times': t[1:],'amplitude_values': go_u})
				d_input = sim.Create('step_current_generator', 1)
				sim.SetStatus(d_input,{'amplitude_times': t[1:],'amplitude_values': go_d})

				sim.Connect(l_input,l,'all_to_all')
				sim.Connect(r_input,r,'all_to_all')
				sim.Connect(u_input,d,'all_to_all')
				sim.Connect(d_input,u,'all_to_all')

				exc_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True})
				sim.Connect(exc,exc_spikes)

				bump_init = sim.Create('step_current_generator', 1, params = {'amplitude_times':[0.1,0.1+I_init_dur],'amplitude_values':[I_init,0.0]})
				sim.Connect(bump_init,[exc[216]])

				tic = tm.time()
				sim.Simulate(sim_len)
				print(f'Simulation run time: {np.around(tm.time()-tic,2)}s  Simulated time: {np.around(sim_len/1000,2)}s')


				from collections import Counter 
				ev = sim.GetStatus(exc_spikes)[0]['events']
				t = ev['times']
				sp = ev['senders']

				occurence_count = Counter(sp) 
				print(sp)
				cell = occurence_count.most_common(5)[0][0]
				print(cell)

				spktms = t[sp==cell]
				spktms = (spktms//20)*20
				spktms=spktms[1:]

				xs = np.empty((len(spktms)))
				ys = np.empty((len(spktms)))

				for i,spk in enumerate(spktms):
				    xs[i] = pos_x[np.where(time == spk)[0][0]]
				    ys[i] = pos_y[np.where(time == spk)[0][0]]
				        
				fig = plt.figure(figsize=(5, 5),facecolor='w')
				plt.plot(pos_x[:len(time)],pos_y[:len(time)])
				plt.plot(xs,ys,'.')

				plt.savefig(f'{txt}.png')

				dictparams = {	'gain': gain, 
							'sh':sh, 
							'base_cj':base_cj, 
							'Nx':Nx, 
							'Ny':Ny, 
							'sigma':sigma, 
							'mu':mu, 
							'delay':delay, 
							'base_ex':base_ex, 
							'base_in':base_in,
							'w_ex_cj':w_ex_cj,
							'y_dim':y_dim}

				np.save(f'{txt}_params',dictparams)
