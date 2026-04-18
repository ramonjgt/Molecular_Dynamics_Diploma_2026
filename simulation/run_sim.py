import hoomd
import hoomd.md
import hoomd.write
import numpy as np
import itertools

def run_ht_folding(init_file="ht_model_initial.gsd", output_traj="folding_trajectory.gsd"):
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=init_file)
    
    harmonic_bond = hoomd.md.bond.Harmonic()
    harmonic_bond.params['backbone'] = dict(k=1000.0, r0=1.0)
    
    harmonic_angle = hoomd.md.angle.Harmonic()
    harmonic_angle.params['harmonic'] = dict(k=20.0, t0=1.8326)
    
    opls_dihedral = hoomd.md.dihedral.OPLS()
    opls_dihedral.params['normal'] = dict(k1=2.4, k2=0.0, k3=2.4, k4=0.0)
    opls_dihedral.params['bend'] = dict(k1=0.0, k2=0.0, k3=0.4, k4=0.0)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    tipos_particulas = ['B', 'L', 'N']
    
    lj = hoomd.md.pair.LJ(nlist=nl)
    for t1, t2 in itertools.product(tipos_particulas, repeat=2):
        lj.params[(t1, t2)] = dict(epsilon=0.0, sigma=1.0)
        lj.r_cut[(t1, t2)] = 0.0
        
    lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[('B', 'B')] = 2.5

    table_width = 1000
    r_min_val = 0.5 
    r_cut_tab = 2.5
    r_array = np.linspace(r_min_val, r_cut_tab, table_width, endpoint=False)
    
    V_L_alpha = 4.0 * (2.0/3.0) * (r_array**-12 + r_array**-6)
    F_L_alpha = 4.0 * (2.0/3.0) * (12.0*r_array**-13 + 6.0*r_array**-7)
    
    V_N_alpha = 4.0 * (r_array**-12)
    F_N_alpha = 4.0 * (12.0*r_array**-13)
    
    zero_array = np.zeros(table_width)
    table = hoomd.md.pair.Table(nlist=nl)
    
    for t1, t2 in itertools.product(tipos_particulas, repeat=2):
        table.params[(t1, t2)] = dict(r_min=r_min_val, U=zero_array, F=zero_array)
        table.r_cut[(t1, t2)] = 0.0

    table.params[('L', 'L')] = dict(r_min=r_min_val, U=V_L_alpha, F=F_L_alpha)
    table.params[('L', 'B')] = dict(r_min=r_min_val, U=V_L_alpha, F=F_L_alpha)
    table.params[('B', 'L')] = dict(r_min=r_min_val, U=V_L_alpha, F=F_L_alpha)
    table.r_cut[('L', 'L')] = table.r_cut[('L', 'B')] = table.r_cut[('B', 'L')] = r_cut_tab

    for t in tipos_particulas:
        table.params[('N', t)] = dict(r_min=r_min_val, U=V_N_alpha, F=F_N_alpha)
        table.params[(t, 'N')] = dict(r_min=r_min_val, U=V_N_alpha, F=F_N_alpha)
        table.r_cut[('N', t)] = table.r_cut[(t, 'N')] = r_cut_tab

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.extend([harmonic_bond, harmonic_angle, opls_dihedral, lj, table])
    
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=2.0)
    for particle_type in tipos_particulas:
        langevin.gamma[particle_type] = 1.0 
    
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    gsd_writer = hoomd.write.GSD(filename=output_traj,
                                 trigger=hoomd.trigger.Periodic(100),
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)

    sim.run(50000) 

    ramp = hoomd.variant.Ramp(A=2.0, B=0.1, t_start=sim.timestep, t_ramp=500000)
    langevin.kT = ramp
    
    sim.run(500000)

if __name__ == "__main__":
    run_ht_folding()