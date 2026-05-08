import hoomd
import hoomd.md
import hoomd.write
import numpy as np
import itertools

def run_ht_folding(init_file, output_traj, temp_init, temp_finish):
    
    # 1. INICIALIZAR LA SIMULACIÓN
    # El paper utiliza "Noisy Molecular Dynamics" para simular el entorno del solvente.
    # [Ref: Pág. 695, Synopsis; Pág. 700, Simulation Methods]
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=888)
    sim.create_state_from_gsd(filename=init_file)
    
    # =========================================================================
    # 2. POTENCIALES ENLAZADOS (Basado en el modelo de cadena de carbonos alfa)
    # =========================================================================
    
    # Los enlaces se consideran de longitud fija (r0=1.0) en el paper.
    # [Ref: Pág. 698, Columna 2, Secc. Construction of the Model; Pág. 700, Columna 2]
    # k=1000 simula la rigidez para aproximar la restricción rígida mencionada en la Pág. 699.
    harmonic_bond = hoomd.md.bond.Harmonic()
    harmonic_bond.params['backbone'] = dict(k=1000.0, r0=1.0)
    
    # El ángulo de equilibrio theta_0 es 105° (1.8326 rad).
    # [Ref: Pág. 699, Eq. 5 y Fig. 2]
    harmonic_angle = hoomd.md.angle.Harmonic()
    harmonic_angle.params['harmonic'] = dict(k=20.0, t0=1.8326)
    
    # Potencial diedro: Crucial para estabilizar la forma de barril-beta.
    # [Ref: Pág. 698, Columna 2, Secc (c); Pág. 699, Eq. 13]
    # Se usan parámetros A=B=1.2 para promover la configuración trans/gauche.
    # [Ref: Pág. 699, Columna 2, Dihedral Angle Potential]
    opls_dihedral = hoomd.md.dihedral.OPLS()
    opls_dihedral.params['normal'] = dict(k1=2.4, k2=0.0, k3=2.4, k4=0.0)
    
    # En las regiones de giro (loops/N), el potencial es más débil (A=0, B=0.2) 
    # para facilitar la formación de curvas.
    # [Ref: Pág. 699, Columna 2, Párrafo final; Pág. 700, Fig. 2]
    opls_dihedral.params['bend'] = dict(k1=0.0, k2=0.0, k3=0.4, k4=0.0)

    # =========================================================================
    # 3. POTENCIALES NO ENLAZADOS (Interacciones de largo alcance)
    # =========================================================================
    nl = hoomd.md.nlist.Cell(buffer=0.4)
    # B: Hidrofóbico, L: Hidrofílico, N: Neutro [Ref: Pág. 698, Columna 2]
    tipos_particulas = ['B', 'L', 'N']
    
    # A) Interacciones Lennard-Jones: Representan la atracción hidrofóbica.
    # [Ref: Pág. 698, Columna 2, Secc (a); Pág. 699, Eq. 2]
    # Solo los pares B-B se atraen para inducir el colapso de la proteína.
    # [Ref: Pág. 698, Columna 2, "B residues attract other B residues"]
    lj = hoomd.md.pair.LJ(nlist=nl)
    for t1, t2 in itertools.product(tipos_particulas, repeat=2):
        lj.params[(t1, t2)] = dict(epsilon=0.0, sigma=1.0)
        lj.r_cut[(t1, t2)] = 0.0
        
    lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[('B', 'B')] = 2.5

    # B) Interacciones Repulsivas Tabuladas
    # Implementa V_L_alpha y V_N_alpha descritos en el paper.
    # [Ref: Pág. 699, Eqs. 3 y 4; Pág. 700, Fig. 2]
    table_width = 1000
    r_min_val = 0.5 
    r_cut_tab = 2.5
    r_array = np.linspace(r_min_val, r_cut_tab, table_width, endpoint=False)
    
    # Potencial para residuos L: Puramente repulsivo con epsilon_L = 2/3 epsilon_h.
    # [Ref: Pág. 699, Eq. 3 y Columna 1 "purely repulsive"]
    V_L_alpha = 4.0 * (2.0/3.0) * (r_array**-12 + r_array**-6)
    F_L_alpha = 4.0 * (2.0/3.0) * (12.0*r_array**-13 + 6.0*r_array**-7)
    
    # Potencial para residuos N: Repulsión de esfera blanda (r^-12).
    # [Ref: Pág. 698, Columna 2, "short-range repulsion"; Pág. 699, Eq. 4]
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

    # =========================================================================
    # 4. INTEGRACIÓN Y TERMOSTATO (Langevin / Stochastic Dynamics)
    # =========================================================================
    # El uso de fricción (gamma) y ruido controla la temperatura físicamente.
    # [Ref: Pág. 700, Columna 2, "noisy molecular dynamics"; Pág. 701, Eq. 11]
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.extend([harmonic_bond, harmonic_angle, opls_dihedral, lj, table])
    
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=2.0)
    for particle_type in tipos_particulas:
        langevin.gamma[particle_type] = 1.0 
    
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

# =========================================================================
    # 5. PROTOCOLO DE PLEGAMIENTO (Simulated Annealing) & LOGGING
    # =========================================================================
    
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(thermo, quantities=['potential_energy', 'kinetic_energy'])

    gsd_writer = hoomd.write.GSD(filename=output_traj,
                                 trigger=hoomd.trigger.Periodic(100),
                                 mode='wb')
                                 
    # --- Imprimimos la información del logger ---
    gsd_writer.logger = logger  
    # ------------------------------------------------------------
    
    sim.operations.writers.append(gsd_writer)

    print("Iniciando fase de desnaturalización (T=2.0)...")
    sim.run(50000) 

    print("Iniciando enfriamiento lento (Simulated Annealing)...")
    ramp = hoomd.variant.Ramp(A=temp_init, B=temp_finish, t_start=sim.timestep, t_ramp=500000)
    langevin.kT = ramp
    
    # --- Comenzamos el enfriamiento ---
    sim.run(500000) # Esta línea ejecuta únicamente la rampa de enfriamiento.
    
    print(f"Iniciando equilibración a T={temp_finish}...")
    sim.run(500000) # Corremos otros 500,000 pasos para asegurar el equilibrio.
    # --------------------------------------------

    print(f"Simulación terminada. Trayectoria guardada en: {output_traj}")

if __name__ == "__main__":
    for i in np.arange(0, 2.05, 0.05):
        run_ht_folding(init_file="ht_model_initial.gsd", output_traj=f'folding_t_{i:.2f}.gsd', temp_init = 2.0, temp_finish = i)