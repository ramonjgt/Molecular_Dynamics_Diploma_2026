import gsd.hoomd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt # --- NEW: Importar librería de graficación ---

def calculate_cv_for_temperature(gsd_file, T):
    """
    Calculates the heat capacity Cv from the potential energy fluctuations.
    """
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"File not found or corrupted: {gsd_file}")
        return np.nan

    total_frames = len(traj)

    equilibration_frames = 4000
    
    if total_frames < equilibration_frames:
        print(f"Warning: {gsd_file} has only {total_frames} frames. Using the last half.")
        frames_to_analyze = traj[-int(total_frames * 0.5):]
    else:
        frames_to_analyze = traj[-equilibration_frames:]

    potential_energies = []
    
    N = frames_to_analyze[0].particles.N

    log_key = 'md/compute/ThermodynamicQuantities/potential_energy'

    for frame in frames_to_analyze:
        if frame.log and log_key in frame.log:
            pe = frame.log[log_key]
            if isinstance(pe, (list, np.ndarray)):
                pe = pe[0]
            potential_energies.append(pe)
        else:
            print(f"Error: Potential energy not found in the log of {gsd_file}. Did you add the logger?")
            return np.nan

    if not potential_energies:
        return np.nan

    U = np.array(potential_energies)
    
    # 1. Calculate the Variance of the Potential Energy (<U^2> - <U>^2)
    energy_variance = np.var(U, ddof=0)
    

    Cv_configurational = energy_variance / (T**2)
    
    # 3. Calculate the kinetic contribution to Specific Heat
    # For a 3D system, Cv_kinetic = (3/2) * N * kB
    Cv_kinetic = 1.5 * N
    
    # Total Heat Capacity
    Cv_total = Cv_configurational + Cv_kinetic
    
    return Cv_total

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    results = []
    
    # Iterate over the temperatures used in your simulation
    temperatures = np.arange(0.0, 2.0, 0.05)
    
    print("Calculating Specific Heat (Cv). Please wait...")
    for T in temperatures:
        # Format T to 2 decimal places to match your file naming convention
        T_str = f"{T:.2f}"
        filename = f"folding_t_{T_str}.gsd"
        
        if os.path.exists(filename):
            cv = calculate_cv_for_temperature(filename, T)
            if not np.isnan(cv):
                print(f"T = {T_str} | Cv = {cv:.4f}")
                results.append({'T': T, 'Cv': cv})
        else:
            print(f"Skipping T = {T_str}, file {filename} does not exist.")

    # Create a DataFrame, save the data, and plot
    if results:
        df = pd.DataFrame(results)
        df.to_csv("specific_heat.csv", index=False)
        print("\nCalculation finished! Results saved to 'specific_heat.csv'")
        
        # --- NEW: Plotting steps ---
        print("Generando gráfico de Calor Específico vs Temperatura...")
        plt.figure(figsize=(8, 6))
        plt.plot(df['T'], df['Cv'], marker='o', linestyle='-', color='r', label='Cv')
        
        plt.title('Specific Heat vs Temperature')
        plt.xlabel('Temperature (T)')
        plt.ylabel(r'Specific Heat ($C_v$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('specific_heat_vs_T.png', dpi=300)
        print("Gráfico guardado como 'specific_heat_vs_T.png'")
        # -----------------------------
    else:
        print("\nNo valid GSD files with logged energies were found.")