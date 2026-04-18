import gsd.hoomd
import freud
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def calculate_average_rg(gsd_file):
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"File not found or corrupted: {gsd_file}")
        return np.nan, np.nan

    total_frames = len(traj)
    equilibration_frames = 4000
    
    if total_frames < equilibration_frames:
        print(f"Warning: {gsd_file} has few frames ({total_frames}).")
        frames_to_analyze = traj[-int(total_frames*0.5):]
    else:
        frames_to_analyze = traj[-equilibration_frames:]

    rg_squared_list = []
    rg_list = []
    
    cl_props = freud.cluster.ClusterProperties()
    
    for frame in frames_to_analyze:
        box = frame.configuration.box
        positions = frame.particles.position
        
        cluster_keys = np.zeros(len(positions), dtype=np.int32)
        system = (box, positions)
        
        cl_props.compute(system, cluster_idx=cluster_keys)
        
        rg = cl_props.radii_of_gyration[0]
        
        rg_list.append(rg)
        rg_squared_list.append(rg**2)
        
    avg_rg = np.mean(rg_list)
    avg_rg_squared = np.mean(rg_squared_list)
    
    return avg_rg, avg_rg_squared

if __name__ == "__main__":
    results = []
    temperatures = np.arange(0.0, 2.0, 0.05)
    
    print("Calculating radii of gyration. Please wait...")
    for T in temperatures:
        T_str = f"{T:.2f}"
        filename = f"folding_t_{T_str}.gsd"
        
        if os.path.exists(filename):
            avg_rg, avg_rg2 = calculate_average_rg(filename)
            print(f"T = {T_str} | <Rg> = {avg_rg:.4f} | <Rg^2> = {avg_rg2:.4f}")
            results.append({'T': T, 'Radii': avg_rg, 'Rg_squared': avg_rg2})
        else:
            print(f"Skipping T = {T_str}, file {filename} does not exist.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv("paper_radius_gyration.csv", index=False)
        print("\nCalculation finished! Results saved to 'paper_radius_gyration.csv'")
        
        print("Generating Squared Radius of Gyration vs Temperature plot...")
        plt.figure(figsize=(8, 6))
        
        plt.plot(df['T'], df['Rg_squared'], marker='o', linestyle='-', color='b', label=r'$\langle R_g^2 \rangle$')
        
        plt.title('Squared Radius of Gyration vs Temperature')
        plt.xlabel('Temperature (T)')
        plt.ylabel(r'$\langle R_g^2 \rangle$')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('rg_squared_vs_T.png', dpi=300)
        print("Plot successfully saved as 'rg_squared_vs_T.png'")
    else:
        print("\nNo GSD files found to analyze.")