import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

temperatures = np.arange(0, 2.05, 0.05)
log_key = 'md/compute/ThermodynamicQuantities/potential_energy'

valid_temps = []
avg_energies = []

plt.figure(figsize=(12, 6))

print("Reading GSD files and plotting Time-Series...")
for T in temperatures:
    filename = f"folding_t_{T}.gsd"
    
    if not os.path.exists(filename):
        continue
        
    try:
        traj = gsd.hoomd.open(filename, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"File {filename} is corrupted or unreadable.")
        continue

    energies = []
    
    for frame in traj:
        if frame.log and log_key in frame.log:
            pe = frame.log[log_key]
            if isinstance(pe, (list, np.ndarray)):
                pe = pe[0]
            energies.append(pe)
            
    if not energies:
        print(f"Warning: No energy data found in {filename}.")
        continue

    color = plt.cm.jet(T / 2.0)
    plt.plot(energies, color=color, alpha=0.6, linewidth=1.5)
    
    equilibration_cutoff = int(len(energies) * 0.7)
    eq_energies = energies[equilibration_cutoff:]
    
    avg_energies.append(np.mean(eq_energies))
    valid_temps.append(T)

plt.title('Potential Energy vs Simulation Time (All Temperatures)')
plt.xlabel('Saved Frame')
plt.ylabel('Potential Energy (U)')
plt.grid(True, linestyle='--', alpha=0.7)

sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=2.0))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Target Temperature (T)')

plt.tight_layout()
plt.savefig('energy_vs_time.png', dpi=300)
print("Saved 'energy_vs_time.png'")

if valid_temps and avg_energies:
    plt.figure(figsize=(8, 6))
    
    plt.plot(valid_temps, avg_energies, marker='o', linestyle='-', color='indigo', markersize=6)
    
    plt.title('Average Equilibrium Potential Energy vs Temperature')
    plt.xlabel('Temperature (T)')
    plt.ylabel(r'$\langle U \rangle$')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('average_energy_vs_T.png', dpi=300)
    print("Saved 'average_energy_vs_T.png'")
    
    df = pd.DataFrame({'T': valid_temps, 'Avg_Potential_Energy': avg_energies})
    df.to_csv('average_energies.csv', index=False)
    print("Saved 'average_energies.csv'")
else:
    print("Not enough data to plot Average Energy vs Temperature.")