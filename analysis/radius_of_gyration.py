import gsd.hoomd
import freud
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def calculate_average_rg(gsd_file):
    # Intentamos abrir el archivo
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"Archivo no encontrado o corrupto: {gsd_file}")
        return np.nan, np.nan

    total_frames = len(traj)
    
    # Los últimos 500,000 steps son a temperatura constante (equilibrio).
    # Esto equivale a los últimos 5000 frames. 
    # Por seguridad, tomaremos los últimos 4000 frames para asegurar que el sistema 
    # ya se relajó completamente en esa temperatura.
    equilibration_frames = 4000
    
    if total_frames < equilibration_frames:
        print(f"Advertencia: {gsd_file} tiene pocos frames ({total_frames}).")
        frames_to_analyze = traj[-int(total_frames*0.5):] # Usa la última mitad si hay pocos
    else:
        frames_to_analyze = traj[-equilibration_frames:]

    rg_squared_list = []
    rg_list = []
    
    # Inicializamos freud
    cl_props = freud.cluster.ClusterProperties()
    
    for frame in frames_to_analyze:
        box = frame.configuration.box
        positions = frame.particles.position
        
        # Asumiendo que todo el sistema es un solo polímero
        cluster_keys = np.zeros(len(positions), dtype=np.int32)
        system = (box, positions)
        
        # Calcular propiedades
        cl_props.compute(system, cluster_idx=cluster_keys)
        
        # Extraer el radio de giro de este frame
        rg = cl_props.radii_of_gyration[0]
        
        rg_list.append(rg)
        rg_squared_list.append(rg**2)
        
    # Calcular promedios
    avg_rg = np.mean(rg_list)
    avg_rg_squared = np.mean(rg_squared_list)
    
    return avg_rg, avg_rg_squared

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    results = []
    
    # Iterar sobre las temperaturas (igual que en tu script de simulación)
    temperatures = np.arange(0.0, 2.0, 0.05)
    
    print("Calculando radios de giro. Por favor, espera...")
    for T in temperatures:
        # Formateamos T a 1 decimal para que coincida con el nombre de tus archivos
        T_str = f"{T:.2f}"
        filename = f"folding_t_{T_str}.gsd"
        
        if os.path.exists(filename):
            avg_rg, avg_rg2 = calculate_average_rg(filename)
            print(f"T = {T_str} | <Rg> = {avg_rg:.4f} | <Rg^2> = {avg_rg2:.4f}")
            results.append({'T': T, 'Radii': avg_rg, 'Rg_squared': avg_rg2})
        else:
            print(f"Saltando T = {T_str}, archivo {filename} no existe.")

    # Crear un DataFrame y guardar los datos
    if results:
        df = pd.DataFrame(results)
        df.to_csv("paper_radius_gyration.csv", index=False)
        print("\n¡Cálculo terminado! Los resultados se han guardado en 'paper_radius_gyration.csv'")
        
        # --- NEW: Plotting steps ---
        print("Generando gráfico de Radio de Giro al Cuadrado vs Temperatura...")
        plt.figure(figsize=(8, 6))
        
        # Graficamos T vs Rg_squared (que es lo que se reporta en la Fig 5 del paper)
        plt.plot(df['T'], df['Rg_squared'], marker='o', linestyle='-', color='b', label=r'$\langle R_g^2 \rangle$')
        
        plt.title('Squared Radius of Gyration vs Temperature')
        plt.xlabel('Temperature (T)')
        plt.ylabel(r'$\langle R_g^2 \rangle$')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('rg_squared_vs_T.png', dpi=300)
        print("Gráfico guardado exitosamente como 'rg_squared_vs_T.png'")
        # -----------------------------
    else:
        print("\nNo se encontraron archivos GSD para analizar.")