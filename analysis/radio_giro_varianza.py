import gsd.hoomd
import freud
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def calcular_varianza_rg(gsd_file):
    """
    Calcula el promedio y la varianza del radio de giro (Rg) y del 
    radio de giro al cuadrado (Rg^2) para analizar fluctuaciones estructurales.
    """
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"Archivo no encontrado o corrupto: {gsd_file}")
        return np.nan, np.nan, np.nan, np.nan

    total_frames = len(traj)
    
    # Solo analizamos los últimos 4000 cuadros (fase de equilibrio)
    equilibration_frames = 4000
    
    if total_frames < equilibration_frames:
        print(f"Advertencia: {gsd_file} solo tiene {total_frames} cuadros. Usando la última mitad.")
        frames_to_analyze = traj[-int(total_frames * 0.5):]
    else:
        frames_to_analyze = traj[-equilibration_frames:]

    rg_list = []
    rg_squared_list = []
    
    # Inicializamos freud para calcular propiedades del polímero
    cl_props = freud.cluster.ClusterProperties()
    
    for frame in frames_to_analyze:
        box = frame.configuration.box
        positions = frame.particles.position
        
        # Asumiendo que todo el sistema es un solo polímero (cluster único)
        cluster_keys = np.zeros(len(positions), dtype=np.int32)
        system = (box, positions)
        
        # Calcular propiedades del cluster
        cl_props.compute(system, cluster_idx=cluster_keys)
        
        # Extraer el radio de giro de este cuadro
        rg = cl_props.radii_of_gyration[0]
        
        rg_list.append(rg)
        rg_squared_list.append(rg**2)
        
    if not rg_list:
        return np.nan, np.nan, np.nan, np.nan

    # Convertir a arreglos de numpy
    rg_array = np.array(rg_list)
    rg2_array = np.array(rg_squared_list)
    
    # 1. Calcular promedios
    avg_rg = np.mean(rg_array)
    avg_rg2 = np.mean(rg2_array)
    
    # 2. Calcular la varianza (Fluctuaciones estructurales)
    # ddof=0 devuelve la varianza poblacional (<X^2> - <X>^2)
    var_rg = np.var(rg_array, ddof=0)
    var_rg2 = np.var(rg2_array, ddof=0)
    
    return avg_rg, avg_rg2, var_rg, var_rg2

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    results = []
    
    # Iterar sobre las temperaturas
    temperatures = np.arange(0.0, 2.0, 0.05)
    
    print("Calculando fluctuaciones (varianza) del Radio de Giro. Por favor espera...")
    for T in temperatures:
        T_str = f"{T:.2f}"
        filename = f"folding_t_{T_str}.gsd"
        
        if os.path.exists(filename):
            avg_rg, avg_rg2, var_rg, var_rg2 = calcular_varianza_rg(filename)
            
            if not np.isnan(var_rg):
                print(f"T = {T_str} | Var(Rg) = {var_rg:.4f} | Var(Rg^2) = {var_rg2:.4f}")
                results.append({
                    'T': T, 
                    'Rg_promedio': avg_rg, 
                    'Rg2_promedio': avg_rg2, 
                    'Varianza_Rg': var_rg,
                    'Varianza_Rg2': var_rg2
                })
        else:
            print(f"Omitiendo T = {T_str}, el archivo {filename} no existe.")

    # Crear DataFrame, guardar datos y graficar
    if results:
        df = pd.DataFrame(results)
        df.to_csv("varianza_radio_giro.csv", index=False)
        print("\n¡Cálculo finalizado! Resultados guardados en 'varianza_radio_giro.csv'")
        
        # --- GRAFICAR VARIANZA DE Rg ---
        print("Generando gráfico de Varianza del Radio de Giro vs Temperatura...")
        plt.figure(figsize=(8, 6))
        
        # Graficamos T vs Varianza de Rg (Fluctuaciones)
        plt.plot(df['T'], df['Varianza_Rg'], marker='s', linestyle='-', color='darkorange', label=r'Var($R_g$)')
        
        plt.title('Fluctuaciones Estructurales (Varianza de $R_g$) vs Temperatura')
        plt.xlabel('Temperatura (T)')
        plt.ylabel(r'Varianza del Radio de Giro $\langle R_g^2 \rangle - \langle R_g \rangle^2$')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('varianza_rg_vs_T.png', dpi=300)
        print("Gráfico guardado como 'varianza_rg_vs_T.png'")
        
        # --- OPCIONAL: GRAFICAR VARIANZA DE Rg^2 ---
        plt.figure(figsize=(8, 6))
        plt.plot(df['T'], df['Varianza_Rg2'], marker='^', linestyle='-', color='purple', label=r'Var($R_g^2$)')
        plt.title('Fluctuaciones de $R_g^2$ vs Temperatura')
        plt.xlabel('Temperatura (T)')
        plt.ylabel(r'Varianza de $R_g^2$')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('varianza_rg2_vs_T.png', dpi=300)
        print("Gráfico adicional guardado como 'varianza_rg2_vs_T.png'")

    else:
        print("\nNo se encontraron archivos GSD válidos para analizar.")