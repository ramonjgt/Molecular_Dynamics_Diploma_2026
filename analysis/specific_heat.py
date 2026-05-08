import gsd.hoomd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt # --- NUEVO: Importar librería de graficación ---

def calcular_cv_por_temperatura(gsd_file, T):
    """
    Calcula la capacidad calorífica Cv a partir de las fluctuaciones de la energía potencial.
    """
    try:
        traj = gsd.hoomd.open(gsd_file, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"Archivo no encontrado o corrupto: {gsd_file}")
        return np.nan

    total_frames = len(traj)
    
    # Solo analizamos los últimos 4000 cuadros (que representan la fase final a temperatura constante)
    # para asegurar que el sistema esté en equilibrio y no en la rampa de enfriamiento.
    equilibration_frames = 4000
    
    if total_frames < equilibration_frames:
        print(f"Advertencia: {gsd_file} solo tiene {total_frames} cuadros. Usando la última mitad.")
        frames_to_analyze = traj[-int(total_frames * 0.5):]
    else:
        frames_to_analyze = traj[-equilibration_frames:]

    potential_energies = []
    
    # Obtener el número de partículas (N) del primer cuadro
    N = frames_to_analyze[0].particles.N

    # El nombre de la clave utilizada por HOOMD-blue para registrar la energía potencial
    log_key = 'md/compute/ThermodynamicQuantities/potential_energy'

    for frame in frames_to_analyze:
        if frame.log and log_key in frame.log:
            # Extraer la energía potencial (usualmente guardada como un arreglo de 1 elemento)
            pe = frame.log[log_key]
            if isinstance(pe, (list, np.ndarray)):
                pe = pe[0]
            potential_energies.append(pe)
        else:
            print(f"Error: Energía potencial no encontrada en el log de {gsd_file}. ¿Añadiste el logger?")
            return np.nan

    if not potential_energies:
        return np.nan

    # Convertir a un arreglo de numpy para cálculos matemáticos rápidos
    U = np.array(potential_energies)
    
    # 1. Calcular la Varianza de la Energía Potencial (<U^2> - <U>^2)
    # ddof=0 devuelve la varianza, que es la que usa la mecánica estadística
    energy_variance = np.var(U, ddof=0)
    
    # 2. Calcular el Calor Específico configuracional (Cv_conf = Var(U) / (kB * T^2))
    # En unidades reducidas, la constante de Boltzmann kB = 1
    # Verificación para evitar división por cero si T=0.0
    if T == 0.0:
        Cv_configuracional = 0.0
    else:
        Cv_configuracional = energy_variance / (T**2)
    
    # 3. Calcular la contribución cinética al Calor Específico
    # Para un sistema 3D, Cv_cinetico = (3/2) * N * kB
    Cv_kinetic = 1.5 * N
    
    # Capacidad Calorífica Total
    Cv_total = Cv_configuracional + Cv_kinetic
    
    return Cv_total

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    results = []
    
    # Iterar sobre las temperaturas utilizadas en la simulación
    temperatures = np.arange(0.0, 2.0, 0.05)
    
    print("Calculando Calor Específico (Cv). Por favor espera...")
    for T in temperatures:
        # Formatear T a 2 decimales para coincidir con la convención de nombres de archivo
        T_str = f"{T:.2f}"
        filename = f"folding_t_{T_str}.gsd"
        
        if os.path.exists(filename):
            cv = calcular_cv_por_temperatura(filename, T)
            if not np.isnan(cv):
                print(f"T = {T_str} | Cv = {cv:.4f}")
                results.append({'T': T, 'Cv': cv})
        else:
            print(f"Omitiendo T = {T_str}, el archivo {filename} no existe.")

    # Crear un DataFrame, guardar los datos y graficar
    if results:
        df = pd.DataFrame(results)
        df.to_csv("calor_especifico.csv", index=False)
        print("\n¡Cálculo finalizado! Resultados guardados en 'calor_especifico.csv'")
        
        # --- Pasos de graficación ---
        print("Generando gráfico de Calor Específico vs Temperatura...")
        plt.figure(figsize=(8, 6))
        plt.plot(df['T'], df['Cv'], marker='o', linestyle='-', color='r', label='Cv')
        
        plt.title('Calor Específico vs Temperatura')
        plt.xlabel('Temperatura (T)')
        plt.ylabel(r'Calor Específico ($C_v$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('calor_especifico_vs_T.png', dpi=300)
        print("Gráfico guardado como 'calor_especifico_vs_T.png'")
        # -----------------------------
    else:
        print("\nNo se encontraron archivos GSD válidos con energías registradas.")