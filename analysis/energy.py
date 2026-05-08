import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt
import os

# Definir las mismas temperaturas utilizadas en tu script de generación
temperatures = np.arange(0, 2.05, 0.05)
log_key = 'md/compute/ThermodynamicQuantities/potential_energy'

valid_temps = []
avg_energies = []

# =============================================================================
# Gráfica 1: Energía vs Tiempo (Cuadros) para todas las simulaciones
# =============================================================================
plt.figure(figsize=(12, 6))

print("Leyendo archivos GSD y graficando series temporales...")
for T in temperatures:
    # Usar exactamente el mismo formato de f-string que usó tu script de generación
    filename = f"folding_t_{T}.gsd"
    
    if not os.path.exists(filename):
        continue
        
    try:
        traj = gsd.hoomd.open(filename, 'r')
    except (FileNotFoundError, RuntimeError):
        print(f"El archivo {filename} está corrupto o es ilegible.")
        continue

    energies = []
    
    # Extraer energía de cada cuadro (frame)
    for frame in traj:
        if frame.log and log_key in frame.log:
            pe = frame.log[log_key]
            if isinstance(pe, (list, np.ndarray)):
                pe = pe[0]
            energies.append(pe)
            
    if not energies:
        print(f"Advertencia: No se encontraron datos de energía en {filename}.")
        continue

    # Graficar la evolución temporal
    # Usamos un mapa de colores para asignar colores fríos a T baja y cálidos a T alta
    color = plt.cm.jet(T / 2.0)
    plt.plot(energies, color=color, alpha=0.6, linewidth=1.5)
    
    # --- Calcular la Energía Promedio en Equilibrio ---
    # Solo promediamos el último 30% de los cuadros para asegurar que solo
    # medimos el sistema DESPUÉS de que la rampa de recocido simulado haya terminado.
    equilibration_cutoff = int(len(energies) * 0.7)
    eq_energies = energies[equilibration_cutoff:]
    
    avg_energies.append(np.mean(eq_energies))
    valid_temps.append(T)

# Formatear la gráfica de series temporales
plt.title('Energía Potencial vs Tiempo de Simulación (Todas las Temperaturas)')
plt.xlabel('Cuadro Guardado (Frame)')
plt.ylabel('Energía Potencial (U)')
plt.grid(True, linestyle='--', alpha=0.7)

# Añadir una barra de color para representar las temperaturas
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=2.0))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Temperatura Objetivo (T)')

plt.tight_layout()
plt.savefig('energia_vs_tiempo.png', dpi=300)
print("Guardado 'energia_vs_tiempo.png'")

# =============================================================================
# Gráfica 2: Energía Promedio en Equilibrio vs Temperatura Objetivo
# =============================================================================
if valid_temps and avg_energies:
    plt.figure(figsize=(8, 6))
    
    plt.plot(valid_temps, avg_energies, marker='o', linestyle='-', color='indigo', markersize=6)
    
    plt.title('Energía Potencial Promedio en Equilibrio vs Temperatura')
    plt.xlabel('Temperatura (T)')
    plt.ylabel(r'$\langle U \rangle$')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('energia_promedio_vs_T.png', dpi=300)
    print("Guardado 'energia_promedio_vs_T.png'")
    
    # Guardar los datos en un CSV también para uso futuro
    import pandas as pd
    df = pd.DataFrame({'T': valid_temps, 'Energia_Potencial_Prom': avg_energies})
    df.to_csv('energias_promedio.csv', index=False)
    print("Guardado 'energias_promedio.csv'")
else:
    print("No hay suficientes datos para graficar Energía Promedio vs Temperatura.")