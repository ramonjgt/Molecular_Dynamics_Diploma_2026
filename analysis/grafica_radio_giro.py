import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el archivo CSV
file_path = 'promedio_radio_giro_completo.csv'
df = pd.read_csv(file_path)

# 2. Configuración de la gráfica
plt.figure(figsize=(10, 6))
plt.plot(df['T'], df['Radii'], marker='d', linestyle='-', color='green', label='Radio de Giro ($<R_g^2>$)')

# 3. Personalización de etiquetas y diseño
plt.title('Promedio del Radio de Giro ($<R_g^2>$) vs Temperatura ($T$)', fontsize=14)
plt.xlabel('Temperatura ($T$)', fontsize=12)
plt.ylabel('Radio de Giro ($<R_g^2>$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 4. Guardar y mostrar la gráfica
plt.tight_layout()
plt.savefig('grafica_rg_proteina.png')
plt.show()