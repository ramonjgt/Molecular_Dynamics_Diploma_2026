import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el archivo CSV
file_path = 'promedio_energia_completo.csv'
df = pd.read_csv(file_path)

# 2. Configuración de la gráfica
plt.figure(figsize=(10, 6))
plt.plot(df['T'], df['Energia_Potencial_Prom'], marker='o', linestyle='-', color='b', label='Energía Promedio')

# 3. Personalización de etiquetas y título
plt.title('Promedio de Energía Potencial vs Temperatura (T)', fontsize=14)
plt.xlabel('Temperatura (T)', fontsize=12)
plt.ylabel('Energía Potencial Promedio', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 4. Mostrar y/o guardar la gráfica
plt.tight_layout()
plt.savefig('grafica_energia_proteina.png')
plt.show()