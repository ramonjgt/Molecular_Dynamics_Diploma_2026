import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos
file_path = 'promedio_calor_especifico_completo.csv'
df = pd.read_csv(file_path)

# 2. Configuración de la gráfica
plt.figure(figsize=(10, 6))
plt.plot(df['T'], df['Cv'], marker='s', linestyle='-', color='red', label='Calor Específico ($C_v$)')

# 3. Personalización
plt.title('Calor Específico Promedio ($C_v$) vs Temperatura ($T$)', fontsize=14)
plt.xlabel('Temperatura ($T$)', fontsize=12)
plt.ylabel('Calor Específico ($C_v$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 4. Guardar y mostrar
plt.tight_layout()
plt.savefig('grafica_cv_proteina.png')
plt.show()