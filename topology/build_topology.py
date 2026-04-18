import gsd
import gsd.hoomd
import numpy as np

def build_ht_model(filename="ht_model_initial.gsd"):
    """
    Construye la topología inicial para el modelo de proteína de grano grueso (coarse-grained)
    de Honeycutt y Thirumalai. Genera una cadena lineal completamente desplegada.
    """

    seq = (
        ['B'] * 9 + 
        ['N'] * 3 + 
        ['L', 'B'] * 4 + 
        ['N'] * 3 + 
        ['B'] * 9 + 
        ['N'] * 3 + 
        ['L', 'B'] * 5 + 
        ['L']
    )
    n_particles = len(seq)

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.dimensions = 3
    
    frame.configuration.box = [150.0, 150.0, 150.0, 0, 0, 0]

    frame.particles.N = n_particles
    
    frame.particles.types = ['B', 'L', 'N']
    
    type_map = {'B': 0, 'L': 1, 'N': 2}
    frame.particles.typeid = [type_map[res] for res in seq]
    
    frame.particles.mass = np.ones(n_particles, dtype=float)

    # Coordenadas iniciales: Colocamos las partículas en una línea recta a lo largo del eje X.
    positions = np.zeros((n_particles, 3), dtype=np.float32)
    positions[:, 0] = np.arange(n_particles) * 1.0
    positions[:, 0] -= np.mean(positions[:, 0])
    frame.particles.position = positions

    # Unimos la partícula i con la i+1 consecutivamente.
    frame.bonds.N = n_particles - 1
    frame.bonds.types = ['backbone']
    frame.bonds.group = np.array([[i, i+1] for i in range(n_particles - 1)])
    
    frame.bonds.typeid = np.zeros(n_particles - 1, dtype=int)

    # Definimos el ángulo formado por tres partículas consecutivas: i, i+1, i+2
    frame.angles.N = n_particles - 2
    frame.angles.types = ['harmonic']
    frame.angles.group = np.array([[i, i+1, i+2] for i in range(n_particles - 2)])
    
    # Solo hay un tipo de ángulo armónico (typeid = 0), con objetivo de 105 grados
    frame.angles.typeid = np.zeros(n_particles - 2, dtype=int)

    # Un diedro define la torsión en 3D de cuatro partículas consecutivas: i, i+1, i+2, i+3
    frame.dihedrals.N = n_particles - 3
    
    # El paper define dos tipos de diedros: 
    # 'normal': Fuerza de torsión estándar que favorece la conformación trans (estirada).
    # 'bend'  : Fuerza más débil que permite a la cadena doblarse sobre sí misma.
    frame.dihedrals.types = ['normal', 'bend']
    
    d_groups = []
    d_typeids = []
    
    for i in range(n_particles - 3):
        group = [i, i+1, i+2, i+3]
        d_groups.append(group)
        
        # Obtenemos los tipos de aminoácidos involucrados en este diedro de 4 partículas
        res_in_dihedral = [seq[j] for j in group]
        
        # Si hay al menos un residuo Neutro ('N') 
        # en el conjunto de 4 partículas, este diedro se considera una zona de giro ('bend').
        if 'N' in res_in_dihedral:
            d_typeids.append(1)
        else:
            d_typeids.append(0)

    frame.dihedrals.group = np.array(d_groups)
    frame.dihedrals.typeid = np.array(d_typeids)

    with gsd.hoomd.open(name=filename, mode='w') as f:
        f.append(frame)
        
    print(f"Topología inicial construida exitosamente en: {filename}")
    print(f"La simulación contendrá: {n_particles} partículas, {frame.bonds.N} enlaces, {frame.angles.N} ángulos y {frame.dihedrals.N} diedros.")

if __name__ == "__main__":
    build_ht_model()