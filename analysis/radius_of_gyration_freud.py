import gsd.hoomd
import freud
import numpy as np

# 1. Open the GSD trajectory file
# Replace 'trajectory.gsd' with your actual file path
traj = gsd.hoomd.open('folding_t_1_3.gsd', 'r')

# 2. Select a specific frame (e.g., the last frame in the trajectory)
frame = traj[-1]

# 3. Extract the simulation box and particle positions
box = frame.configuration.box
positions = frame.particles.position

# 4. Define your clusters (molecules/polymers)
# freud needs to know which particles belong together to calculate Rg.
# 
# Scenario A: The whole system is a SINGLE molecule.
# All particles get the same cluster ID (0).
cluster_keys = np.zeros(len(positions), dtype=np.int32)

# Scenario B: Multiple molecules (e.g., 50 polymers, each with 200 beads).
# You need an array mapping each particle to its molecule ID.
# cluster_keys = np.repeat(np.arange(50), 200)

# Scenario C: If your GSD file already contains molecule tags, extract them:
# (Note: Not all GSD files have this unless you explicitly saved it)
# cluster_keys = frame.particles.body  # or however you stored molecule tags

# 5. Initialize and compute Cluster Properties
cl_props = freud.cluster.ClusterProperties()

# Calculate properties using the system (box and positions) and cluster keys.
# We also pass the particle masses (defaults to 1.0 if not provided) to ensure 
# the center of mass and Rg are weighted correctly if you have varying masses.
masses = frame.particles.mass
system = (box, positions)

cl_props.compute(system, cluster_idx=cluster_keys, masses=masses)

# 6. Extract the radii of gyration
# This returns a NumPy array containing the Rg for each unique cluster_key
rg_array = cl_props.radii_of_gyration

# Print the result
for cluster_id, rg in enumerate(rg_array):
    print(f"Cluster/Molecule {cluster_id}: Radius of Gyration = {rg:.4f}")