import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure
import trimesh
import pymeshlab
import os
from helpers import load_config
import time

# load the configuration file
config = load_config()

corner_stls = config["corner_stls"]
resolution = config["resolution"]
project_dir = config["project_dir"]
samples_per_dim = config["samples_per_dim"]
epsilon = config["epsilon"]
project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
print(project_dir)


def generate_barycentric_weights(num_corners, resolution):
    """Generate barycentric weights on an (n-1)-simplex grid."""
    if num_corners < 2:
        raise ValueError("At least two corner STLs are required for interpolation.")
    if resolution <= 0:
        raise ValueError("samples_per_dim must be positive.")

    integer_weights = []

    def backtrack(prefix, remaining, corners_left):
        if corners_left == 1:
            integer_weights.append(prefix + [remaining])
            return
        for coeff in range(remaining + 1):
            backtrack(prefix + [coeff], remaining - coeff, corners_left - 1)

    backtrack([], resolution, num_corners)
    weights = np.array(integer_weights, dtype=np.float32) / float(resolution)
    return weights


def interpolate_sdf_set(sdfs, weight_vector):
    return np.tensordot(weight_vector, sdfs, axes=(0, 0))


num_corners = len(corner_stls)
sdfs_dir = f"{project_dir}/sdfs"
stls_dir = f"{project_dir}/stls"
os.makedirs(sdfs_dir, exist_ok=True)
os.makedirs(stls_dir, exist_ok=True)

# load coordinate transform metadata
transform_metadata = np.load(f'{project_dir}/transform_metadata.npy', allow_pickle=True).item()
dx = transform_metadata["dx"]
origin = np.array(transform_metadata["origin"])
original_centers = np.array(transform_metadata["original_centers"])
smooth_iter = config["smooth_iter"]
print(f"Loaded transform metadata: dx={dx:.6f}, origin={origin}")
print(f"Original centers: {original_centers}")

corner_sdfs = []
t0 = time.time()
for idx in range(num_corners):
    corner = np.load(f"{project_dir}/corner_{idx}.npy")
    filled = binary_fill_holes(corner.astype(bool))
    sdf = distance(~filled) - distance(filled)
    corner_sdfs.append(sdf.astype(np.float32))
    np.save(f"{sdfs_dir}/corner_{idx}.npy", sdf.astype(np.float32))

corner_sdfs = np.stack(corner_sdfs, axis=0)
t1 = time.time()
print(f"Timings :: SDF (distance transform): {t1 - t0:.2f} (s)")

weights = generate_barycentric_weights(num_corners, samples_per_dim)
np.save(f"{project_dir}/weights.npy", weights)
for weight_idx in range(num_corners):
    np.save(f"{project_dir}/w{weight_idx}.npy", weights[:, weight_idx])

print(f"Total samples: {weights.shape[0]}")

for sample_idx, weight_vector in enumerate(weights):
    interpolated_sdf = interpolate_sdf_set(corner_sdfs, weight_vector)
    np.save(f"{sdfs_dir}/sample_{sample_idx}.npy", interpolated_sdf.astype(np.float32))

    interface = (interpolated_sdf >= -epsilon) & (interpolated_sdf <= epsilon)
    scalar_field = interface.astype(np.float32)

    # extract interface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
    
    # transform vertices back to original coordinate system
    vertices_physical = vertices * dx + origin
    
    # compute interpolated original center and apply reverse centering
    interpolated_center = np.dot(weight_vector, original_centers)
    vertices_physical += interpolated_center
    
    mesh = trimesh.Trimesh(vertices=vertices_physical, faces=faces)
    
    # export to temporary STL file for smoothing
    temp_stl_path = os.path.join(stls_dir, f'temp_{sample_idx}.stl')
    mesh.export(temp_stl_path)
    
    # apply laplacian smoothing
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_stl_path)
    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=smooth_iter)
    
    # save final STL
    final_stl_path = os.path.join(stls_dir, f'{sample_idx}.stl')
    ms.save_current_mesh(final_stl_path)
    
    # cleanup temporary file
    os.remove(temp_stl_path)
    
    print(sample_idx)

# end
