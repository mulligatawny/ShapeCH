import bpy
import sys
import os
import json

# ensure STL importer is available when running headless
if not bpy.utils.is_addon_enabled("io_mesh_stl"):
    bpy.ops.wm.addon_enable(module="io_mesh_stl")
bpy.ops.wm.addon_enable(module="io_mesh_stl")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_path(base_path, relative_path):
    """Resolve project-relative paths consistently with config usage."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.abspath(os.path.join(base_path, relative_path))


if len(sys.argv) <= 5:
    raise RuntimeError("Expected STL index after '--'. Example: blender --background --python remesh_stls.py -- 12")

stl_index = sys.argv[5]

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
config_path = os.path.join(repo_root, "config.json")

config = load_config(config_path)
resolution = config["resolution"]
samples_per_dim = config["samples_per_dim"]
project_dir = config["project_dir"]
project_dir = f"{project_dir}_r{resolution}_n{samples_per_dim}"

stl_source_dir = resolve_path(repo_root, os.path.join(project_dir, "stls"))
remeshed_dir = resolve_path(repo_root, os.path.join(project_dir, "stls-remeshed"))
os.makedirs(remeshed_dir, exist_ok=True)

source_stl_path = os.path.join(stl_source_dir, f"{stl_index}.stl")
if not os.path.exists(source_stl_path):
    raise FileNotFoundError(f"Source STL not found: {source_stl_path}")

target_stl_path = os.path.join(remeshed_dir, f"{stl_index}.stl")
if os.path.exists(target_stl_path):
    print(f"Output file {target_stl_path} already exists. Skipping.")
    sys.exit(0)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.import_mesh.stl(filepath=source_stl_path)

if not bpy.context.selected_objects:
    raise RuntimeError("No objects imported from STL.")

obj = bpy.context.selected_objects[0]
if obj.type != "MESH":
    raise RuntimeError("Imported object is not a mesh.")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.object.mode_set(mode="OBJECT")

decimate = obj.modifiers.new(name="Decimate", type="DECIMATE")
decimate.ratio = 0.025
decimate.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier="Decimate")

remesh = obj.modifiers.new(name="Remesh", type="REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = 2.0
remesh.use_remove_disconnected = True
bpy.ops.object.modifier_apply(modifier="Remesh")

bpy.ops.export_mesh.stl(filepath=target_stl_path)
print(f"Remeshed STL written to {target_stl_path}")
