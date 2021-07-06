
from os.path import dirname, join, abspath
import erquy_py

urdf_name = "idefX"
urdf_path = join(dirname(str(abspath(__file__))), "data", urdf_name, urdf_name + ".urdf")
meshes_path = join(dirname(str(abspath(__file__))), "data", urdf_name)

viz = erquy_py.Visualizer(urdf_path, meshes_path)
world = erquy_py.World()
world.loadUrdf (urdf_path, meshes_path)


print("It works !!!")