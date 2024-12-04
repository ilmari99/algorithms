import numpy as np
import trimesh
import plotly.graph_objects as go
import open3d as o3d
# Load the 3D face model using Open3D
mesh = o3d.io.read_triangle_mesh('/home/ilmari/python/algorithms/muscular-bodybuilder-boxing-fighter/source/model.glb')
# Swap axes [x,y,z] -> [z,x,y]
mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices)[:, [2, 0, 1]])
mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles)[:, [0, 2, 1]])
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# Extract vertices and faces
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
print(vertices.shape)
print(faces.shape)

# Scale the mesh to fit the unit cube
scale = 1.0 / np.max(np.abs(vertices))
vertices *= scale

# Center the mesh at the origin
vertices -= np.mean(vertices, axis=0)

# Extract vertex colors if available
if mesh.has_vertex_colors():
    vertex_colors = np.asarray(mesh.vertex_colors)
else:
    vertex_colors = np.ones((vertices.shape[0], 3))  # Default to white if no colors are available
print(vertex_colors.shape)

# Create a Plotly figure
fig = go.Figure()

# Add the mesh to the figure
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    opacity=1.0,
    vertexcolor=vertex_colors,
    colorscale='Viridis'
))

# Set the axis labels
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)
fig.show()

# Measure the distance from all x,y,-150 to the surface of the mesh, i.e. x,y,z0
# find the first collision of the ray starting at each (150,y,z), and going in the +x direction

# Define the rays: scale -150 - 150
y = np.linspace(-1, 1, 500)
z = np.linspace(-1, 1, 500)
Y, Z = np.meshgrid(y, z)
print(f"Y: {Y.shape}, Z: {Z.shape}")
ray_origins = np.column_stack((np.ones(Y.size), Y.ravel(), Z.ravel()))
ray_directions = np.column_stack((-np.ones(Y.size), np.zeros(Y.size), np.zeros(Y.size)))

print(f"Ray origins: {ray_origins.shape}, Ray directions: {ray_directions.shape}")

# Find the first collision of each ray with the mesh
raycasting_scene = o3d.t.geometry.RaycastingScene()
_ = raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)
results = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

# Extract collision locations:
#t_hit is the distance to the intersection. The unit is defined by the length of the ray direction. If there is no intersection this is inf
#geometry_ids gives the id of the geometry hit by the ray. If no geometry was hit this is RaycastingScene.INVALID_ID
#primitive_ids is the triangle index of the triangle that was hit or RaycastingScene.INVALID_ID
#primitive_uvs is the barycentric coordinates of the intersection point within the triangle.
#primitive_normals is the normal of the hit triangle.

# To get the collision locations, we can add the t_hit to the ray origins in the direction of the ray
collision_mask = results['t_hit'].isfinite()
collision_locations_from_positive_x = ray_origins + ray_directions * results['t_hit'].numpy()[:, None]
collision_locations_from_positive_x = collision_locations_from_positive_x[collision_mask.numpy()]

# Only take the origins that had a collision
ray_origins = ray_origins[collision_mask.numpy()]
distances_from_positive_x = np.linalg.norm(collision_locations_from_positive_x - ray_origins, axis=1)
print(f"Collision locations from positive x: {collision_locations_from_positive_x}")

# Create a Plotly figure
fig = go.Figure()

# Add the mesh to the figure
fig.add_trace(go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    opacity=1.0,
    vertexcolor=vertex_colors,
    colorscale='Viridis'
))

# Create a scatter plot of the collision points
fig.add_trace(go.Scatter3d(
    x=collision_locations_from_positive_x[:, 0],
    y=collision_locations_from_positive_x[:, 1],
    z=collision_locations_from_positive_x[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='red'
    )
))

# Now do the same from -150x to +x direction
ray_origins = np.column_stack((np.ones(Y.size)*-1, Y.ravel(), Z.ravel()))
ray_directions = np.column_stack((np.ones(Y.size), np.zeros(Y.size), np.zeros(Y.size)))

# Scale and center the ray origins
ray_origins *= scale
ray_origins -= np.mean(vertices, axis=0)

# Find the first collision of each ray with the mesh
rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)
results = raycasting_scene.cast_rays(o3d.core.Tensor(rays))

# Extract collision locations
collision_mask = results['t_hit'].isfinite()
collision_locations_from_negative_x = ray_origins + ray_directions * results['t_hit'].numpy()[:, None]
collision_locations_from_negative_x = collision_locations_from_negative_x[collision_mask.numpy()]

ray_origins = ray_origins[collision_mask.numpy()]
distances_from_negative_x = np.linalg.norm(collision_locations_from_negative_x - ray_origins, axis=1)

# Plot
fig.add_trace(go.Scatter3d(
    x=collision_locations_from_negative_x[:, 0],
    y=collision_locations_from_negative_x[:, 1],
    z=collision_locations_from_negative_x[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue'
    )
))

fig.show()

print(collision_locations_from_negative_x.shape)
print(collision_locations_from_negative_x)


# Create a reconstruction based on the collisions
fig = go.Figure()


# Combine the collision locations from positive and negative x directions
combined_collision_locations = np.vstack((collision_locations_from_positive_x, collision_locations_from_negative_x))

# Create a point cloud from the combined collision locations
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(combined_collision_locations)

# Create a triangle mesh from the point cloud
triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    point_cloud,
    alpha=0.15,
)

# Extract the vertices and faces
reconstructed_vertices = np.asarray(triangle_mesh.vertices)
reconstructed_faces = np.asarray(triangle_mesh.triangles)

# Add the combined mesh to the figure
fig.add_trace(go.Mesh3d(
    x=reconstructed_vertices[:, 0],
    y=reconstructed_vertices[:, 1],
    z=reconstructed_vertices[:, 2],
    i=reconstructed_faces[:, 0],
    j=reconstructed_faces[:, 1],
    k=reconstructed_faces[:, 2],
    opacity=1.0,
    color='purple'
))

fig.show()
