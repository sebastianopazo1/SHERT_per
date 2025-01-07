import open3d as o3d
import numpy as np

def visualize_point_cloud_with_highlights(ply_file_path, highlighted_ply_path=None):
    """
    Visualiza una nube de puntos y, opcionalmente, puntos destacados como esferas rojas pequeñas.

    Args:
        ply_file_path (str): Ruta del archivo PLY que contiene la nube de puntos.
        highlighted_ply_path (str): Ruta del archivo PLY con los puntos destacados (opcional).
    """
    # Cargar la nube de puntos principal
    pcd = o3d.io.read_point_cloud(ply_file_path)

    if pcd.is_empty():
        print(f"La nube de puntos en {ply_file_path} está vacía o no se pudo cargar.")
        return

    geometries = [pcd]

    # Cargar los puntos destacados si están disponibles
    if highlighted_ply_path:
        highlighted_pcd = o3d.io.read_point_cloud(highlighted_ply_path)
        if highlighted_pcd.is_empty():
            print(f"El archivo de puntos destacados en {highlighted_ply_path} está vacío o no se pudo cargar.")
        else:
            # Convertir los puntos destacados en esferas rojas pequeñas
            for point in np.asarray(highlighted_pcd.points):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # Radio de la esfera pequeña
                sphere.translate(point)  # Mover la esfera a la posición del punto destacado
                sphere.paint_uniform_color([1, 0, 0])  # Color rojo
                geometries.append(sphere)

    # Visualización
    print("Presiona 'q' para salir de la visualización.")
    o3d.visualization.draw_geometries(geometries)

# Ejemplo de uso
if __name__ == "__main__":
    ply_file_path = "../../Descargas/output.ply"  # Cambia a la ruta de tu archivo PLY
    highlighted_ply_path = "../../Descargas/output_highlighted.ply"  # Cambia a la ruta de puntos destacados si existe

    visualize_point_cloud_with_highlights(ply_file_path, highlighted_ply_path)
