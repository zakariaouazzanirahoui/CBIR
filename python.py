
import trimesh
from scipy.spatial.distance import cosine
import numpy as np
import os
import json

from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory
import trimesh
from scipy.spatial.distance import cosine
import numpy as np
import os
import json
from flask_cors import CORS, cross_origin



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
#app.config['CORS_HEADERS'] = 'Content-Type'

class Object3D:
    def get_features(self, your_mesh):
        # Calculate the inertia tensor
        inertia_tensor = your_mesh.moment_inertia

        # Calculate the eigenvalues and eigenvectors of the inertia tensor
        # The eigenvectors are the principal axes of inertia
        evals, evecs = np.linalg.eig(inertia_tensor)

        # Get the first principal axis
        first_axis = evecs[:, 0]

        # Compute the dot product of the vertex normals with the first axis to get the distance
        distances = np.dot(your_mesh.vertex_normals, first_axis)

        # Compute the moment of inertia about the first axis
        moment_inertia = np.dot(distances, distances)

        # Compute the average distance
        average_distance = np.mean(distances)

        # Compute the variance of distance
        variance_distance = np.var(distances)

        # Create a feature vector containing the calculated values
        feature_vector = [first_axis, moment_inertia, average_distance, variance_distance]

        return feature_vector

    def reduce_mesh(self, mesh, face_count):
        # The face_count parameter is the target number of faces in the reduced mesh

        # Make sure the mesh is watertight
        if not mesh.is_watertight:
            mesh.fill_holes()

        # Make sure the mesh is consistently oriented
        if not mesh.is_winding_consistent:
            mesh.fix_normals()

        # Reduce the mesh
        reduced_mesh = mesh.simplify_quadric_decimation(face_count)
        return reduced_mesh

    def get_features_from_file(self, file_path):
        # Load the mesh from the file and calculate the features
        mesh = trimesh.load_mesh(file_path)
        object3d = Object3D()
        features = object3d.get_features(mesh)
        return features

    def search_similar_file(self, features, file_path):
        target_features = features[-3:]  # Extract the last 3 features
        similarities = []

        for file_name in os.listdir(file_path):
            file_features = self.get_features_from_file(os.path.join(file_path, file_name))  # Get the features of the file
            file_last_features = file_features[-3:]  # Extract the last 3 features of each file
            similarity = 1 - cosine(target_features, file_last_features)
            similarities.append((file_name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_files = [file_name for file_name, _ in similarities[:5]]

        return top_similar_files

    

@app.route('/search', methods=['POST', 'GET'])
def search():
    try: 
        if not request.data:
            raise ValueError("Empty request data")
        data = json.loads(request.data)
        image_name = data.get('image_name')
        image_name = image_name.replace(".jpg", "")  # Remove the .jpg extension
        print(image_name)
        object3d = Object3D()
        mesh1 = trimesh.load_mesh(f'3D Models\\all models\\{image_name}.obj')
        
        reduced_mesh1 = object3d.reduce_mesh(mesh1, 1000)

        features = object3d.get_features(reduced_mesh1)
        first_axis, moment_inertia, average_distance, variance_distance = [feature.tolist() if isinstance(feature, np.ndarray) else feature for feature in features[:4]]

        top_similar_images = object3d.search_similar_file(features, '3D Models\\all models')
        print(top_similar_images)
        similar_images = [os.path.splitext(file_path)[0] for file_path in top_similar_images]
        return jsonify({"first_axis": first_axis, "moment_inertia": moment_inertia, "average_distance": average_distance, "variance_distance": variance_distance, "similar_images": similar_images})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run()


