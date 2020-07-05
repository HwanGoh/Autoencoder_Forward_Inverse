import tensorflow as tf
from Utilities.load_FEM_matrices import load_prematrices,\
        load_boundary_matrices_and_load_vector

def load_FEM_matrices_tf(run_options, file_paths):
    premass, prestiffness = load_prematrices(
            file_paths, run_options.parameter_dimensions)
    boundary_matrix, load_vector = load_boundary_matrices_and_load_vector(
            file_paths, run_options.parameter_dimensions)
    boundary_matrix = run_options.boundary_matrix_constant*boundary_matrix
    load_vector = -run_options.load_vector_constant*load_vector

    premass = tf.cast(premass, tf.float32)
    prestiffness = tf.cast(prestiffness, tf.float32)
    boundary_matrix = tf.cast(boundary_matrix, tf.float32)
    load_vector = tf.cast(load_vector, tf.float32)

    return premass, prestiffness, boundary_matrix, load_vector
