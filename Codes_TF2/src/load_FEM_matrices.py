import pandas as pd

def load_prematrices(file_paths, num_nodes):
    #=== Premass ===#
    df_premass = pd.read_csv(file_paths.premass_savefilepath + '.csv')
    premass = df_premass.to_numpy()
    premass = premass.reshape((num_nodes**2, num_nodes))

    #=== Prestiffness ===#
    df_prestiffness = pd.read_csv(file_paths.prestiffness_savefilepath + '.csv')
    prestiffness = df_prestiffness.to_numpy()
    prestiffness = prestiffness.reshape((num_nodes**2, num_nodes))

    return premass, prestiffness

def load_boundary_matrices_and_load_vector(file_paths, num_nodes):
    #=== Boundary Matrix ===#
    df_boundary_matrix = pd.read_csv(file_paths.boundary_matrix_savefilepath + '.csv')
    boundary_matrix = df_boundary_matrix.to_numpy()
    boundary_matrix = boundary_matrix.reshape((num_nodes, num_nodes))

    #=== Load Vector ===#
    df_load_vector = pd.read_csv(file_paths.load_vector_savefilepath + '.csv')
    load_vector = df_load_vector.to_numpy()

    return boundary_matrix, load_vector