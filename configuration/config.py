import os

config = {
    'root_dir': os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."),
    'path_to_remove': "/cm/local/apps/cuda/libs/current/pynvml",
    'data_file': "data/smiles/guacamol_initial_rediscovery_ref_3.smi",
    'batch_size': 40,
    'initial_size': 100,
    'workers': 1,
    'threads': 2,
    'generations': 75,
    'archive': {
        'name': "ref_3",
        'size': 150,
        'accuracy': 25000
    },
    'descriptor': {
        'properties': ["Descriptors.ExactMolWt"],
        'ranges': [225, 555]
    },
    'spec_params': {
        'rocs_type': "shape_and_color",
        'max_confs': 75,
        'max_centers': 3,
        'force_flip': True,
        'enum_nitrogen': True
    },
    'fitness': {
        'target': "data/sdf/fastrocs_Ref_3.sdf",
        'type': "ROCS_3D"
    },
    'arbiter': {
        'rules': "Glaxo"
    }  
}
