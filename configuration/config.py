config = {
    'data_file': "data/smiles/guacamol_initial_rediscovery_ref_1.smi",
    'batch_size': 40,
    'initial_size': 100,
    'workers': 1,
    'threads': 2,
    'generations': 75,
    'archive': {
        'name': "ref_1",
        'size': 150,
        'accuracy': 25000
    },
    'descriptor': {
        'properties': "ROCS",
        'ranges': [0, 2]
    },
    'spec_params': {
        'rocs_type': "shape_and_color",
        'max_confs': 75,
        'max_centers': 3,
        'force_flip': True,
        'enum_nitrogen': True
    },
    'fitness': {
        'target': "data/sdf/ref_1_single_conf.sdf",
        'type': "ROCS_3D"
    },
    'arbiter': {
        'rules': "Glaxo"
    }  
}
