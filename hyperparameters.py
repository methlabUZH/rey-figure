"""
Training and evaluation parameters 
"""

train_params = {
    '116 150': {
        'multilabel-classifier': {
            "workers": 10, "is_binary": 0, "eval_test": True, "id": "final", "epochs": 75, 
            "batch_size": 64, "lr": 0.01, "gamma": 0.95, "wd": 0, "weighted_sampling": 1, 
            "augment": 0, "workers": 10, "image_size": [116, 150], "simulated_data": None, 
            "max_simulated": -1
        },
        'regressor': {
            "simulated_data": None, "workers": 10, "eval_test": True, "id": "final-small", "epochs": 75, 
            "batch_size": 64, "lr": 0.01, "gamma": 0.95, "wd": 0.0, "weighted_sampling": 1, "augment": 0, 
            "image_size": [116, 150], "max_simulated": -1, "arch": "v2"
        }
    },

    '232 300': {
        'multilabel-classifier': {
            "simulated_data": None, "max_simulated": -1, "workers": 8, "is_binary": 0, "eval_test": True, 
            "id": "final-bigsize", "epochs": 75, "batch_size": 16, "lr": 0.01, "gamma": 0.95, "wd": 0, 
            "weighted_sampling": 1, "augment": 0, "image_size": [232, 300]
        },
        'regressor': {
            "simulated_data": None, "max_simulated": -1, "workers": 8, "is_binary": 0, "eval_test": True, 
            "id": "final-bigsize-aug", "epochs": 75, "batch_size": 16, "lr": 0.01, "gamma": 0.95, "wd": 0, 
            "weighted_sampling": 1, "augment": 1, "image_size": [232, 300]
        }
    }
}

eval_params = {
    '116 150': {
        'multilabel-classifier': {
            "workers": 10, "batch_size": 100, "image_size": [116, 150], 'tta': True,
            'angles': [-2, -1, 0, 1, 2], 'validation': False
        },
        'regressor': {
            "workers": 10, "batch_size": 100, "image_size": [116, 150], 'tta': True,
            'angles': [-2, -1, 0, 1, 2], 'validation': False
        }
    },

    '232 300': {
        'multilabel-classifier': {
            "workers": 10, "batch_size": 100, "image_size": [232, 300], 'tta': True,
            'angles': [-2, -1, 0, 1, 2], 'validation': False
        },
        'regressor': {
            "workers": 10, "batch_size": 100, "image_size": [232, 300], 'tta': True,
            'angles': [-2, -1, 0, 1, 2], 'validation': False
        }
    }
}
