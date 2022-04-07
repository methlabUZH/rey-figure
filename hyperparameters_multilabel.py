"""
Training and evaluation parameters 
"""

train_params = {
    '78 100': {
        "workers": 10, "is_binary": 0, "eval_test": True, "id": "final", "epochs": 75, "batch_size": 64, "lr": 0.01,
        "gamma": 0.95, "wd": 0, "weighted_sampling": 1, "image_size": [78, 100], "simulated_data": None,
        "max_simulated": -1
    },
    '116 150': {
        "workers": 10, "is_binary": 0, "eval_test": True, "id": "final", "epochs": 75, "batch_size": 64, "lr": 0.01,
        "gamma": 0.95, "wd": 0, "weighted_sampling": 1, "image_size": [116, 150], "simulated_data": None,
        "max_simulated": -1
    },
    '232 300': {
        "simulated_data": None, "max_simulated": -1, "workers": 8, "is_binary": 0, "eval_test": True, "id": "final",
        "epochs": 75, "batch_size": 16, "lr": 0.01, "gamma": 0.95, "wd": 0, "weighted_sampling": 1,
        "image_size": [232, 300]
    },
    '348 450': {
        "simulated_data": None, "max_simulated": -1, "workers": 8, "is_binary": 0, "eval_test": True, "id": "final",
        "epochs": 75, "batch_size": 8, "lr": 0.01, "gamma": 0.95, "wd": 0, "weighted_sampling": 1,
        "image_size": [348, 450]
    }
}

# eval_params = {
#     '116 150': {
#         "workers": 10, "batch_size": 100, "image_size": [116, 150], 'tta': True,
#         'angles': [-2, -1, 0, 1, 2], 'validation': False},
#     '232 300': {
#         "workers": 10, "batch_size": 100, "image_size": [232, 300], 'tta': True,
#         'angles': [-2, -1, 0, 1, 2], 'validation': False
#     },
# }
