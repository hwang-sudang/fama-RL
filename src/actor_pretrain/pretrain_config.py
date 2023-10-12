'''
def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})

'''




# 2: Define the search space
sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'RMSE'
        },
    'parameters': 
    {
        'window': {'values': [60]},
        'batch_size': {'values': [32, 64]},
        'epochs' : {'values': [1000, 2000, 3000]},
        'lr': {'max': 1e-4, 'min': 1e-6, 'distribution': 'uniform'},
        'layer_size': {'values': [32,64,128]},

     }
}