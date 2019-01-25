import pandas as pd
class DiffusionVAEParams(object):
    '''
    classdocs
    '''

    # latent dimension?


    def __init__(self,image_size,
                 steps = 10,
                 intermediate_dim = 512,
                 truncation_radius = 0.5,
                 max_log_t = -5.0,
                 min_log_t = -7.5,
                 var_x = 1.0,
                 r_loss = "mse",
                 d = 2,
                 constant_t=False,
                 log_t_fixed = -2.0,
                 unconstrained_t = False):
        '''
        Constructor
        '''

        # Data parameters
        self.image_size = image_size

        # Architecture parameters
        self.intermediate_dim = intermediate_dim
        self.r_loss = r_loss
        self.unconstrained_t = unconstrained_t


        # VAE parameters
        self.max_log_t = max_log_t
        self.min_log_t = min_log_t
        self.var_x = var_x
        self.steps = steps
        self.truncation_radius = truncation_radius
        self.constant_t = constant_t
        self.log_t_fixed = log_t_fixed
        # Manifold parameters
        self.d = d

    def params_to_df(self):
        data = {"image_size": [self.image_size],
                "intermediate_dim": [self.intermediate_dim],
                "latent_dim": [self.intermediate_dim],
                "r_loss": [self.r_loss],
                "max_log_t": [self.max_log_t],
                "min_log_t": [self.min_log_t],
                "var_x": [self.var_x],
                "steps": [self.steps],
                "truncation_radius": [self.truncation_radius]}
        df = pd.DataFrame(data)
        return df


        
        
