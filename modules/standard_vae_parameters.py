import pandas as pd
class StandardVAEParams(object):
    '''
    classdocs
    '''

    # latent dimension?


    def __init__(self,image_size,
                 latent_dim = 3,
                 intermediate_dim = 512,
                 var_x = 1.0,
                 r_loss = "mse"):
        '''
        Constructor
        '''

        # Data parameters
        self.image_size = image_size

        # Architecture parameters
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.r_loss = r_loss



        # VAE parameters
        self.var_x = var_x

    def params_to_df(self):
        """
        Function for extracting the parameters into a pandas dataframe
        :return: pandas dataframe
        """
        data = {"image_size": [self.image_size],
                "intermediate_dim": [self.intermediate_dim],
                "latent_dim": [self.latent_dim],
                "r_loss": [self.r_loss],
                "max_log_t": "None",
                "min_log_t": "None",
                "var_x": [self.var_x],
                "steps": "None",
                "truncation_radius": "None"}
        df = pd.DataFrame(data)
        return df


        
        
