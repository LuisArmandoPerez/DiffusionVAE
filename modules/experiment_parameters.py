import pandas as pd
class ExperimentParams(object):
    '''
    classdocs
    '''

    # number of epochs
    # batch size 
    
    # which diffusion vae to use?


    def __init__(self, epochs, batch_size):
        '''
        Constructor
        '''
        self.epochs = epochs
        self.batch_size = batch_size

    def params_to_df(self):
        data = {"epochs": [self.epochs],
                "batch_size": [self.batch_size]}
        df = pd.DataFrame(data)
        return df
    
