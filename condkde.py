# -*- coding: utf-8 -*-
"""
Conditional KDE estimation, univariate and multivariate
rlafarguette@imf.org, 2019
Time-stamp: "2019-12-01 15:52:30 rlafarguette"
"""

###############################################################################
#%% Modules and Functions Imports
###############################################################################
## Modules
import pandas as pd                                     # Dataframes
import numpy as np                                      # Numeric tools

# Functions
from sklearn.preprocessing import StandardScaler        # Standard zscore 
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

###############################################################################
#%% Local Ancilary Functions
###############################################################################
def string_sequence(character, length):
    """ String sequence of the same character for a given length """    
    string_seq = character
    for _ in range(1, length):
        string_seq = string_seq + character
    return(string_seq)

def gen_borders(array, delta_range=0.1):
    """ 
    Generate borders from a an array, with extension from min, max 

    delta_range: float or tuple
      In case of tuple, can adjust differently the left support with the right
    """
    if isinstance(delta_range, (int, float)):
        # Same extra distance on both sides
        delta = (np.max(array) - np.min(array))*delta_range
        array_min = np.min(array) - delta
        array_max = np.max(array) + delta
    elif isinstance(delta_range, tuple):
        f_range = (np.max(array) - np.min(array))
        array_min = np.min(array) - f_range*delta_range[0] 
        array_max = np.max(array) + f_range*delta_range[1] 
    else:
        raise ValueError('delta range parameter misspecified')
    
    return((array_min, array_max))

def gen_support(array, delta_range=4, len_support=100):
    """ Generate linearly equally spaced samples from an array"""
    array_min, array_max =  gen_borders(array, delta_range)
    support = np.linspace(array_min, array_max, len_support)
    return(support)

###############################################################################
#%% Local Ancilary Class
###############################################################################
class ZScaler(object):
    """
    Create a zscore scaler which adjusts the scaling based on historical data

    Parameters
    ----------
    historical_data: list, numpy array, pandas frame, etc.
      the data to fit the zscore upon

    data: list, numpy array, pandas frame, etc.
      the data to transform

    Usage:
    -----
    var_scaler = ZScaler([x for x in range(100)])
    var_scaler.transform(1)
    """
    def __init__(self, historical_data):
        self.hist_data = historical_data
        self.mean = np.nanmean(self.hist_data)
        self.std = np.nanstd(self.hist_data, ddof=0)

    # Public method for transformation (keep parameters from historical data)
    def transform(self, data):
        trans_data = (data - self.mean)/self.std
        return(trans_data)
        
###############################################################################
#%% Multivariate Conditional KDE
###############################################################################
class CondKDE(object):
    """ 
    Non Parametric Multivariate KDE Estimator

    Parameters
    ----------
    endog_l: list
      Endogeneous variables (can be multivariate) 

    exog_l: list
      Exogeneous variables

    data: pd.DataFrame
      Data to train the model on

    bw: str, "normal_reference", "cv_ls", "cv_ml"
      Approach to pick up the bandwidth

    Optional
    --------
    dep_type: either 'c', 'u' or 'o'
       
    Returns
    ------
    A CondKDE object
   
    Example
    -------
    nobs = 300
    c1 = np.random.normal(-10, 1, size=(nobs,1))
    c2 = np.random.normal(10, 1, size=(nobs,1))
    c3 = np.random.normal(3, 1, size=(nobs,1))
    c4 = np.random.normal(-2, 1, size=(nobs,1))

    dc = pd.DataFrame(c1, columns=['c1'])
    dc['c2'] = c2
    dc['c3'] = c3
    dc['c4'] = c4

    ckd = CondKDE(['c1', 'c2'], ['c3', 'c4'], dc)    
    """
    __description = "NonParametric Multivar KDE Estimator based on statsmodels"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"

    # Initializer
    def __init__(self, endog_l, exog_l, data, dep_type=None, indep_type=None,
                 bw='normal_reference', bw_adj=None):
        self.endog_l = endog_l
        self.exog_l = exog_l
        self.data = data[self.endog_l + self.exog_l].dropna().copy()
        self.num_endog = len(self.endog_l)
        self.num_exog = len(self.exog_l)
        self.bw = bw
        
        # Manage types (default values change with variables length)
        if dep_type:
            self.dep_type = dep_type
        else: # by default, consider that the variables are continuous
            self.dep_type = string_sequence('c', len(self.endog_l))
        
        if indep_type:
            self.indep_type = indep_type
        else: # by default, consider that the variables are continuous
            self.indep_type = string_sequence('c', len(self.exog_l))
                
        # Parameters for the non-parametric fit
        endog_values = list()
        for var in self.endog_l:
            endog_values.append(self.data[var].values)

        # Exogeneous variables    
        exog_values = list()
        
        # Normalize the input variables
        for var in self.exog_l:
            exog_values.append(self.data[var].values)
                
        np_params_dict = {'endog': endog_values,
                          'exog': exog_values, 
                          'dep_type': self.dep_type,
                          'indep_type': self.indep_type,
                          'bw': self.bw}
        
        # Fit the conditional distribution over the historical data
        cond_distrib = KDEMultivariateConditional(**np_params_dict)
        opt_bw = cond_distrib.bw # The bandwidth with optimal parameters
        
        # Bandwidth adjustment, if necessary
        if bw_adj==None:
            self.cond_distrib = cond_distrib
            
        elif isinstance(bw_adj, (int, float)):
            constrained_bw = opt_bw*bw_adj # Adjust the bandwidth
            np_params_dict['bw'] = constrained_bw # Update the parameters dict
            self.cond_distrib = KDEMultivariateConditional(**np_params_dict)

        else:
            raise ValueError('Bandwidth adjustment parameter misspecified')
        
        self.nobs = self.cond_distrib.nobs # Number of observations
        
    # Conditional fit, function wrapper as a separate class
    def cond_fit(self, exog_cond, delta_range=0.1, len_support=100,
                 borders=None):
        return(CondFit(self, exog_cond=exog_cond, delta_range=delta_range,
                       len_support=len_support, borders=borders))


# Define a "conditional fit" wrapper class called in the parent class
class CondFit(object):
    """
    Wrapper class for CondKDE
    Fit a conditional distribution from some points
    To keep a clean code, separate the historical fit from the conditional fit
    Following the way statsmodels is coded

    Parameters
    ----------
    CondKDE: a CondKDE object
        From the CondKDE class defined above

    exog_cond : dict or pandas dataframe (columns should be exog vars name)
        Conditioning values. Attention ! Order should be the same as exog_l

    len_support: int
        length of support for endogeneous variables

    delta_range: a positive float
        How to adjust the sample length. 

    Returns
    -------
    CondFit = A conditional fit object

    Example (cf. parent class example to obtain ckd)
    -------
    len_support = 100
    exog_cond = {'c3': np.mean(dc['c3']), 'c4': np.mean(dc['c4'])}
    cft = ckd.cond_fit(exog_cond, len_support=100)
    
    """
    __description = "Conditional Fit and Sampling"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org, 2019"

    # Initialization of the class
    def __init__(self, CondKDE, exog_cond, delta_range, len_support,
                 borders):

        # Initialization from parent
        self.__dict__.update(CondKDE.__dict__) # Pass all attributes 
        
        # Parameters checks
        assert isinstance(len_support, int), 'len_support should be int'

        # Treatment of the exogeneous variables, could be dict or frame
        # Make sure to have consistent shape
        msg_exog = 'Conditioning variables should contain exog variables'     
        if isinstance(exog_cond, dict):
            assert set(self.exog_l).issubset(set(exog_cond.keys())), msg_exog
            exog_c_l = list()
            for var in self.exog_l:
                exog_c_l.append(exog_cond[var])
                    
            # Reshape the set of exogenous conditions in a nice array 
            self.exog_cond = np.array(exog_c_l).reshape(1,-1) 
            
        elif isinstance(exog_cond, pd.DataFrame):
            assert set(self.exog_l).issubset(set(exog_cond.columns)), msg_exog
            self.exog_cond = pd.DataFrame()
            for var in self.exog_l:
                self.exog_cond[var] = exog_cond[var].values
                    
        # Wrapper class attributes        
        self.len_support = len_support

        # Generate the borders for endogeneous variables
        if isinstance(borders, tuple):
            borders_d = {v: borders for v in self.endog_l} 
        else:
            borders_d = {v: gen_borders(self.data[v].values,
                                        delta_range=delta_range)
                         for v in self.endog_l}        
                    
        # Generate the support
        endog_support_d = {v: np.linspace(*borders_d[v], self.len_support)
                           for v in self.endog_l}

        # Create a meshgrid to retrieve all combinations of endogeneous vars
        # (in the case of joint densities)
        # Note that I keep the indexation on self.endog_l to preserve order
        endog_support_l = [endog_support_d[v] for v in self.endog_l]
        endog_mgrid = np.meshgrid(*endog_support_l)

        # Reshape in a clean vertical array format 
        self.endog_grid = np.array(endog_mgrid).T.reshape(-1, self.num_endog)
        self.len_endog_grid = self.endog_grid.shape[0]
        
        # Broadcast the conditions grid to fit the endog_grid (unit repetition)
        self.exog_grid = np.repeat(self.exog_cond,
                                   repeats=[self.len_endog_grid],
                                   axis=0)

        # Need to repeat the endog_grid so that it fits (array repetition)
        self.rp_endog_grid = np.tile(self.endog_grid,
                                     (self.exog_cond.shape[0],1))
        
        # Estimate the pdf over the support
        # List format of all inputs
        endog_g_l = self.rp_endog_grid.T.tolist()
        exog_g_l = self.exog_grid.T.tolist()

        # Estimate the pdf
        self.pdf = self.cond_distrib.pdf(endog_predict=endog_g_l,
                                         exog_predict=exog_g_l).reshape(-1,1)

        # Estimate the cdf
        self.cdf = self.cond_distrib.cdf(endog_predict=endog_g_l,
                                         exog_predict=exog_g_l).reshape(-1,1)

        self.summary_frame = self.__summary_frame_call()

        
    # Private methods    
    def __summary_frame_call(self):
        """ 
        Store the results in a pandas dataframe
        """               
        endog_frame = pd.DataFrame(self.endog_grid, columns=self.endog_l)
        exog_frame = pd.DataFrame(self.exog_grid, columns=self.exog_l)
        pdf_frame = pd.DataFrame(self.pdf, columns=['pdf'])
        cdf_frame = pd.DataFrame(self.cdf, columns=['cdf'])
        final_frame = pd.concat([endog_frame, exog_frame,
                                 pdf_frame, cdf_frame], axis=1)
        
        # The pdf can be normalized only with a single conditioning set
        if self.exog_cond.shape[0]==1:
            final_frame['pdf_norm'] = (
                final_frame['pdf']/np.sum(final_frame['pdf']))
            
        return(final_frame)
    
    # Public Methods                        
    def ppf(self, tau):
        """ 
        Return the percentage point function 
        (quantile associated with a given probability) 
        Computation done with simple linear interpolation
        Parameters
        ----------
        tau = float or list of floats between 0 and 1

        Returns:
        -------
        The quantile of the endogeneous variable associated with a given tau
        """
        
        assert (tau>=0) & (tau<=1), 'tau should be between 0 and 1'
        tau_val = np.interp(tau, self.cdf.ravel(), self.endog_grid.ravel(),
                            left=0, right=1)
        return(tau_val)

    
    def sample(self, len_sample=None, frame=False):
        """ 
        Directly sample from the pdf grid 
        Using probabilistic random choice with replacement

        Parameters
        ----------
        len_sample: int
          Lenght of the sample to draw. Default: original grid length

        frame: Boolean
          Return a numpy array (default) or pandas dataframe

        Returns
        -------
        Sampled endogeneous vars matrix, for each conditioning set

        Example
        -------
        cft.sample(len_sample=500, frame=True)
        
        """
        
        # Default sample length
        if len_sample is None:
            len_sample = self.len_endog_grid # by default, grid length

        # Length of conditioning set
        cond_len = self.exog_cond.shape[0]
            
        # Subgrids: split densities and endogeneous support by conditioning set
        self.pdf_subgrid_l = np.split(self.pdf, cond_len, axis=0)
        self.endog_subgrid_l = np.split(self.rp_endog_grid, cond_len, axis=0)

        # PS: could have taken endog_grid directly, here make sure it matches
        exog_sample_grid = np.repeat(self.exog_cond, repeats=[len_sample],
                                     axis=0)

        # Create a generic index, according to the length of each sublist
        gen_index = np.arange(self.len_endog_grid)

        # Container
        endog_sample_l = list()
        
        # Sample over each set of probabilities and endogeneous subgrids
        # Using 1-D sampling on the index (fast)
        for pdf_a, endog_g in zip(self.pdf_subgrid_l, self.endog_subgrid_l):

            # Normalize the density
            pdf_n = pdf_a/np.sum(pdf_a)
            
            # For clarity, package the parameters in a dictionary
            sample_params_d = {
                'a': gen_index,
                'size': len_sample, # Size of the sample to draw
                'replace': True, # Important to draw with replacement
                'p': np.squeeze(pdf_n) # The estimated density
            }


            # Draw from the index using the cythonized np.random.choice
            random_index = np.random.choice(**sample_params_d)

            # True sample: simply pick the index of interest among the support
            endog_sample = endog_g[random_index]

            # Store
            endog_sample_l.append(endog_sample)

        # Stack all the sampling into a single array
        sample_stack = np.vstack(endog_sample_l)

        if frame==True:
            # Provide both the exogeneous and sampled endogeneous variables
            # Useful to check if the sample corresponds to the right conditions
            stack_mat = np.hstack([exog_sample_grid, sample_stack])
            mat_cols = self.exog_l + self.endog_l
            return(pd.DataFrame(stack_mat, columns=mat_cols))
        
        else:
            # Just return the endogeneous sample
            return(sample_stack)

                

    

    
