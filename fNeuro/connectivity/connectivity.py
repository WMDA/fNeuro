import numpy as np
import cyclicityanalysis.orientedarea as cao
import cyclicityanalysis.coom as coom
import pandas as pd

class Cyclic_analysis:
    def __init__(self, to_vectorize=True) -> None:
        '''
        Class to run Cyclic analysis on time series data

        Parameters
        ----------
        to_vectorize: bool (default is true)
            If true then will do nilearn Connectivity measure type vectorization

        Return
        ------
        None

        '''
        self.to_vectorize = to_vectorize

    def remove_diagonals(self, array) -> np.array:
        '''
        Function to remove Diagnoals. Does this when
        array is vectorised

        Parameters
        ---------
        array: np.array
            2D matrix

        Returns
        -------
        np.array: array
            2D matrix 
        '''
        return array[~np.eye(len(array), dtype=bool)].reshape(len(array), -1) 

    def vectorize(self, array: np.array) -> np.array:
        '''
        Nilearn's vectorize to return the lower triangluation of a matrix
        in 1D format.
    
        Parameters
        ----------
        array: np.array 
            Correlation matrix
        
        Returns
        -------
        np.array: 1D numpy array
            array of lower triangluation of a matrix
        '''
        symmetric = self.remove_diagonals(array)
        return symmetric[np.tril(symmetric, -1).astype(bool)]

    def cyclic_analysis(self, time_series: np.array) -> np.array:

        '''
        Function to perform cyclic analysis on an individual time series
        
        Parameters
        ----------
        time_series: np.array
            time series
        
        Returns
        -------
        np.array: 1D numpy array
            array of lower triangluation of a matrix
        '''
    
        df = pd.DataFrame(time_series)
        oriented_area = cao.OrientedArea(df)
        lead_lag_df = oriented_area.compute_lead_lag_df()
        if self.to_vectorize == True:
            return self.vectorize(lead_lag_df.values)
        else:
            return lead_lag_df.values
    
    def fit(self, time_series: np.array) -> np.array:
    
        '''
        Function to perform a cyclic analysis of time series
    
        Parameters
        ----------
        time_series: np.array
            time series
        
        Returns
        -------
        np.array: array 
            either a  1D numpy array
            array of lower triangluation of a matrix
            OR 2D matrix
        '''
        
        return np.array(list(map(self.cyclic_analysis, time_series)))
    
def adj_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    
    '''
    Function to create an Adjacency matrix
    needed for plotting

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values

    column: str
        str of column of value to use in 
        adj matrix
    
    Returns
    ------
    adj_matrix: pd.DataFrame
        Adjacency matrix
    '''
    
    adj = pd.DataFrame(data={
        df['correlation_names'].values[0].split('-')[0].rstrip(): df[column],
        df['correlation_names'].values[0].split('-')[1].rstrip().lstrip(): df[column]
    })
    
    adj_matrix = pd.DataFrame(np.zeros((adj.shape[1], adj.shape[1])), 
                              columns=adj.columns, index=adj.columns)
    adj_matrix.iloc[0,1] = df[column]
    adj_matrix.iloc[1,0] = df[column]
    return adj_matrix

def connectome_plotting(df: pd.DataFrame, column: str, labels: pd.DataFrame ) -> dict:
    
    '''
    
    Function to get adj matrix and 
    co-ordinates needed for plotting

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values

    column: str
        str of column of value to use in 
        adj matrix
    
    labels: pd.Dataframe
        Dataframe with co-ordinates of 
        regions
    
    Returns
    -------
    dict: dictionary object
        dict with adj matrix and 
        co-ordinates for plotting
    '''
    adj = adj_matrix(df, column)
    coords = (labels[labels['labels'].str.contains(adj.columns[0])]['region_coords'].reset_index(drop=True)[0],
          labels[labels['labels'].str.contains(adj.columns[1])]['region_coords'].reset_index(drop=True)[0])
    return {
        'adj': adj,
        'coords': coords
        }

def cyclic_order(lead_lag_df: pd.DataFrame, order: int) -> pd.DataFrame:

    '''
    Function to determine the sequential order of 
    time-series

    Parameters
    ----------
    lead_lag_df: pd.DataFrame
        lead lag correlation 
        matrix

    order: int
        order

    Returns
    -------
    pd.DataFrame: Dataframe
        DataFrame ordered by sequential order
        with eigenvalue moduli, leading eigenvector
        and leading eigenvector component phases
    
    '''

    eigen = coom.COOM(lead_lag_df)
    leading_eigenvector,leading_eigenvector_component_phases, sequential_order_dict = eigen.compute_sequential_order(order)
    df = pd.DataFrame(data={
        'regions': lead_lag_df.columns,
        'eigenvalue_moduli': eigen.eigenvalue_moduli,
        'leading_eigenvector': leading_eigenvector,
        'leading_eigenvector_component_phases': leading_eigenvector_component_phases,
    })
    order_df = pd.DataFrame(sequential_order_dict.values()).rename(columns={0: 'regions'})
    return pd.merge(df, order_df, how='right', on='regions')


def get_network_names(msdl_overview_df: pd.DataFrame, df: pd.DataFrame) -> list:
    
    '''
    Function to return network names

    Parameters
    ----------
    msdl_overview_df: pd.DataFrame
        Dataframe with labels and networks 
    df: pd.DataFrame
        long form df with correlations, 

    Returns
    -------
    network_names: list
        list of network names

    '''
    
    network_names = []
    for correlation in df['corr_names']:
        splitted_name = correlation.split('-')
        region_one = splitted_name[0].rstrip()
        region_two = splitted_name[1].lstrip()
        network_one = msdl_overview_df[msdl_overview_df['labels'] == region_one]['networks'].values[0]
        network_two = msdl_overview_df[msdl_overview_df['labels'] == region_two]['networks'].values[0]
        network_names.append(f"{network_one} - {network_two}")
    return network_names

def get_correlation_long_df(correlation_matrix, msdl_overview_df):
    
    '''
    Function to turn a correlation matrix into a long form
    dataframe. Currently specific to the msdl atlas

    Parameters
    ----------
    Correlation matrix: pd.DataFrame
        Correlation matrix that is labeled

    msdl_overview_df: pd.DataFrame
        Dataframe with labels and networks 

    Returns
    -------
    df: pd.DataFrame
        long form df with correlations, named 
        regions and networks

    '''

    df_corr = correlation_matrix.where(np.tril(correlation_matrix).astype(bool))
    df = df_corr.stack().reset_index().rename(columns={0: 'correlation'})
    df['corr_names'] = df['level_1'] + ' - ' + df['level_0']
    df = df.drop(df[df['level_0'] == df['level_1']].index).drop(columns=['level_0', 'level_1'])
    df = df[df.columns[::-1]]
    network_names = get_network_names(msdl_overview_df, df)
    df['network_names'] = network_names
    return df

def get_mean_correlation_matrix(group: np.array, labels: list):
    
    '''
    Function to get mean correlation matrix from group time series

    Parameters
    ----------
    group: np.array
        array of time series
    
    labels: list
        list of labels

    Returns
    -------
    dict: dictionary object
        dict of two dataframes of mean AN and HC 
        correlation matrix as DataFrames 
    '''
    
    full_correlation_matrix = Cyclic_analysis(to_vectorize=False).fit(group)
    an_mean_correlations = pd.DataFrame(full_correlation_matrix[0:65].mean(axis=0))
    an_mean_correlations.columns = labels
    an_mean_correlations.index = labels
    hc_mean_correlations = pd.DataFrame(full_correlation_matrix[65:].mean(axis=0))
    hc_mean_correlations.columns = labels
    hc_mean_correlations.index = labels

    return {
        'an_mean_correlations': an_mean_correlations,
        'hc_mean_correlations': hc_mean_correlations
    }
