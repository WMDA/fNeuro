import pandas as pd

def order_networks(msdl_overview_df: pd.DataFrame, markers_df: pd.DataFrame, col_name: str) -> dict:
    '''
    Function to organise connections into 
    within network connections and between networks.
    Currently only supports MSDL atlas

    Parameters
    ----------
    msdl_overview_df: pd.DataFrame
        DataFrame with 
    
    markers_df: pd.DataFrame 
        DataFrame that must have a column
        with network connections organised as
        network - network
    
    col_name: str
        str of column name in the markers df
        that has the network connection  

    Returns
    ------
    network_values: dict
        dictionary of DataFrames organised by
        network value

    '''

    network_values = dict(zip([network for network in msdl_overview_df['networks'].unique()], 
                          [{} for network in msdl_overview_df['networks'].unique()]))
    for network in network_values.keys(): 
        network_df = markers_df[markers_df[col_name].str.startswith(network)]
        within_network_df = network_df[network_df[col_name] == f'{network} - {network}'] 
        between_network_df = network_df.drop(within_network_df.index).reset_index(drop=True) 
        network_values[network]['within'] = within_network_df
        network_values[network]['between'] = between_network_df

    return network_values

def seggregate_networks(network_values: dict):
    '''
    Function to concat the output of order networks
    into two dataframes

    Parameters
    ----------
    network_vals: dict
        output from seggregate_networks function
    
    Returns
    -------
    dict: Dictionary object
        dict object of within network and
        between network concat dataframes

    '''
    within_df = []
    between_df = []
    for key in network_values.keys():
        if len(network_values[key]['within']) > 0:
            within_df.append(network_values[key]['within'])
        if len(network_values[key]['between']) > 0:
            between_df.append(network_values[key]['between'])
    return {
        'within_df': pd.concat(within_df).sort_values(by='svr_values',key=abs, ascending=False).reset_index(drop=True),
        'between_df': pd.concat(between_df).sort_values(by='svr_values',key=abs, ascending=False).reset_index(drop=True)
    }