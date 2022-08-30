import pandas as pd
import numpy as np
from os import listdir
from scipy import io

def load_and_format_data(relative_path, cutoff=0):
    path = relative_path  # change to correct path
    fnames = listdir(path)

    # define order and trial parameters
    T = 100
    n_subs = len(fnames)
    order = np.tile(range(1,5), (25,1)).flatten(order = 'F')
    mini_blocks = np.arange(1, T + 1)

    # inititate data frame
    data = pd.DataFrame(columns = ['gain', 
                                   'start_points', 
                                   'log_rt_1', 
                                   'log_rt_sum', 
                                   'subject', 
                                   'block_number', 
                                   'phase', 
                                   'order',
                                   'block_index',
                                   'end_points'])

    states = []
    responses = []
    
    subject = 0
    valid = np.ones(len(fnames)).astype(bool)
    for i,f in enumerate(fnames):
        parts = f.split('_')
        tmp = io.loadmat(path+f)
        points = tmp['data']['Points'][0, 0]

        # get response times
        rts = np.nan_to_num(tmp['data']['Responses'][0,0]['RT'][0,0])

        # get number of trials in each mini-block
        notrials = tmp['data']['Conditions'][0,0]['notrials'][0,0].flatten()

        # get points at the last trial of the miniblock
        end_points = points[range(100), (np.nan_to_num(notrials)-1).astype(int)]
        
        if end_points[-1] > cutoff:
            subject += 1
            states.append(tmp['data']['States'][0,0] - 1)
            responses.append(tmp['data']['Responses'][0,0]['Keys'][0,0] - 1)

            start_points = 1000
            df = pd.DataFrame()

            df['gain']= np.diff(np.hstack([start_points, end_points]))

            df['start_points'] = np.hstack([start_points, end_points])[:-1]
            df['end_points'] = end_points

            # define log_rt as the natural logarithm of the sum of response times over all trials
            df['log_rt_1']= np.log(rts[:, 0])
            df['log_rt_sum'] = np.log(np.nansum(rts, -1))

            df['subject'] = subject
            df['ID'] = parts[1]
            df['block_number'] = mini_blocks

            if notrials[0] == 3:
                df['phase'] = np.hstack([order[50:], order[:50]])
                df['order'] = 2
                df['block_index'] = np.hstack([mini_blocks[50:], mini_blocks[:50]])

            else:
                df['phase'] = order
                df['order'] = 1
                df['block_index'] = mini_blocks

            data = pd.concat([data, df], ignore_index=True, sort=False)
        else:
            valid[i] = False
    
    print('fraction of excluded participants ', 1. - subject/len(fnames))
    return data, states, responses, valid