# -*- coding: utf-8 -*-
# Sampling tools created by Changchun Wang, IMF
"""
Created on Tue Jul 16 13:25:12 2019

@author: CWang2
"""
import pandas as pd
import numpy as np               
from scipy.stats.kde import gaussian_kde 

def extrapolate(df, cols=None):
    n=len(df)
    if cols==None:
        cols=df.columns
    df['dumbx']=list(range(n))

    for col in cols:
        y=df[df[col].notnull()][col].values
        x=df[df[col].notnull()]['dumbx'].values

        xn=df[pd.isnull(df[col])]['dumbx'].values
        df.loc[df[pd.isnull(df[col])].index,col]=np.poly1d(np.polyfit(x,y,1))(xn)
    df.drop('dumbx',axis=1,inplace=True)
    return df

def sample_mode_adj(samples, mode, alpha=0.6):
    '''
    Shift samples toward new mode,
    the shifted distance is proportional to the distance to the original mode.

    - Alpha is the parameter of the rejection algorithm

    '''
    mins = min(samples)
    maxs = max(samples)
    support = np.linspace(mins+1, maxs-1, 200)
    tmpkde = gaussian_kde(samples) # Temporary Kernel
    kvals = [tmpkde.pdf(x) for x in support] # Get the pdf 
    mv = max(kvals)
    for i,e in enumerate(kvals):
        if e==mv:
            oldmode=support[i] # Measure the mode == highest value
            break    
    '''
    Do a bisect search in a small range to make the mode of KDE exactly 
    at the projection.
    '''
    left=mode-0.5
    right=mode+0.5
    iteration=2

    # Rejection sampling algorithm to recreate a sample with the correct mode
    while iteration<20:    
        tar = (left+right)/2
        delta = (maxs-mins)/20*abs((oldmode-tar)/oldmode)
        maxshift = tar-oldmode
        nsample = []
        mins -= delta
        maxs += delta
        reject_thres = (abs(oldmode-tar)/(oldmode))**alpha         
        for e in samples:
            if e<oldmode:
                if mode>=oldmode or np.random.rand()>reject_thres:
                    nsample.append(e+maxshift*((e-mins)**alpha/((oldmode-mins)**alpha)))
            else:
                if mode<=oldmode or np.random.rand()>reject_thres:
                    nsample.append(e+maxshift*((maxs-e)**alpha/((maxs-oldmode)**alpha)))
        support=np.arange(left, right, 0.005)
        tmpkde=gaussian_kde(nsample)
        kvals=[tmpkde.pdf(x) for x in support]
        mv=max(kvals)
        for i,e in enumerate(kvals):
            if e==mv:
                nmode=support[i]
                break
        if abs(nmode-mode)<0.01:
            break
        
        if nmode<mode:
            left=tar
        else:
            right=tar
        iteration+=1
    
    return(nsample)
    
    
    
if __name__=='__main__':
    df=pd.DataFrame(index=range(4),columns=range(4))
    df.loc[0,1]=3
    df.loc[0,4]=2
    df.loc[1,0]=1
    df.loc[1,1]=1
    df.loc[2,2]=1
    df.loc[3,3]=1
    df.loc[1,3]=5
    df.loc[3,2]=1
    df.loc[4,4]=2
    df.loc[3,0]=2
    df.loc[4,1]=4
    print(df)
    df=extrapolate(df)
    print(df)
