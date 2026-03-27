import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# helper functions

def parse_date(s):
    '''
    Parses an ISO date string into a Python datetime object
    or returns none
    '''
    if not s or pd.isna(s):
        return None
    try:
        return datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except:
        return None


def normalize(matrix):
    '''
    makes all values in a matrix fit the range [0, 1] so 
    all 7 features are on the same scale before combining
    '''
    mn, mx = matrix.min(), matrix.max()
    if mx - mn == 0:
        return np.zeros_like(matrix)
    return (matrix - mn) / (mx - mn)

# feature 4: same category 
def compute_feature4(market_df):
    '''if both markets have the same cateogry --> 1
    else --> 0'''
    n = len(market_df)
    matrix = np.zeros((n, n))
    categories = market_df['category'].tolist()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pd.notna(categories[i]) and pd.notna(categories[j]) and categories[i] == categories[j]:
                matrix[i][j] = 1
    return matrix
