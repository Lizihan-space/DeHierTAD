from collections import defaultdict
import cooler
import os
import pandas as pd
import numpy as np
from scipy.sparse import diags,triu, issparse
from cvxopt import matrix, solvers
from scipy.signal import find_peaks
from typing import Union,  Dict, Tuple
import warnings
from functools import partial
from multiprocessing import Pool
import argparse

#============================================================================#
class CleanHelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        return ', '.join(action.option_strings)
    
parser = argparse.ArgumentParser(description='Identify the multi-level TAD organization from Hi-C contact matrices.', 
                                 formatter_class=CleanHelpFormatter)
parser.add_argument("-c", "--COOL_PATH", type=str, required=True,
                    help='Path to the cool file.')
parser.add_argument("-chr", "--CHROMOSOME", type=str, required=True, 
                    help='Chromosome number.')
parser.add_argument("-r", "--resolution", type=int, required=True,
                    help='Resolution of the cool file.')
parser.add_argument('-A', '--Area', type=int, default=1000000,
                    help='Hi-C data area for first-level iteration.')
parser.add_argument("-o", "--output_folder", type=str, required=True, 
                    help='Output directory.')
parser.add_argument("-w", "--workers", type=int, default=4,
                    help='Number of worker threads.')
parser.add_argument("-p", "--prominence", type=float, required=True,
                    help='Prominence threshold.')
parser.add_argument('-q', "--quality", action='store_true', 
                    help='Output TAD quality scores(Optional).')
parser.add_argument('-m', "--m", type=int, default=10,
                    help='Calculation interval: Area/m.')
parser.add_argument('-g', "--g", type=int, default=2,
                    help='Second-level iteration area: Area/g.')

args = parser.parse_args()
R11 = args.Area
m = args.m
g = args.g
COOL_PATH = args.COOL_PATH
CHROMOSOME = args.CHROMOSOME
output_folder = args.output_folder
workers = args.workers
prominence = args.prominence
resolution = args.resolution
thresholds = 0
R11 = R11
R21 = R11 / m 
R12 = R11 / g 
R22 = R12 / m
parameter_groups = [
    {'R1': int(R11), 'R2': int(R21)},  
    {'R1': int(R12), 'R2': int(R22)}   
]

#============================================================================#
def process_cooler_matrix_internal(
    clr: Union[cooler.Cooler, str],
    chromosome: str, 
    balance: bool = True,
    nan_replace: float = 0.0,
    int_columns: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    if isinstance(clr, str):
        clr = cooler.Cooler(clr)   
    try:
        matrix_df = clr.matrix(
            sparse=False,
            balance=balance,
            as_pixels=True,
            join=False
        ).fetch(chromosome)
        processed_df = data_clean(
            matrix_df,
            int_columns=int_columns,
            nan_replace=nan_replace
        )
        processed_df['chromosome'] = chromosome        
        if verbose:
            print(f"[SUCCESS] Chromosome {chromosome} processed | Shape: {processed_df.shape}")            
        return processed_df
    except Exception as e:
        if verbose:
            print(f"[ERROR] Chromosome {chromosome}: {str(e)}")
        return pd.DataFrame(columns=matrix_df.columns.tolist() + ['chromosome'])

def data_clean(
    df: pd.DataFrame,
    int_columns: int,
    nan_replace: float
) -> pd.DataFrame:
    processed_df = df.fillna(nan_replace)
    int_cols = processed_df.columns[:int_columns]
    float_cols = processed_df.columns[int_columns:]
    processed_df[int_cols] = processed_df[int_cols].round().astype(int)
    processed_df[float_cols] = processed_df[float_cols].apply(pd.to_numeric, downcast='float')   
    return processed_df

def calculate_balanced_sum(
    matrix_df: pd.DataFrame, 
    bin_span: int, 
    bin1_min: int,
    bin1_max: int
) -> pd.DataFrame:
    lookup_dict = defaultdict(dict)
    for row in matrix_df.itertuples(index=False):
        lookup_dict[row.bin1_id][row.bin2_id] = row.balanced    
    results = []
    for current_bin in range(bin1_min, bin1_max + 1):
        max_bin = min(current_bin + bin_span + 1, bin1_max + 1)
        for target_bin in range(current_bin, max_bin):
            balanced = lookup_dict.get(current_bin, {}).get(target_bin, 0.0)
            results.append((current_bin, target_bin, balanced))    
    return pd.DataFrame(results, columns=['current_bin1_id', 'bin2_id', 'balanced_value'])

def process_hic_dataframe(
    matrix_df: pd.DataFrame,
    R1: int,
    R2: int,
    resolution: int = 10000
) -> Dict[int, pd.DataFrame]:
    results_dict = {}
    bin1_min = matrix_df['bin1_id'].min()
    bin1_max = matrix_df['bin1_id'].max()
    bin_span = R1 // resolution
    result_df = calculate_balanced_sum(
        matrix_df=matrix_df,
        bin_span=bin_span,
        bin1_min=bin1_min,
        bin1_max=bin1_max
    )
    results_dict[R1] = result_df    
    return results_dict

#============================================================================#

class TADStorage:
    def __init__(self):
        self.data = defaultdict(dict)   
    def add(self, R1: int, R2: int, boundary_type: str, df: pd.DataFrame):
        if boundary_type not in ['upper', 'lower']:
            raise ValueError("The type must be upper or lower.")
        self.data[(R1, R2)][boundary_type] = df    
    def get(self, R1: int, R2: int, boundary_type: str) -> pd.DataFrame:
        return self.data.get((R1, R2), {}).get(boundary_type, pd.DataFrame())

def data_dimension_reduction(df: pd.DataFrame, resolution, R2: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    upper = df.groupby('current_bin1_id', as_index=False).agg(
        val_sum=('balanced_value', 'sum'),
    )
    upper['val_sum_score'] = np.log2(upper['val_sum'] / upper['val_sum'].mean() + 1)
    upper = upper[['current_bin1_id', 'val_sum_score']]
    lower = df.groupby('bin2_id', as_index=False).agg(
        val_sum=('balanced_value', 'sum'),
    )
    lower['val_sum_score'] = np.log2(lower['val_sum'] / lower['val_sum'].mean() + 1)
    lower = lower[['bin2_id', 'val_sum_score']]
    step_bins = R2 // resolution
    upper['up_score'], upper['down_score'] = calculate_up_down_scores(upper, step_bins)
    lower['up_score'], lower['down_score'] = calculate_up_down_scores(lower, step_bins)    
    return upper, lower

def calculate_up_down_scores(df: pd.DataFrame, step: int):
    num_bins = len(df)
    up_scores = np.zeros(num_bins)
    down_scores = np.zeros(num_bins)   
    for i in range(num_bins):
        up_start = max(0, i - step)
        up_end = i
        up_window = df['val_sum_score'].iloc[up_start:up_end]
        up_scores[i] = up_window.mean() if not up_window.empty else 0
        down_start = i + 1
        down_end = min(num_bins, i + step + 1)
        down_window = df['val_sum_score'].iloc[down_start:down_end]
        down_scores[i] = down_window.mean() if not down_window.empty else 0    
    return up_scores, down_scores

def process_TAD_scores(df: pd.DataFrame, boundary_type: str) -> pd.DataFrame:
    id_column = 'current_bin1_id' if boundary_type == 'upper' else 'bin2_id'
    if 'up_score' not in df or 'down_score' not in df:
        raise ValueError("DataFrame must include up_score and down_score.")
    df['TAD_score'] = (df['up_score'] - df['down_score']) if boundary_type == 'upper' else (df['down_score'] - df['up_score'])    
    return df[[id_column, 'up_score', 'down_score', 'TAD_score']]

#============================================================================#

class Smoothing_spline:
    def __init__(self, x, y, w, lamda):
        self.x = np.array(x)
        self.x = np.sort(self.x)
        self.y = np.array(y)
        self.h = self.x[1:] - self.x[:-1]
        self.w = w
        self.lamda = lamda
        self.dim = len(x)

    def gen_coef_matrix(self):
        yT = np.zeros(2 * self.dim - 2)
        yT[:self.dim] = self.w * self.y
        
        h_inverse = 1 / self.h
        QT = diags(diagonals=[h_inverse[:-1], -(h_inverse[:-1] + h_inverse[1:]), h_inverse[1:]],  
                   offsets=[0, 1, 2], shape=(self.dim - 2, self.dim)).toarray()
        R = 1 / 6 * diags(diagonals=[self.h[1:], 2 * (self.h[:-1] + self.h[1:]), self.h[1:]], 
                           offsets=[-1, 0, 1], shape=(self.dim - 2, self.dim - 2)).toarray()
        AT = np.hstack([QT, -R])        
        B = np.zeros((2 * self.dim - 2, 2 * self.dim - 2))
        B[:self.dim, :self.dim] = np.eye(self.dim) * self.w
        B[self.dim:, self.dim:] = self.lamda * R
        return yT, AT, B

    def solve_g_gamma(self):
        yT, AT, B = self.gen_coef_matrix()
        yT, AT, B = matrix(yT, tc="d"), matrix(AT, tc="d"), matrix(B, tc="d")
        b = matrix(np.zeros(self.dim - 2), tc="d")
        g_gamma = solvers.qp(P=B, q=-yT, A=AT, b=b)["x"]
        g_gamma = np.array(g_gamma).reshape(2 * self.dim - 2,)    
        g = g_gamma[:self.dim]
        gamma = np.zeros_like(g)
        gamma[1:-1] = g_gamma[self.dim:]
        self.g = g
        self.gamma = gamma
        return g, gamma

    def fit(self):
        g, gamma = self.solve_g_gamma()        
        ai = g[:-1]
        bi = (g[1:] - g[:-1]) / self.h - self.h / 6 * (2 * self.gamma[:-1] + self.gamma[1:])
        ci = gamma[:-1] / 2
        di = (gamma[1:] - gamma[:-1]) / (6 * self.h)
        coef = np.array([ai, bi, ci, di])
        self.coef = coef

    def eval(self, xn):
        yn = np.zeros(len(xn))
        for i in range(len(xn)):
            if xn[i] <= self.x[0]:
                a, b = self.coef[:2, 0]
                yn[i] = a + b * (xn[i] - self.x[0])
            elif xn[i] >= self.x[-1]:
                a = self.g[-1]
                b = (self.g[-1] - self.g[-2]) / self.h[-1] + self.h[-1] / 6 * (2 * self.gamma[-1] + self.gamma[-2])
                yn[i] = a + b * (xn[i] - self.x[-1])
            else:
                xn_idx = np.where(self.x <= xn[i])[0][-1]
                a, b, c, d = self.coef[:, xn_idx]
                yn[i] = a + b * (xn[i] - self.x[xn_idx]) + c * (xn[i] - self.x[xn_idx]) ** 2 + d * (xn[i] - self.x[xn_idx]) ** 3 
        return yn

def Smoothing_spline_curve(
    storage: TADStorage, 
    block_size: int, 
    lamda: int
) -> dict:
    smoothed_results = defaultdict(dict)
    for (R1, R2), boundaries in storage.data.items():
        group_results = {}
        for boundary in ['upper', 'lower']:
            if boundary not in boundaries:
                continue
                
            df = boundaries[boundary]
            if df.empty:
                continue
            col = 'current_bin1_id' if boundary == 'upper' else 'bin2_id'
            num_blocks = len(df) // block_size
            smoothed_blocks = []
            for i in range(num_blocks):
                start_idx = i * block_size
                end_idx = (i + 1) * block_size
                block_df = df.iloc[start_idx:end_idx]
                smoother = Smoothing_spline(
                    x=block_df[col].values,
                    y=block_df['TAD_score'].values,
                    w=1,
                    lamda=lamda
                )
                smoother.fit()
                block_smoothed_df = block_df.copy()
                block_smoothed_df['TAD_score_smoothed'] = smoother.eval(block_df[col].values)
                smoothed_blocks.append(block_smoothed_df)
            if len(df) % block_size != 0:
                start_idx = num_blocks * block_size
                block_df = df.iloc[start_idx:]                
                smoother = Smoothing_spline(
                    x=block_df[col].values,
                    y=block_df['TAD_score'].values,
                    w=1,
                    lamda=lamda
                )
                smoother.fit()                
                block_smoothed_df = block_df.copy()
                block_smoothed_df['TAD_score_smoothed'] = smoother.eval(block_df[col].values)
                smoothed_blocks.append(block_smoothed_df)
            group_results[boundary] = pd.concat(smoothed_blocks, ignore_index=True)
        smoothed_results[(R1, R2)] = group_results    
    return smoothed_results

#============================================================================#

def detect_valley_points_1(
    smoothed_data: dict,  
    chromosome: str,      
    threshold: float,
    prominence: float
) -> dict:
    valley_dict = defaultdict(dict)    
    for (R1, R2), boundaries in smoothed_data.items():
        if 'upper' in boundaries:
            upper_df = boundaries['upper']
            y_upper = -upper_df['TAD_score_smoothed'].values
            peaks, properties = find_peaks(y_upper, height=threshold, prominence=prominence)          
            upper_result = pd.DataFrame({
                'chromosome': chromosome,
                'position': upper_df['current_bin1_id'].iloc[peaks].values,
                'score': -properties['peak_heights']
            })
        else:
            upper_result = pd.DataFrame()
        if 'lower' in boundaries:
            lower_df = boundaries['lower']
            y_lower = -lower_df['TAD_score_smoothed'].values
            peaks, properties = find_peaks(y_lower, height=threshold, prominence=prominence)            
            lower_result = pd.DataFrame({
                'chromosome': chromosome,
                'position': lower_df['bin2_id'].iloc[peaks].values,
                'score': -properties['peak_heights']
            })
        else:
            lower_result = pd.DataFrame()        
        valley_dict[(R1, R2)] = {
            'upper': upper_result,
            'lower': lower_result
        }
    return dict(valley_dict)

def detect_valley_points_2(
    smoothed_data: dict,  
    chromosome: str,      
    threshold: float,
) -> dict:
    valley_dict = defaultdict(dict)    
    for (R1, R2), boundaries in smoothed_data.items():
        if 'upper' in boundaries:
            upper_df = boundaries['upper']
            y_upper = -upper_df['TAD_score_smoothed'].values
            peaks, properties = find_peaks(y_upper, height=threshold)            
            upper_result = pd.DataFrame({
                'chromosome': chromosome,
                'position': upper_df['current_bin1_id'].iloc[peaks].values,
                'score': -properties['peak_heights']
            })
        else:
            upper_result = pd.DataFrame()
        if 'lower' in boundaries:
            lower_df = boundaries['lower']
            y_lower = -lower_df['TAD_score_smoothed'].values
            peaks, properties = find_peaks(y_lower, height=threshold)            
            lower_result = pd.DataFrame({
                'chromosome': chromosome,
                'position': lower_df['bin2_id'].iloc[peaks].values,
                'score': -properties['peak_heights']
            })
        else:
            lower_result = pd.DataFrame()        
        valley_dict[(R1, R2)] = {
            'upper': upper_result,
            'lower': lower_result
        }
    return dict(valley_dict)

#============================================================================#

def merge_and_labels(valley_results: dict) -> pd.DataFrame:
    all_dfs = []    
    for (R1, R2), boundaries in valley_results.items():
        if 'upper' in boundaries and not boundaries['upper'].empty:
            upper_df = boundaries['upper'].copy()
            upper_df['label'] = 1
            all_dfs.append(upper_df)
        if 'lower' in boundaries and not boundaries['lower'].empty:
            lower_df = boundaries['lower'].copy()
            lower_df['label'] = 2
            all_dfs.append(lower_df)
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df[['chromosome', 'position', 'score', 'label']]
    else:
        merged_df = pd.DataFrame()   
    return merged_df
def create_label_3(merge_boundaries: pd.DataFrame, resolution:int) -> pd.DataFrame:
    df = merge_boundaries.sort_values(['chromosome', 'position']).reset_index(drop=True)   
    changed = True
    while changed:
        changed = False
        to_remove = set()
        new_rows = []
        i = 0
        while i < len(df) - 1:
            current = df.iloc[i]
            next_row = df.iloc[i + 1]
            if current['chromosome'] != next_row['chromosome']:
                i += 1
                continue
            pos_diff = (next_row['position'] - current['position']) * resolution
            if pos_diff <= 30000:
                if current['label'] != next_row['label']:
                    new_pos = int((current['position'] + next_row['position']) // 2)
                    new_score = (current['score'] + next_row['score']) / 2
                    new_rows.append({
                        'chromosome': current['chromosome'],
                        'position': new_pos,
                        'score': round(new_score, 4),
                        'label': 3
                    })
                    to_remove.update({i, i + 1})
                    changed = True
                    i += 2
                else:
                    if current['score'] > next_row['score'] or \
                       (current['score'] == next_row['score'] and current['position'] > next_row['position']):
                        to_remove.add(i)
                    else:
                        to_remove.add(i + 1)
                    changed = True
                    i += 1
            else:
                i += 1
        df = df.drop(index=list(to_remove)).reset_index(drop=True)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df.sort_values(['chromosome', 'position']).reset_index(drop=True)
    return df

def process_label_1(candidate_boundaries: pd.DataFrame) -> pd.DataFrame:
    df = candidate_boundaries.reset_index(drop=True)
    pair_df = pd.DataFrame(columns=[
        'start_boundary', 'start_label', 'start_boundary_score',
        'end_boundary', 'end_label', 'end_boundary_score'
    ])
    def find_last_2_index(start_index: int) -> int:
        last_2_index = start_index
        while last_2_index + 1 < len(df):
            next_row = df.iloc[last_2_index + 1]
            if next_row['label'] == 2:
                last_2_index += 1
            else:
                break
        return last_2_index
    i = 0
    while i < len(df):
        current_row = df.iloc[i]
        if current_row['label'] == 1:
            next_index = i + 1
            found = False
            while next_index < len(df):
                next_row = df.iloc[next_index]
                if next_row['label'] in [2, 3]:
                    found = True
                    break
                next_index += 1
            if found:
                if next_row['label'] == 3:
                    end_index = next_index
                    end_row = next_row
                    pair_df = pd.concat(
                        [pair_df, pd.DataFrame([{
                            'start_boundary': current_row['position'],
                            'start_label': current_row['label'],
                            'start_boundary_score': current_row['score'],
                            'end_boundary': end_row['position'],
                            'end_label': end_row['label'],
                            'end_boundary_score': end_row['score']
                        }])],
                        ignore_index=True
                    )
                    i = end_index
                else:
                    last_2_index = find_last_2_index(next_index)
                    end_index = last_2_index
                    end_label = 2
                    check_3_index = end_index + 1
                    if check_3_index < len(df) and df.iloc[check_3_index]['label'] == 3:
                        end_index = check_3_index
                        end_label = 3
                    end_row = df.iloc[end_index]
                    pair_df = pd.concat(
                        [pair_df, pd.DataFrame([{
                            'start_boundary': current_row['position'],
                            'start_label': current_row['label'],
                            'start_boundary_score': current_row['score'],
                            'end_boundary': end_row['position'],
                            'end_label': end_label,
                            'end_boundary_score': end_row['score']
                        }])],
                        ignore_index=True
                    )
                    i = end_index
            else:
                i += 1
        else:
            i += 1
    return pair_df

def process_label_3(candidate_boundaries: pd.DataFrame) -> pd.DataFrame:
    df = candidate_boundaries.reset_index(drop=True)
    pair_df = pd.DataFrame(columns=[
        'start_boundary', 'start_label', 'start_boundary_score',
        'end_boundary', 'end_label', 'end_boundary_score'
    ])
    def find_last_2_index(start_index: int) -> int:
        last_2_index = start_index
        while last_2_index + 1 < len(df):
            next_row = df.iloc[last_2_index + 1]
            if next_row['label'] == 2:
                last_2_index += 1
            else:
                break
        return last_2_index
    i = 0
    while i < len(df):
        current_row = df.iloc[i]
        if current_row['label'] == 3:
            next_index = i + 1
            found = False
            while next_index < len(df):
                next_row = df.iloc[next_index]
                if next_row['label'] in [2, 3]:
                    found = True
                    break
                next_index += 1
            if found:
                if next_row['label'] == 3:
                    end_index = next_index
                    end_row = next_row
                    pair_df = pd.concat(
                        [pair_df, pd.DataFrame([{
                            'start_boundary': current_row['position'],
                            'start_label': current_row['label'],
                            'start_boundary_score': current_row['score'],
                            'end_boundary': end_row['position'],
                            'end_label': end_row['label'],
                            'end_boundary_score': end_row['score']
                        }])],
                        ignore_index=True
                    )
                    i = end_index
                else:
                    last_2_index = find_last_2_index(next_index)
                    end_index = last_2_index
                    end_label = 2
                    check_3_index = end_index + 1
                    if check_3_index < len(df) and df.iloc[check_3_index]['label'] == 3:
                        end_index = check_3_index
                        end_label = 3
                    end_row = df.iloc[end_index]
                    pair_df = pd.concat(
                        [pair_df, pd.DataFrame([{
                            'start_boundary': current_row['position'],
                            'start_label': current_row['label'],
                            'start_boundary_score': current_row['score'],
                            'end_boundary': end_row['position'],
                            'end_label': end_label,
                            'end_boundary_score': end_row['score']
                        }])],
                        ignore_index=True
                    )
                    i = end_index
            else:
                i += 1
        else:
            i += 1
    return pair_df

def filter_overlapping_intervals(df):
    if df.empty:
        return df
    df_sorted = df.copy().sort_values(by='start_boundary').reset_index(drop=True)
    df_sorted['length'] = df_sorted['end_boundary'] - df_sorted['start_boundary']
    result_df = pd.DataFrame()
    current_max_end = float('-inf')
    last_retained_index = -1
    for idx, row in df_sorted.iterrows():
        start = row['start_boundary']
        end = row['end_boundary']
        if start < current_max_end:
            prev_row = df_sorted.iloc[last_retained_index]
            if row['length'] > prev_row['length']:
                if not result_df.empty:
                    result_df = result_df.iloc[:-1]
                result_df = pd.concat([result_df, row.to_frame().T])
                current_max_end = max(current_max_end, end)
                last_retained_index = idx
        else:
            result_df = pd.concat([result_df, row.to_frame().T])
            current_max_end = end
            last_retained_index = idx
    return result_df.reset_index(drop=True).drop(columns=['length'])

#============================================================================#

def permutation_and_combination(full_slice, resolution:int):   

    full_slice = full_slice.sort_values('position').reset_index(drop=True)
    full_slice['score'] = full_slice['score'].astype(float)
    pairs = [{
        'start_boundary': full_slice.iloc[0]['position'],
        'start_label': full_slice.iloc[0]['label'],
        'start_boundary_score': full_slice.iloc[0]['score'],
        'end_boundary': full_slice.iloc[-1]['position'],
        'end_label': full_slice.iloc[-1]['label'],
        'end_boundary_score': full_slice.iloc[-1]['score']
    }]
    processed_intervals = [(full_slice.iloc[0]['position'], full_slice.iloc[-1]['position'])]
    position_to_row = full_slice.set_index('position').to_dict('index')
    internal_slice = full_slice.iloc[1:-1].sort_values(
        ['score', 'position'], 
        ascending=[True, True]
    ).reset_index(drop=True)   
    for idx, row in internal_slice.iterrows():
        current_pos = row['position']
        current_label = row['label']
        found_pairs = []
        possible_parents = [iv for iv in processed_intervals 
                           if iv[0] <= current_pos <= iv[1]]
        if not possible_parents:
            continue
        current_parent = min(possible_parents, key=lambda x: x[1] - x[0])
        parent_start, parent_end = current_parent        
        if current_label == 1:
            candidates = [
                pos for pos in position_to_row 
                if (current_pos < pos <= parent_end and 
                    position_to_row[pos]['label'] in {2, 3})
            ]
            if candidates:
                best_pos = min(candidates, key=lambda p: (
                    position_to_row[p]['score'], 
                    p
                ))
                found_pairs.append((current_pos, best_pos))        
        elif current_label == 2:
            candidates = [
                pos for pos in position_to_row 
                if (parent_start <= pos < current_pos and 
                    position_to_row[pos]['label'] in {1, 3})
            ]
            if candidates:
                best_pos = min(candidates, key=lambda p: (
                    position_to_row[p]['score'], 
                    p
                ))
                found_pairs.append((best_pos, current_pos))       
        elif current_label == 3:
            up_candidates = [
                pos for pos in position_to_row 
                if (parent_start <= pos < current_pos and 
                    position_to_row[pos]['label'] in {1, 3})
            ]
            if up_candidates:
                up_best = min(up_candidates, key=lambda p: (
                    position_to_row[p]['score'], 
                    p
                ))
                found_pairs.append((up_best, current_pos))
            down_candidates = [
                pos for pos in position_to_row 
                if (current_pos < pos <= parent_end and 
                    position_to_row[pos]['label'] in {2, 3})
            ]
            if down_candidates:
                down_best = min(down_candidates, key=lambda p: (
                    position_to_row[p]['score'], 
                    p
                ))
                found_pairs.append((current_pos, down_best))
        valid_pairs = []
        for pair in found_pairs:
            start, end = pair
            length = (end - start) * resolution
            if length <= 30000:
                continue
            if not (parent_start <= start <= parent_end and 
                    parent_start <= end <= parent_end):
                continue                
            conflict = False
            for existing in processed_intervals:
                overlap = (start < existing[1]) and (end > existing[0])
                is_nested = (
                    (start >= existing[0] and end <= existing[1]) or
                    (existing[0] >= start and existing[1] <= end)
                )
                if overlap and not is_nested:
                    conflict = True
                    break                    
            if not conflict:
                valid_pairs.append(pair)
                processed_intervals.append(pair)
        for pair in valid_pairs:
            start_pos, end_pos = pair
            start_row = position_to_row[start_pos]
            end_row = position_to_row[end_pos]
            pairs.append({
                'start_boundary': start_pos,
                'start_label': start_row['label'],
                'start_boundary_score': start_row['score'],
                'end_boundary': end_pos,
                'end_label': end_row['label'],
                'end_boundary_score': end_row['score']
            })   
    return pd.DataFrame(pairs)

def process_slices(candidate_boundaries, global_first_TAD, resolution: int):
    merged_results = pd.DataFrame(columns=[
        'start_boundary', 'start_label', 'start_boundary_score',
        'end_boundary', 'end_label', 'end_boundary_score'
    ])
    all_processed_slices = []
    candidate_boundaries = candidate_boundaries.sort_values('position')
    for _, tad_row in global_first_TAD.iterrows():
        start = tad_row['start_boundary']
        end = tad_row['end_boundary']       
        mask = (candidate_boundaries['position'] > start) & (candidate_boundaries['position'] < end)
        current_slice = candidate_boundaries[mask].copy()        
        if current_slice.empty:
            continue
        start_header = pd.DataFrame({
            'chromosome': [current_slice['chromosome'].iloc[0]],
            'position': [start],
            'score': [tad_row['start_boundary_score']],
            'label': [tad_row['start_label']]
        })
        end_footer = pd.DataFrame({
            'chromosome': [current_slice['chromosome'].iloc[0]],
            'position': [end],
            'score': [tad_row['end_boundary_score']],
            'label': [tad_row['end_label']]
        })
        full_slice = pd.concat([start_header, current_slice, end_footer], ignore_index=True)
        slice_result = permutation_and_combination(full_slice, resolution)
        all_processed_slices.append(slice_result)
    if all_processed_slices:
        merged_results = pd.concat(all_processed_slices, ignore_index=True)
    sorted_TAD = global_first_TAD.sort_values('start_boundary').reset_index(drop=True)    
    gap_slices = []
    for i in range(1, len(sorted_TAD)):
        prev_end = sorted_TAD.iloc[i-1]['end_boundary']
        curr_start = sorted_TAD.iloc[i]['start_boundary']
        gap_bp = (curr_start - prev_end) * resolution
        if gap_bp > 100000:
            mask = (candidate_boundaries['position'] > prev_end) & \
                   (candidate_boundaries['position'] < curr_start)
            gap_slice = candidate_boundaries[mask].copy()            
            if not gap_slice.empty:
                gap_result = permutation_and_combination(gap_slice, resolution)
                gap_slices.append(gap_result)
    if gap_slices:
        merged_results = pd.concat([merged_results] + gap_slices, ignore_index=True)    
    return merged_results.sort_values('start_boundary').reset_index(drop=True)

#============================================================================#

def bin_to_base(
    hierarchical_structure: pd.DataFrame,
    COOL_PATH: str,
    resolution: int,
    chrom: str
) -> pd.DataFrame:
    clr = cooler.Cooler(COOL_PATH)
    region = clr.extent(chrom)
    start_offset = region[0]
    chrom_length = clr.chromsizes[chrom]
    df = hierarchical_structure.copy()
    df["start"] = (df["start_boundary"] - start_offset + 1) * resolution
    raw_end = (df["end_boundary"] - start_offset + 1) * resolution
    df["end"] = np.minimum(raw_end, chrom_length)
    result_df = pd.DataFrame({
        "chrom": chrom,
        "start": df["start"],
        "end": df["end"],
        "start_boundary_score": df["start_boundary_score"],
        "end_boundary_score": df["end_boundary_score"]
    })  
    result_df = result_df.sort_values('start').reset_index(drop=True)    
    return result_df

def organize_TAD(base_result):
    df = base_result.copy()
    df = df.sort_values(by='start').reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()    
    groups = []
    current_group = 1
    current_group_end = df.iloc[0]['end']
    groups.append(current_group)    
    for i in range(1, len(df)):
        if df.iloc[i]['start'] < current_group_end:
            groups.append(current_group)
            current_group_end = max(current_group_end, df.iloc[i]['end'])
        else:
            current_group += 1
            groups.append(current_group)
            current_group_end = df.iloc[i]['end']    
    df['group'] = groups
    def adjust_group(group):
        starts = sorted(group['start'].unique())
        start_map = {}
        for s in group['start']:
            candidates = [x for x in starts if x < s and (s - x) <= 30000]
            start_map[s] = min(candidates) if candidates else s
        ends = sorted(group['end'].unique(), reverse=True)
        end_map = {}
        for e in group['end']:
            candidates = [x for x in ends if x > e and (x - e) <= 30000]
            end_map[e] = max(candidates) if candidates else e        
        group['start'] = group['start'].map(start_map)
        group['end'] = group['end'].map(end_map)
        return group    
    df = df.groupby('group', group_keys=False).apply(adjust_group)
    df = df.drop_duplicates(subset=['chrom', 'start', 'end'])
    cols_to_drop = ['start_boundary_score', 'end_boundary_score']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    df['length'] = df['end'] - df['start']
    df = df[df['length'] <= 2000000].drop(columns=['length'])
    def calculate_level(group):
        sorted_group = group.sort_values(by=['start', 'end'], ascending=[True, False])
        intervals = sorted_group[['start', 'end']].values
        level = [1] * len(intervals)        
        for i in range(len(intervals)):
            for j in range(i):
                if intervals[j][0] <= intervals[i][0] and intervals[j][1] >= intervals[i][1]:
                    level[i] += 1        
        sorted_group['level'] = level
        return sorted_group    
    df = df.groupby('group', group_keys=False).apply(calculate_level)
    df['group_rank'] = df.groupby('group').cumcount() + 1
    df['id'] = 'TAD_' + df['group'].astype(str) + '_' + df['level'].astype(str) + '_' + df['group_rank'].astype(str)
    df = df.drop(columns=['group_rank'])    
    return df.reset_index(drop=True)

#============================================================================#

def safe_sparse_mean(matrix_slice):
    if matrix_slice.shape[0] == 0 or matrix_slice.shape[1] == 0:
        return 0.0
    if issparse(matrix_slice):
        dense_matrix = matrix_slice.toarray()
    else:
        dense_matrix = np.asarray(matrix_slice)
    dense_matrix = np.nan_to_num(dense_matrix, nan=0.0)
    total = dense_matrix.sum()
    num_elements = dense_matrix.size
    return total / num_elements if num_elements > 0 else 0.0

def process_chromosome(args):
    chrom, chrom_df, cool_path, resolution = args
    clr = cooler.Cooler(cool_path)
    results = []   
    try:
        bins = clr.bins().fetch(chrom)
        matrix = clr.matrix(balance=True, sparse=True).fetch(chrom).tocsr()
        max_bin = bins.shape[0] - 1
        chrom_df["start_bin"] = (chrom_df["start"] // resolution - 1).astype(int)
        chrom_df["end_bin"] = (chrom_df["end"] // resolution - 1).astype(int)
        valid_mask = (
            (chrom_df["start_bin"] >= 0) &
            (chrom_df["end_bin"] <= max_bin)
        )
        chrom_df = chrom_df[valid_mask].copy()
        data = chrom_df[["start", "end", "start_bin", "end_bin", "level"]].to_numpy()        
        for row in data:
            start, end, start_bin, end_bin, level = row
            tad_length = end_bin - start_bin + 1
            intra_triu = triu(matrix[start_bin:end_bin+1, start_bin:end_bin+1], k=1)
            intra_mean = safe_sparse_mean(intra_triu)
            down_start = end_bin + 1
            down_end = min(end_bin + tad_length, max_bin)
            down_mean = 0.0
            if down_start <= max_bin:
                down_region = matrix[start_bin:end_bin+1, down_start:down_end+1]
                down_mean = safe_sparse_mean(down_region)
            up_start = max(start_bin - tad_length, 0)
            up_end = start_bin - 1
            up_mean = 0.0
            if up_start <= up_end:
                up_region = matrix[up_start:up_end+1, start_bin:end_bin+1]
                up_mean = safe_sparse_mean(up_region)
            results.append([
                str(chrom), 
                int(start), 
                int(end), 
                int(level),
                float(intra_mean),
                float(up_mean),
                float(down_mean),
                float(intra_mean - up_mean),
                float(intra_mean - down_mean)
            ])
            
    except Exception as e:
        print(f"Error processing {chrom}: {str(e)}")        
    return results

def calculate_tad_interactions(cool_path, tad_df, resolution, workers):
    required_columns = {'chrom', 'start', 'end', 'level'}
    if not required_columns.issubset(tad_df.columns):
        raise ValueError(f"DataFrame must include : {required_columns}.")
    tad_df = tad_df.astype({'chrom': str})
    chrom_groups = [
        (chrom, group, cool_path, resolution)
        for chrom, group in tad_df.groupby('chrom', sort=False)
    ]
    with Pool(workers) as pool:
        results = pool.map(process_chromosome, chrom_groups)
    tad_interactions_df = pd.DataFrame(
        [item for sublist in results for item in sublist],
        columns=[
            'chrom', 'start', 'end', 'level',
            'intra_mean', 'upstream_mean', 'downstream_mean',
            'upstream_mean_diff', 'downstream_mean_diff'
        ]
    )    
    return tad_interactions_df

def calculate_tad_scores(df):
    df['adj_diff'] = (df['upstream_mean_diff'] + df['downstream_mean_diff']) / 2
    return df[['chrom', 'start', 'end', 'level', 'intra_mean', 'adj_diff']].copy() 

def optimized_tad_scores(df):
    df = df.copy()
    condition = (df['intra_mean'] == 0) | (df['adj_diff'] < 0)
    df['quality_score'] = np.where(
        condition,
        0.0,
        (df['intra_mean'] - df['adj_diff']) / df['intra_mean']
    )
    return df

def print_level_stats(final_df):
    stats = final_df.groupby('level')['quality_score'].agg(
        Mean_Score='mean',
        Median_Score='median'
    ).reset_index()
    stats_formatted = stats.round(3)    
    print("\nStatistics of hierarchical TAD quality score:")
    print("="*35)
    print(f"{'Level':<8} {'Average score':<12}  {'Median':<12}")
    print("-"*35)
    for _, row in stats_formatted.iterrows():
        print(f"{row['level']:<8} {row['Mean_Score']:<12} {row['Median_Score']:<12}")
    print("="*35 + "\n")

#============================================================================#

if __name__ == "__main__":
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    results = process_cooler_matrix_internal(
        clr=COOL_PATH,
        chromosome=CHROMOSOME,
        balance=True,
        int_columns=3
    )
    
    os.makedirs(output_folder, exist_ok=True)
    global_first_TAD = None
    functions = [partial(process_hic_dataframe, matrix_df=results)]
    
    for idx, params in enumerate(parameter_groups):
        R1_val = params['R1']
        R2_val = params['R2']
        group_dir = os.path.join(output_folder, f"A_{R1_val}_I_{R2_val}_{CHROMOSOME}_{prominence}")
        os.makedirs(group_dir, exist_ok=True)        
        for func in functions: 
            balance_sum = func(**params)
            current_tad_storage = TADStorage()            
            df = balance_sum[R1_val]
            upper_df, lower_df = data_dimension_reduction(
                df, 
                R2=R2_val,
                resolution=resolution
            )            
            tad_upper = process_TAD_scores(upper_df, 'upper')
            tad_lower = process_TAD_scores(lower_df, 'lower')
            current_tad_storage.add(R1_val, R2_val, 'upper', tad_upper)
            current_tad_storage.add(R1_val, R2_val, 'lower', tad_lower)

            smoothed_data = Smoothing_spline_curve(
                storage=current_tad_storage,
                block_size=1000,
                lamda=5
            )            
            for boundary in ['upper', 'lower']:
                df_smooth = smoothed_data.get((R1_val, R2_val), {}).get(boundary, pd.DataFrame())
                if not df_smooth.empty:
                    col = 'current_bin1_id' if boundary == 'upper' else 'bin2_id'
                    out_type = 'up' if boundary == 'upper' else 'down'
                    out_path = os.path.join(group_dir, f"smoothed_{out_type}.txt")
                    df_smooth[[col, 'TAD_score_smoothed']].to_csv(
                        out_path,
                        sep='\t',
                        index=False,
                        float_format='%.9f'
                    )

            if idx==0:                
                valley_results= detect_valley_points_1(
                    smoothed_data=smoothed_data,
                    chromosome=CHROMOSOME,
                    threshold=thresholds,
                    prominence=prominence
                )
                
                merge_boundaries = merge_and_labels(
                    valley_results = valley_results
                )
                
                candidate_boundaries = create_label_3(
                    merge_boundaries = merge_boundaries,
                    resolution = resolution
                )
                
                TAD_boundaries_bins_1 = process_label_1(
                    candidate_boundaries = candidate_boundaries
                )
                
                TAD_boundaries_bins_3 = process_label_3(
                    candidate_boundaries = candidate_boundaries
                )
                
                TAD_boundaries_bins = pd.concat(
                    [TAD_boundaries_bins_1, TAD_boundaries_bins_3],
                    ignore_index=True
                )
                TAD_boundaries_bins = filter_overlapping_intervals(
                    df = TAD_boundaries_bins
                )
                global_first_TAD = TAD_boundaries_bins.copy()                           
            else:                
                valley_results= detect_valley_points_2(
                    smoothed_data=smoothed_data,
                    chromosome=CHROMOSOME,
                    threshold=thresholds,
                )
                
                merge_boundaries = merge_and_labels(
                    valley_results = valley_results
                )
                
                candidate_boundaries = create_label_3(
                    merge_boundaries = merge_boundaries,
                    resolution = resolution
                )
                
                hierarchical_structure = process_slices(
                    candidate_boundaries = candidate_boundaries,
                    global_first_TAD = global_first_TAD,
                    resolution = resolution
                )
                
                base_result = bin_to_base(
                    hierarchical_structure = hierarchical_structure,
                    COOL_PATH = COOL_PATH,
                    resolution = resolution,
                    chrom = CHROMOSOME
                )
                  
                hierarchical_TAD_result = organize_TAD(base_result)
                               
                tad_interactions_df = calculate_tad_interactions(
                    cool_path=COOL_PATH,
                    tad_df=hierarchical_TAD_result,
                    resolution=resolution,
                    workers = workers
                )
                
                quality_scored_df = calculate_tad_scores(tad_interactions_df)
                final_quality_scored_df = optimized_tad_scores(quality_scored_df)
                
                zero_score_tads = final_quality_scored_df[
                final_quality_scored_df['quality_score'] == 0
                ][['chrom', 'start', 'end', 'level']].drop_duplicates()

                filter_hierarchical_TAD_result = hierarchical_TAD_result.merge(
                    zero_score_tads,
                    on=['chrom', 'start', 'end', 'level'],
                    how='left',
                    indicator=True
                ).query('_merge == "left_only"').drop('_merge', axis=1)
                
                filter_hierarchical_TAD_result = organize_TAD(filter_hierarchical_TAD_result)
                out_path = os.path.join(output_folder, f"hierarchical_TAD_chr{CHROMOSOME}_{prominence}.bed")
                column_mapping = {'group': 'X', 'level': 'L'}
                filter_hierarchical_TAD_result.rename(columns=column_mapping).to_csv(
                    out_path, sep='\t', index=False, float_format='%.4f')
                
                tad_interactions_df = calculate_tad_interactions(
                    cool_path=COOL_PATH,
                    tad_df=filter_hierarchical_TAD_result,
                    resolution=resolution,
                    workers = workers
                )                
                quality_scored_df = calculate_tad_scores(tad_interactions_df)
                final_quality_scored_df = optimized_tad_scores(quality_scored_df)                
                if args.quality:
                    final_quality_scored_df = final_quality_scored_df.drop(columns=['intra_mean', 'adj_diff'])
                    out_path = os.path.join(output_folder, f"hierarchical_TAD_chr{CHROMOSOME}_quality_score_{prominence}.txt")
                    column_mapping = {'level': 'L'}
                    final_quality_scored_df.rename(columns=column_mapping).to_csv(
                        out_path, sep='\t', index=False, float_format='%.7f')
                print(f"\n{'='*40}")
                print_level_stats(final_quality_scored_df)
                print(f"{'='*40}\n")                
                print(f"The number of TADs:Before filration {len(hierarchical_TAD_result)},After filration {len(filter_hierarchical_TAD_result)}.")