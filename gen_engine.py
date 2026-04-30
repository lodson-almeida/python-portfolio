'''This is generic Engine modified for accommodating any Strategy'''
import pandas as pd
import gc
import os
import sys
import pickle
import random
from tqdm import tqdm
import logging
from tqdm.contrib.concurrent import process_map
import basic.custom_functions as cf
from backtest_engine.generic.gen_h4m1 import H4M1FastTester as Tester
from backtest_engine.generic.gen_excel import create_res as gen_excel_standard 
from backtest_engine.WFA.wfa_excel import create_res as wfa_excel_wfa
from backtest_engine.generic.gen_performance_metrics import AllMetrics
from backtest_engine.generic.gen_analyzer import analyze_allresults
import numpy as np
import traceback
import time
from typing import Optional, Tuple, Dict, Any, List # Added List
from basic.log_wrapper import LogWrapper # Added import

# Apply optimization for pandas operations
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

def extract_config_attributes(Config: Any) -> Dict[str, Any]:
    """Extract required configuration attributes into a dictionary for easier passing."""
    attributes = [
        'Strategy_name', 'trade_res_path', 'Chk_pt_interval', 'start_year', 'end_year',
        'time_d', 'MAX_SPREAD_OVERRIDES', 'Trade_risk', 'Lot_units', 'reward_factor_range',
        'use_custom_initial_deposit', 'custom_initial_deposit', 'param_combinations','sampled_combinations',
        'logger', 'prepare_data', 'apply_signals', 'unique_params',
        'calculate_reward_factor', 
        'reward_factor_params_names', # ADDED
        'apply_signals_param_names',  # ADD THIS
        'prepare_data_param_names',
        'higher_tf_time_d', # For optional higher timeframe
        'SLIPPAGE_OVERRIDES',
        'final_results_path', 'checkpoints_path', 'No_of_processes', 
        'In_Sample_Run', 'IS_our_curr', 'iterations_sample_size', 
        'current_period', 'pairs_with_combinations', 'run_all_pairs_in_sample',
        'parameter_validation_function', 'trade_delay_hours',
        'calculate_sl_tp_pips_function', # ADD THIS
        'MIN_SL_PIPS', # ADD THIS
        'MIN_TP_PIPS'  # ADD THIS
    ]

    config_dict = {}
    for attr in attributes:
        value = getattr(Config, attr, None)
        config_dict[attr] = value

    # Ensure required attributes are always present
    if 'calculate_reward_factor' not in config_dict:
        config_dict['calculate_reward_factor'] = None
    if 'reward_factor_range' not in config_dict:
        config_dict['reward_factor_range'] = None
    if config_dict.get('reward_factor_params_names') is None:
        config_dict['reward_factor_params_names'] = []
    if config_dict.get('higher_tf_time_d') is None:
        config_dict['higher_tf_time_d'] = None
    if config_dict.get('prepare_data_param_names') is None:
        config_dict['prepare_data_param_names'] = []
    if config_dict.get('apply_signals_param_names') is None:
        config_dict['apply_signals_param_names'] = []
    if config_dict.get('parameter_validation_function') is None: # Add default
        config_dict['parameter_validation_function'] = None
    if config_dict.get('calculate_sl_tp_pips_function') is None: # Add default
        config_dict['calculate_sl_tp_pips_function'] = None


    return config_dict

def _get_granularity_string(time_d_value: Optional[int]) -> Optional[Tuple[str, str]]:
    """
    Converts integer time_d to granularity string and its typical file extension.
    Returns: Tuple (granularity_string, file_extension) or None if time_d_value is None.
    Raises: ValueError if time_d_value is an unsupported integer.
    """
    if time_d_value is None:
        return None
    
    gran_string: str
    extension: str

    # Minutes (parquet format for H1 and below)
    if time_d_value == 0:  # M1
        gran_string = "M1"
        extension = ".parquet"
    elif time_d_value == 5:  # M5
        gran_string = "M5"
        extension = ".parquet"
    elif time_d_value == 15:  # M15
        gran_string = "M15"
        extension = ".parquet"
    elif time_d_value == 30:  # M30
        gran_string = "M30"
        extension = ".parquet"
    # Hours (H1 uses parquet, H4+ uses pickle)
    elif time_d_value == 1:  # H1
        gran_string = "H1"
        extension = ".parquet"
    elif time_d_value == 4:  # H4
        gran_string = "H4"
        extension = ".pkl"
    elif time_d_value == 8:  # H8
        gran_string = "H8"
        extension = ".pkl"
    elif time_d_value == 12:  # H12
        gran_string = "H12"
        extension = ".pkl"
    elif time_d_value == 16:  # H16
        gran_string = "H16"
        extension = ".pkl"
    # Days and higher (pickle format)
    elif time_d_value == 24:  # Daily
        gran_string = "D"
        extension = ".pkl"
    elif time_d_value == 168:  # Weekly
        gran_string = "W"
        extension = ".pkl"
    else:
        raise ValueError(f"Unsupported time_d value for granularity string: {time_d_value}")
    
    return gran_string, extension


def _load_single_optimized_dataframe(file_path: str, logger_obj: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Loads a single Parquet or Pickle file, ensures DatetimeIndex, and UTC timezone."""
    try:
        if not os.path.exists(file_path):
            if logger_obj: logger_obj.error(f"File not found: {file_path}")
            return None

        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        else:
            if logger_obj: logger_obj.error(f"Unsupported file format: {file_path}")
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            if logger_obj: logger_obj.warning(f"Index is not DatetimeIndex in {file_path}. Attempting conversion.")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                if logger_obj: logger_obj.error(f"Failed to convert index to DatetimeIndex for {file_path}: {e}")
                return None
        
        # Timezone handling
        if df.index.tz is None:
            if logger_obj: logger_obj.warning(f"Timezone not set for {file_path}. Assuming UTC and localizing.")
            df = df.tz_localize('UTC')
        # MODIFIED ROBUST UTC CHECK
        elif str(df.index.tz) != 'UTC': 
            if logger_obj: 
                logger_obj.warning(f"Timezone is not UTC for {file_path} (it's {df.index.tz}). Converting to UTC.")
            df = df.tz_convert('UTC')
            
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df
    except Exception as e:
        if logger_obj:
            logger_obj.error(f"Error loading single optimized dataframe {file_path}: {str(e)}\n{traceback.format_exc()}")
        else:
            print(f"Error loading single optimized dataframe {file_path}: {str(e)}\n{traceback.format_exc()}")
        return None


def load_data(pair: str, start_year: int, end_year: int, 
              primary_time_d: int, higher_tf_time_d: Optional[int] = None, 
              logger: Optional[logging.Logger] = None, # Ensure logger is passed correctly
              data_path_prefix: str = "../optimized_data") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]: # Added data_path_prefix
    """
    Load primary, M1, and optionally a higher timeframe dataset from optimized files.
    All returned DataFrames will have a DatetimeIndex.
    """
    try:
        primary_gran_info = _get_granularity_string(primary_time_d)
        # M1 data is always time_d = 0 as per _get_granularity_string logic
        m1_gran_info = _get_granularity_string(0) 
        higher_tf_gran_info = _get_granularity_string(higher_tf_time_d) if higher_tf_time_d is not None else None

        if not primary_gran_info: # This also implicitly checks m1_gran_info due to hardcoded 0
            if logger: logger.error(f"Invalid primary_time_d: {primary_time_d} resulting in no granularity info.")
            return None, None, None
        
        # This check is technically redundant if primary_gran_info is valid, as _get_granularity_string(0) should always succeed or raise.
        # However, keeping it for explicit safety or if _get_granularity_string(0) could return None in some future logic.
        if not m1_gran_info:
             if logger: logger.error("Failed to get M1 granularity info (time_d=0). This should not happen.")
             return None, None, None

        primary_granularity, primary_ext = primary_gran_info
        m1_granularity, m1_ext = m1_gran_info
        
        primary_file_path = os.path.join(data_path_prefix, primary_granularity, f"{pair}_{primary_granularity}{primary_ext}")
        m1_file_path = os.path.join(data_path_prefix, m1_granularity, f"{pair}_{m1_granularity}{m1_ext}")
        
        df_primary = _load_single_optimized_dataframe(primary_file_path, logger_obj=logger)
        df_m1 = _load_single_optimized_dataframe(m1_file_path, logger_obj=logger)
        
        df_higher_tf = None
        if higher_tf_gran_info:
            higher_tf_granularity, higher_tf_ext = higher_tf_gran_info
            if higher_tf_granularity == primary_granularity:
                if logger: logger.warning(f"Higher timeframe ({higher_tf_granularity}) is same as primary. Using primary's data for HTF.")
                df_higher_tf = df_primary.copy() if df_primary is not None else None
            else:
                higher_tf_file_path = os.path.join(data_path_prefix, higher_tf_granularity, f"{pair}_{higher_tf_granularity}{higher_tf_ext}")
                df_higher_tf = _load_single_optimized_dataframe(higher_tf_file_path, logger_obj=logger)
        
        date_start_dt = pd.Timestamp(f"{start_year}-01-01 00:00:00", tz='UTC')
        date_end_dt = pd.Timestamp(f"{end_year}-12-31 23:59:59.999999", tz='UTC')
        
        filtered_dfs = []
        for df_instance in [df_primary, df_m1, df_higher_tf]:
            if df_instance is not None:
                if not df_instance.index.is_monotonic_increasing:
                    df_instance = df_instance.sort_index()
                df_filtered = df_instance.loc[date_start_dt:date_end_dt].copy()
                if df_filtered.empty and logger:
                    logger.warning(f"Data for {pair} (granularity derived from input) is empty after date filtering ({start_year}-{end_year}).")
                filtered_dfs.append(df_filtered)
            else:
                filtered_dfs.append(None)
            
        return tuple(filtered_dfs) # type: ignore

    except Exception as e:
        if logger: logger.error(f"General error in load_data for pair {pair} (Primary time_d={primary_time_d}, HTF time_d={higher_tf_time_d}): {str(e)}\n{traceback.format_exc()}")
        return None, None, None


def load_checkpoint(file_path, config_dict):
    """Loads the checkpoint data from a file."""
    try:
        with open(file_path, 'rb') as file:
            checkpoint_data = pickle.load(file)
        # Ensure trades are loaded as a list of DataFrames if they exist
        if 'trades' in checkpoint_data and not isinstance(checkpoint_data['trades'], list):
            checkpoint_data['trades'] = [] # Or attempt to reconstruct if format is known
        return checkpoint_data
    except FileNotFoundError:
        return {'last_index': 0, 'results': [], 'trades': [], 'num_iterations': 0}
    except Exception as e:
        logger = config_dict.get('logger')
        if logger: logger.error(f"Failed to load checkpoint {file_path}: {e}")
        return {'last_index': 0, 'results': [], 'trades': [], 'num_iterations': 0}

def save_checkpoint(file_path, checkpoint_data, config_dict):
    """Saves the checkpoint data to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(checkpoint_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger = config_dict.get('logger')
        if logger: logger.error(f"Failed to save checkpoint {file_path}: {e}")


def simulate_params(pair: str, df_primary_orig: pd.DataFrame, df_m1_orig: pd.DataFrame, 
                    df_higher_tf_orig: Optional[pd.DataFrame], 
                    time_d: int, pip_location: float, config_dict: Dict[str, Any], 
                    *params: Any) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Simulate with provided parameters. apply_signals is now called here."""
    logger = config_dict.get('logger') # Get logger early
    try:
        param_names_iterated = config_dict['unique_params'][1:]  # skip 'pair', these are the iterated params
        param_dict = dict(zip(param_names_iterated, params))

        prepare_data_param_names = config_dict.get('prepare_data_param_names', [])
        # Ensure all names in prepare_data_param_names are in param_dict (iterated params)
        missing_prep_params = [name for name in prepare_data_param_names if name not in param_dict]
        if missing_prep_params:
            raise ValueError(f"Missing parameters for prepare_data: {missing_prep_params}. Available iterated params: {list(param_dict.keys())}")
        prepare_data_args = [param_dict[name] for name in prepare_data_param_names]

        # Calculate reward_factor
        reward_factor_params_names = config_dict.get('reward_factor_params_names', [])
        if config_dict['calculate_reward_factor'] is not None and reward_factor_params_names:
            # Ensure all names for reward factor calculation are in param_dict
            missing_rf_params = [name for name in reward_factor_params_names if name not in param_dict]
            if missing_rf_params:
                raise ValueError(f"Missing parameters for calculate_reward_factor: {missing_rf_params}. Available iterated params: {list(param_dict.keys())}")
            rf_calc_args = [param_dict[name] for name in reward_factor_params_names]
            reward_factor = config_dict['calculate_reward_factor'](config_dict['logger'], *rf_calc_args)
        elif 'reward_factor' in param_dict: # If reward_factor is one of the iterated params
            reward_factor = np.float32(param_dict['reward_factor'])
        else: 
            # reward_factor = np.float32(1.5) 
            if config_dict['logger']:
                config_dict['logger'].warning("Reward factor not in iterated params and not calculable by name; using default 1.5.")
            else:
                raise ValueError("Reward factor not provided in iterated params and no calculate_reward_factor function defined. Cannot proceed.")
            

        df_primary_copy = df_primary_orig.copy() # Work on a copy
        
        if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
            logger.debug(f"Pair {pair}, Params {params}: df_primary_copy (before prepare_data) shape: {df_primary_copy.shape if df_primary_copy is not None else 'None'}")
            if df_higher_tf_orig is not None:
                 logger.debug(f"Pair {pair}, Params {params}: df_higher_tf_orig (before prepare_data) shape: {df_higher_tf_orig.shape if df_higher_tf_orig is not None else 'None'}")


        # Call prepare_data (strategy specific)
        if df_higher_tf_orig is not None and config_dict.get('higher_tf_time_d') is not None:
            # Pass a copy of higher_tf_orig as well
            df_prepared = config_dict['prepare_data'](df_primary_copy, df_higher_tf_orig.copy(), *prepare_data_args, logger=logger)
        else:
            df_higher_tf_orig = None # Ensure it's None if not provided
            df_prepared = config_dict['prepare_data'](df_primary_copy, df_higher_tf_orig, *prepare_data_args, logger=logger)
        
        # --- GENERIC, CONFIGURABLE, PAIR-SPECIFIC VALIDATION ---
        validation_func = config_dict.get('calculate_sl_tp_pips_function')
        if callable(validation_func):
            if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
                logger.debug(f"Validating params for {pair} with params: {param_dict}, prepared data columns: {df_prepared.columns.tolist() if df_prepared is not None else 'None'}, pip_location: {pip_location}")
            sl_pips, tp_pips = validation_func(param_dict, df_prepared, pip_location)
            if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
                logger.debug(f"Validation results for {pair} with params {param_dict}: SL = {sl_pips}, TP = {tp_pips}")
            if sl_pips is not None:
                MIN_SL_PIPS = config_dict.get('MIN_SL_PIPS', 5.0)
                if sl_pips < MIN_SL_PIPS:
                    if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
                        logger.debug(f"SKIPPING combo for {pair}: SL of {sl_pips:.2f} pips is below the configured minimum of {MIN_SL_PIPS}.")
                    return None, None
            
            if tp_pips is not None:
                MIN_TP_PIPS = config_dict.get('MIN_TP_PIPS', 5.0)
                if tp_pips < MIN_TP_PIPS:
                    if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
                        logger.debug(f"SKIPPING combo for {pair}: TP of {tp_pips:.2f} pips is below the configured minimum of {MIN_TP_PIPS}.")
                    return None, None
        # --- END OF GENERIC VALIDATION LOGIC ---

        if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
            logger.debug(f"Pair {pair}, df_prepared (after prepare_data) shape: {df_prepared.shape if df_prepared is not None else 'None'}")
            logger.debug(f"df_prepared columns: {df_prepared.columns.tolist() if df_prepared is not None else 'None'}")
            if df_higher_tf_orig is not None:
                logger.debug(f"Pair {pair}, df_higher_tf_orig (after prepare_data) shape: {df_higher_tf_orig.shape if df_higher_tf_orig is not None else 'None'}")
                logger.debug(f"df_higher_tf_orig columns: {df_higher_tf_orig.columns.tolist() if df_higher_tf_orig is not None else 'None'}")
        # Call apply_signals (strategy specific) on the prepared data
        # max_spread_absolute = np.float32(config_dict['max_spread'] * pip_location) # DEPRECATED
        
        # Get apply_signals parameter names from config
        apply_signals_param_names = config_dict.get('apply_signals_param_names', [])
        
        if apply_signals_param_names:
            # Pass strategy-specific parameters via kwargs
            apply_signals_kwargs = {name: param_dict[name] for name in apply_signals_param_names 
                               if name in param_dict}
            apply_signals_kwargs['pair'] = pair  # Always include pair
            
            df_with_signals = config_dict['apply_signals'](
                df_prepared, reward_factor, 
                logger=logger, **apply_signals_kwargs
            )
        else:
            # Fallback for strategies that don't need extra parameters
            df_with_signals = config_dict['apply_signals'](
                df_prepared, reward_factor, logger=logger
            )

        debug_file_path = None
        if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
             # Construct the base debug file path using pair and params
            debug_file_path = os.path.join(
                config_dict['trade_res_path'],
                f"df_sig_{'_'.join(str(v) for v in [pair] + list(params))}.pkl"
            )
            # Save df_with_signals to this path
            if df_with_signals is not None:
                try:
                    df_with_signals.to_pickle(debug_file_path)
                    logger.debug(f"Saved df_with_signals to {debug_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save df_with_signals to {debug_file_path}: {e}")
                    logger.error(traceback.format_exc())

            logger.debug(f"Pair {pair}, Params {params}: df_primary_orig shape: {df_primary_orig.shape if df_primary_orig is not None else 'None'}, df_m1_orig shape: {df_m1_orig.shape if df_m1_orig is not None else 'None'}")
            if df_higher_tf_orig is not None:
                logger.debug(f"Pair {pair}, Params {params}: df_higher_tf_orig shape: {df_higher_tf_orig.shape}")
            else:
                logger.debug(f"Pair {pair}, Params {params}: df_higher_tf_orig is None.")
            logger.debug(f"Pair {pair}, Params {params}: df_prepared shape: {df_prepared.shape if df_prepared is not None else 'None'}")
            logger.debug(f"Pair {pair}, Params {params}: df_with_signals shape: {df_with_signals.shape if df_with_signals is not None else 'None'}")
            if df_with_signals is not None and 'SIGNAL' in df_with_signals.columns:
                logger.info(f"Pair {pair}, Params {params}: Signal counts in df_with_signals: {df_with_signals['SIGNAL'].value_counts().to_dict()}")
            elif df_with_signals is not None:
                logger.warning(f"Pair {pair}, Params {params}: 'SIGNAL' column missing in df_with_signals. Columns: {df_with_signals.columns.tolist()}")
            else:
                logger.warning(f"Pair {pair}, Params {params}: df_with_signals is None after apply_signals.")

        # Initialize tester with data that has signals applied
        hm = Tester(
            df_with_signals,      # This DataFrame has signals and is DatetimeIndexed
            df_m1_orig.copy(),    # Pass a copy of m1 data (already DatetimeIndexed)
            pip_location,
            config_dict['MAX_SPREAD_OVERRIDES'], # UPDATED
            config_dict['Lot_units'],
            float(reward_factor), # Pass the calculated/retrieved reward_factor, cast to float
            time_d=time_d, # This is primary_time_d
            pair_name=pair,
            slippage_overrides=config_dict.get('SLIPPAGE_OVERRIDES', {}),
            trade_delay_hours=param_dict.get('trade_delay_hours', 0),
            logger=logger,        # Pass logger to Tester.
            debug_file_path=debug_file_path # Pass debug_file_path to Tester.
        )
        
        df_trades = hm.run_test_vectorized()

        if logger and isinstance(logger, LogWrapper) and logger.level <= logging.DEBUG:
            logger.info(f"Pair {pair}, Params {params}: df_trades shape: {df_trades.shape if df_trades is not None and not df_trades.empty else 'None or Empty'}")

        if df_trades is not None and not df_trades.empty:
            for param_name, param_value in zip(param_names_iterated, params):
                if isinstance(param_value, float):
                    df_trades[param_name] = np.float32(param_value)
                elif isinstance(param_value, int):
                    df_trades[param_name] = np.int32(param_value)
                else:
                    df_trades[param_name] = param_value # Or handle other types
            df_trades['pair'] = pair
            return df_trades, float(reward_factor)
        else:
            return None, None
    except Exception as e:
        # logger already defined at the start of the try block
        if logger: logger.error(f"Error in simulate_params for pair {pair}, params {params}: {e}\n{traceback.format_exc()}")
        return None, None


def process_combinations(
    param_combinations_to_run: List[Tuple], 
    start_idx_offset: int, # The starting index in the *original full* list of combinations for this pair
    df_primary_loaded: pd.DataFrame, 
    df_m1_loaded: pd.DataFrame, 
    df_higher_tf_loaded: Optional[pd.DataFrame],
    time_d: int, 
    pip_location: float, 
    pair: str, 
    config_dict: Dict[str, Any],
    # Parameters for intra-batch checkpointing:
    checkpoint_file: str,
    checkpoint_interval: int,
    initial_results_for_checkpoint: List[Dict[str, Any]], # Results loaded by run_pair before this batch
    initial_trades_for_checkpoint: List[Optional[pd.DataFrame]], # Trades loaded by run_pair before this batch
    initial_iterations_for_checkpoint: int # Iterations loaded by run_pair before this batch
) -> Tuple[List[Dict[str, Any]], List[Optional[pd.DataFrame]], int]:
    """
    Process parameter combinations.
    Handles intra-batch checkpointing.
    Returns results, trades, and iteration count *for this batch only*.
    """
    results_this_batch: List[Dict[str, Any]] = []
    trades_this_batch: List[Optional[pd.DataFrame]] = []
    num_iterations_this_batch = 0

    if config_dict.get('use_custom_initial_deposit', False):
        initial_deposit = np.float32(config_dict['custom_initial_deposit'])
    else:
        initial_deposit = np.float32(1000000) if pip_location == 0.01 else np.float32(10000)
    
    logger = config_dict.get('logger')
    param_names_iterated = config_dict['unique_params'][1:]

    try:
        progress_bar = tqdm(
            enumerate(param_combinations_to_run), # enumerate gives current_batch_idx
            desc=f"Processing {pair}", 
            total=len(param_combinations_to_run), 
            leave=False, 
            miniters=max(1, len(param_combinations_to_run) // 100)
        )
        
        iteration_times = []
        current_batch_idx = 0 # To keep track if param_combinations_to_run is empty

        for current_batch_idx, params_tuple in progress_bar:
            # overall_idx_for_saving is the index in the original full list of combinations for this pair
            overall_idx_for_saving = start_idx_offset + current_batch_idx
            
            start_time_iter = time.perf_counter()
            
            sim_result = simulate_params(
                pair, df_primary_loaded, df_m1_loaded, df_higher_tf_loaded, 
                time_d, pip_location, config_dict, *params_tuple
            )
            sim_trades_df, rf_value = sim_result if sim_result is not None else (None, None)
            
            if sim_trades_df is not None and not sim_trades_df.empty:
                all_met = AllMetrics(sim_trades_df, config_dict['Trade_risk'], initial_deposit, 
                                     config_dict['start_year'], config_dict['end_year'], logger)
                metrics_dict_iter = all_met.get_metrics_dict()
                
                result_dict_iter = {'pair': pair}
                result_dict_iter.update(metrics_dict_iter)
                
                if 'reward_factor' not in param_names_iterated:
                    result_dict_iter['reward_factor'] = rf_value
                    
                for p_name, p_val in zip(param_names_iterated, params_tuple):
                    result_dict_iter[p_name] = p_val
                
                results_this_batch.append(result_dict_iter)
                trades_this_batch.append(sim_trades_df)
                num_iterations_this_batch += 1
            else:
                # Append None to trades_this_batch if you want to keep a 1:1 mapping with param_combinations_to_run
                # For simplicity, we only append actual trade DataFrames or None if sim_trades_df is None
                if sim_trades_df is None: # No trades generated
                     trades_this_batch.append(None)


            # Intra-batch checkpointing
            if checkpoint_interval > 0 and \
               ((current_batch_idx + 1) % checkpoint_interval == 0 or \
                (current_batch_idx + 1) == len(param_combinations_to_run)):
                
                # Data to save in checkpoint: combines initial state with current batch's progress
                results_to_save_in_cp = initial_results_for_checkpoint + results_this_batch
                trades_to_save_in_cp = initial_trades_for_checkpoint + [tdf for tdf in trades_this_batch if tdf is not None] # Filter Nones for saving
                iterations_to_save_in_cp = initial_iterations_for_checkpoint + num_iterations_this_batch

                save_checkpoint(checkpoint_file, {
                    'last_index': overall_idx_for_saving + 1, # Next index to start from
                    'results': results_to_save_in_cp,
                    'trades': trades_to_save_in_cp,
                    'num_iterations': iterations_to_save_in_cp
                }, config_dict)
                if logger:
                    logger.debug(f"Intra-pair checkpoint saved for {pair} at combination index {overall_idx_for_saving + 1}")

            end_time_iter = time.perf_counter()
            iteration_time = end_time_iter - start_time_iter
            iteration_times.append(iteration_time)
            
            if (current_batch_idx + 1) % 10 == 0 or (current_batch_idx + 1) == len(param_combinations_to_run):
                recent_avg = sum(iteration_times[-10:]) / min(10, len(iteration_times)) if iteration_times else 0
                progress_bar.set_postfix_str(f"Avg last 10: {recent_avg:.3f}s")
        
        if iteration_times and logger:
            logger.info(f"{pair} batch processed: {len(iteration_times)} iterations, Avg: {sum(iteration_times)/len(iteration_times):.3f}s, Total: {sum(iteration_times):.1f}s")
    
        return results_this_batch, trades_this_batch, num_iterations_this_batch
        
    except Exception as e:
        # Save checkpoint on error during batch processing
        # overall_idx_for_saving might not be perfectly up-to-date if error is before its calculation in loop
        # Using current_batch_idx which is defined outside the loop for safety
        overall_idx_at_error = start_idx_offset + current_batch_idx 
        
        results_to_save_on_error = initial_results_for_checkpoint + results_this_batch
        trades_to_save_on_error = initial_trades_for_checkpoint + [tdf for tdf in trades_this_batch if tdf is not None]
        iterations_to_save_on_error = initial_iterations_for_checkpoint + num_iterations_this_batch
        
        save_checkpoint(checkpoint_file, {
            'last_index': overall_idx_at_error + 1, # Save next index to attempt
            'results': results_to_save_on_error,
            'trades': trades_to_save_on_error,
            'num_iterations': iterations_to_save_on_error
        }, config_dict)
        if logger: 
            logger.error(f"Error in process_combinations for {pair} (batch_idx {current_batch_idx}, overall_idx {overall_idx_at_error}). Checkpoint saved. Error: {e}\n{traceback.format_exc()}")
        
        # Return what has been processed so far in this batch before the error
        return results_this_batch, trades_this_batch, num_iterations_this_batch


def run_pair(args: Tuple) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Run backtest for a single currency pair, save results directly to disk,
    and return status and file paths. Includes resume-safety for parquet files.
    MODIFIED for memory efficiency and robustness.
    """
    pair: str
    pip_location: float
    checkpoint_dir: str
    config_dict_global: Dict[str, Any]
    param_combinations_for_pair: List[Tuple]

    # Define variables with default values before the try block
    df_primary_loaded, df_m1_loaded, df_higher_tf_loaded, checkpoint_data = None, None, None, None

    if len(args) == 4: 
        pair, pip_location, checkpoint_dir, config_dict_global = args
        param_combinations_for_pair = config_dict_global.get('sampled_combinations', [])
    elif len(args) == 5: 
        pair, pip_location, checkpoint_dir, config_dict_global, param_combinations_for_pair = args
    else:
        logger_temp = config_dict_global.get('logger') if 'config_dict_global' in locals() and isinstance(config_dict_global, dict) else None
        if logger_temp: logger_temp.error(f"run_pair called with incorrect number of arguments: {len(args)}")
        return False, "ArgumentError", None, None

    checkpoint_file = os.path.join(checkpoint_dir, f"{pair}_checkpoint.pkl")
    config_dict_pair = config_dict_global.copy() 
    logger = config_dict_pair.get('logger')

    try:
        df_primary_loaded, df_m1_loaded, df_higher_tf_loaded = load_data(
            pair, 
            config_dict_pair['start_year'], 
            config_dict_pair['end_year'], 
            config_dict_pair['time_d'],
            config_dict_pair.get('higher_tf_time_d'),
            logger=logger
        )

        if df_primary_loaded is None or df_primary_loaded.empty or \
           df_m1_loaded is None or df_m1_loaded.empty:
            if logger: logger.error(f"Data loading failed or returned empty for primary/M1 for {pair}. Skipping.")
            return False, pair, None, None
            
        checkpoint_data = load_checkpoint(checkpoint_file, config_dict_pair)
        start_idx_for_this_run = checkpoint_data.get('last_index', 0)
        
        if not isinstance(param_combinations_for_pair, list):
            if logger: logger.error(f"param_combinations_for_pair for {pair} is not a list. Type: {type(param_combinations_for_pair)}")
            param_combinations_for_pair = []

        param_combinations_to_process_now = param_combinations_for_pair[start_idx_for_this_run:]
        
        if not param_combinations_to_process_now:
            if logger: logger.info(f"All combinations already processed for {pair} as per checkpoint (last_index: {start_idx_for_this_run}).")
            
            period_id = config_dict_pair.get('current_period', 'unknown_period')
            pair_results_path = os.path.join(config_dict_pair['final_results_path'], f"res_{period_id}_{pair}.parquet")
            
            # --- MODIFIED: Memory-Efficient Resume Safety Guard ---
            if not os.path.exists(pair_results_path) and checkpoint_data.get('results'):
                if logger: logger.warning(f"Parquet file missing for completed pair {pair}. Regenerating from checkpoint.")
                
                results_df_final = pd.DataFrame(checkpoint_data['results'])
                
                valid_trades_dfs = [df for df in checkpoint_data.get('trades', []) if isinstance(df, pd.DataFrame) and not df.empty]
                trades_df_final = pd.concat(valid_trades_dfs, ignore_index=True) if valid_trades_dfs else pd.DataFrame()
                
                pair_trades_path = os.path.join(config_dict_pair['trade_res_path'], f"trades_{period_id}_{pair}.parquet")
                
                results_df_final.to_parquet(pair_results_path)
                if not trades_df_final.empty:
                    trades_df_final.to_parquet(pair_trades_path)
                
                # Explicitly release memory
                del results_df_final, trades_df_final, valid_trades_dfs
                gc.collect()
                
                if logger: logger.info(f"Successfully regenerated missing parquet files for {pair}.")
                return True, pair, pair_results_path, (pair_trades_path if os.path.exists(pair_trades_path) else None)
            
            return True, pair, (pair_results_path if os.path.exists(pair_results_path) else None), None

        # If we are here, we need the full checkpoint data to proceed
        results_accumulated_before_batch = checkpoint_data.get('results', [])
        trades_dfs_accumulated_before_batch = checkpoint_data.get('trades', [])
        iterations_accumulated_before_batch = checkpoint_data.get('num_iterations', 0)

        if logger: logger.info(f"Running {pair} with {len(param_combinations_to_process_now)} combinations (resuming from overall index {start_idx_for_this_run}).")
        
        results_from_batch, trades_dfs_from_batch, iterations_from_batch = process_combinations(
            param_combinations_to_process_now, 
            start_idx_for_this_run,
            df_primary_loaded, df_m1_loaded, df_higher_tf_loaded,
            config_dict_pair['time_d'], pip_location, 
            pair, config_dict_pair,
            checkpoint_file,
            config_dict_pair.get('Chk_pt_interval', 0),
            results_accumulated_before_batch,
            trades_dfs_accumulated_before_batch,
            iterations_accumulated_before_batch
        )

        final_results_for_pair = results_accumulated_before_batch + results_from_batch
        final_trades_dfs_for_pair = trades_dfs_accumulated_before_batch + [tdf for tdf in trades_dfs_from_batch if tdf is not None]
        final_total_iterations_for_pair = iterations_accumulated_before_batch + iterations_from_batch
        
        final_last_index = start_idx_for_this_run + len(param_combinations_to_process_now)
        
        save_checkpoint(checkpoint_file, {
            'last_index': final_last_index,
            'results': final_results_for_pair,
            'trades': final_trades_dfs_for_pair, 
            'num_iterations': final_total_iterations_for_pair
        }, config_dict_pair)
        if logger: logger.info(f"Final checkpoint saved for {pair} at index {final_last_index}.")

        if final_results_for_pair:
            results_df_final = pd.DataFrame(final_results_for_pair)
            for col in results_df_final.select_dtypes(include=['float64']).columns:
                results_df_final[col] = results_df_final[col].astype(np.float32)
            
            valid_trades_dfs_final = [df for df in final_trades_dfs_for_pair if isinstance(df, pd.DataFrame) and not df.empty]
            trades_df_final = pd.concat(valid_trades_dfs_final, ignore_index=True) if valid_trades_dfs_final else pd.DataFrame()
            
            period_id = config_dict_pair.get('current_period', 'unknown_period')
            pair_results_path = os.path.join(config_dict_pair['final_results_path'], f"res_{period_id}_{pair}.parquet")
            pair_trades_path = os.path.join(config_dict_pair['trade_res_path'], f"trades_{period_id}_{pair}.parquet")

            try:
                cf.create_directory_if_not_exists(config_dict_pair['final_results_path'])
                cf.create_directory_if_not_exists(config_dict_pair['trade_res_path'])
                
                results_df_final.to_parquet(pair_results_path)
                if not trades_df_final.empty:
                    trades_df_final.to_parquet(pair_trades_path)
                
                if logger: logger.info(f"Saved final results for {pair} to {pair_results_path}")
                
                return True, pair, pair_results_path, (pair_trades_path if not trades_df_final.empty else None)

            except Exception as e_save:
                if logger: logger.error(f"Error saving final files for {pair}: {e_save}")
                return False, pair, None, None
        else:
            if logger: logger.warning(f"No results generated for pair: {pair} after this run.")
            return True, pair, None, None

    except KeyboardInterrupt:
        if logger: logger.error(f"KeyboardInterrupt during {pair}: Stopping execution for this pair.")
        return False, pair, None, None
    except Exception as e:
        if logger: logger.error(f"Critical error in run_pair for {pair}: {str(e)}\n{traceback.format_exc()}.")
        return False, pair, None, None
    finally:
        # --- MODIFIED: Robust Garbage Collection ---
        # Only delete variables if they have been assigned.
        if df_primary_loaded is not None: del df_primary_loaded
        if df_m1_loaded is not None: del df_m1_loaded
        if df_higher_tf_loaded is not None: del df_higher_tf_loaded
        if checkpoint_data is not None: del checkpoint_data
        gc.collect()


def run_Strategy(Config: Any) -> Optional[pd.DataFrame]:
    """Main entry point for running strategy with optimized execution."""
    try:
        from basic.infrastructure.instrument_collection import instrumentCollection as ic
        ic.LoadInstruments(getattr(Config, 'instrument_data_path', "../optimized_data")) 
        
        logger = Config.logger
        
        if Config.In_Sample_Run:
            pairs_to_run_names = [p for p in Config.IS_our_curr if p in ic.instruments_dict]
            
            param_combinations_source = getattr(Config, 'sampled_combinations', [])
            total_combinations_count_info = len(param_combinations_source)
        else:
            pairs_to_run_names = list(Config.pairs_with_combinations.keys())
            total_combinations_count_info = sum(len(v) for v in Config.pairs_with_combinations.values())

        logger.info(f"Engine running {Config.Strategy_name} with {Config.No_of_processes} processes parallely on {len(pairs_to_run_names)} pairs. "
                    f"Total combinations (approx per pair if In_Sample_Run, total if Out-Sample): {total_combinations_count_info}")
        
        pip_locations_map = {pair: ic.instruments_dict[pair].pipLocation 
                             for pair in pairs_to_run_names if pair in ic.instruments_dict}
        
        valid_pairs_to_run = [p for p in pairs_to_run_names if p in pip_locations_map]
        if len(valid_pairs_to_run) < len(pairs_to_run_names):
            logger.warning(f"Filtered out {len(pairs_to_run_names) - len(valid_pairs_to_run)} pairs due to missing pip location.")

        if not valid_pairs_to_run:
            logger.error("No valid pairs to run after checking pip locations. Exiting.")
            return None

        cf.save_parameters_automatically(Config.final_results_path, Config)
        cf.create_directory_if_not_exists(Config.checkpoints_path)
        cf.create_directory_if_not_exists(Config.trade_res_path)
        
        config_dict_global = extract_config_attributes(Config)
        
        process_args_list = []
        if Config.In_Sample_Run:            
            for pair_name in valid_pairs_to_run:
                process_args_list.append((pair_name, pip_locations_map[pair_name], Config.checkpoints_path, config_dict_global))
        else:
            for pair_name in valid_pairs_to_run:
                if pair_name in Config.pairs_with_combinations:
                    pair_specific_combs = Config.pairs_with_combinations[pair_name]
                    process_args_list.append((pair_name, pip_locations_map[pair_name], Config.checkpoints_path, config_dict_global, pair_specific_combs))
                else:
                    logger.warning(f"Pair {pair_name} specified for Out-Sample run but no combinations found.")
        
        if not process_args_list:
            logger.error("No arguments generated for processing pairs. Exiting.")
            return None
        
        all_run_results_tuples: List[Tuple[bool, str, Optional[str], Optional[str]]] = []
        
        if Config.No_of_processes > 1:
            all_run_results_tuples = process_map(
                run_pair, 
                process_args_list, 
                max_workers=Config.No_of_processes,
                chunksize=1,
                desc="Processing pairs in parallel"
            )
        else:
            logger.info("Running pairs sequentially (No_of_processes <= 1).")
            for args_item in tqdm(process_args_list, desc="Processing pairs sequentially"):
               all_run_results_tuples.append(run_pair(args_item))

        successful_results_paths = [res_tuple[2] for res_tuple in all_run_results_tuples if res_tuple and res_tuple[0] and res_tuple[2]]

        if successful_results_paths:
            cf.create_directory_if_not_exists(Config.final_results_path)
            
            logger.info(f"Aggregating results from {len(successful_results_paths)} individual pair files...")
            list_of_results_dfs = [pd.read_parquet(p) for p in successful_results_paths]
            combined_results_df = pd.concat(list_of_results_dfs, ignore_index=True)
            
            # ADDED: Explicit garbage collection after concatenation
            del list_of_results_dfs
            gc.collect()
            
            for col in combined_results_df.select_dtypes(include=['float64']).columns:
                combined_results_df[col] = combined_results_df[col].astype(np.float32)
            
            period_id = getattr(Config, 'current_period', Config.Strategy_name)
            
            results_file_path = os.path.join(Config.final_results_path, f"res_{period_id}.parquet")
            combined_results_df.to_parquet(results_file_path, compression='snappy', engine='pyarrow')
            logger.info(f"Saved combined results to {results_file_path}")
            
            if hasattr(Config, 'in_sample_years'): 
                wfa_excel_wfa(Config, 'Engine')
            else: 
                gen_excel_standard(Config, 'Engine')
                analyze_allresults(Config) 
            
            logger.info(f"Engine completed {period_id} run.")
            return combined_results_df.head(5)
        else:
            logger.warning(f"Engine generated no results for any pairs for period {getattr(Config, 'current_period', 'unknown')}")
            return None

    except KeyboardInterrupt:
        logger_ref = Config.logger if 'Config' in locals() and hasattr(Config, 'logger') else logging.getLogger()
        logger_ref.error("KeyboardInterrupt in run_Strategy: Stopping execution.")
        sys.exit(0)
    except Exception as e:
        logger_ref = Config.logger if 'Config' in locals() and hasattr(Config, 'logger') else logging.getLogger()
        logger_ref.error(f"Critical error in run_Strategy: {str(e)}\n{traceback.format_exc()}")
        return None



