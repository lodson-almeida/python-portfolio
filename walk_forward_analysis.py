import matplotlib
matplotlib.use('Agg')
import os
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set
from backtest_engine.generic import gen_engine as Engine
from backtest_engine.WFA import wfa_analyzer as Analyzer
from backtest_engine.WFA import param_freq_sheet
from backtest_engine.WFA import visualization as vs
import basic.custom_functions as cf
from basic.log_wrapper import LogWrapper # Added import
import traceback
import time
import glob
import gc  # Add garbage collector import
from backtest_engine.generic.universal_config_loader import get_param_combinations, get_config

def create_wfa_folder_structure(strategy_name: str, base_path: str, periods: List[Dict]) -> Tuple[Dict, str]:
    """
    Create the folder structure for Walk-Forward Analysis with proper permissions.
    """
    paths = {}
    strategy_path = os.path.join(base_path, strategy_name)

    try:
        # Create base strategy directory with full permissions
        if not os.path.exists(strategy_path):
            os.makedirs(strategy_path, mode=0o777)
        elif not os.access(strategy_path, os.W_OK):
            os.chmod(strategy_path, 0o777)

        for i, period in enumerate(periods, 1):
            in_sample_start, in_sample_end = period['in_sample']
            out_sample_start, out_sample_end = period['out_sample']

            # Create In-Sample folder with permissions
            in_sample_folder = f"In_Sample_{in_sample_start}-{in_sample_end}"
            in_sample_path = os.path.join(strategy_path, in_sample_folder)
            os.makedirs(in_sample_path, mode=0o777, exist_ok=True)

            # Create subfolders for In-Sample with permissions
            for subfolder in ['checkpoints', 'final_results', 'trades']:
                subfolder_path = os.path.join(in_sample_path, subfolder)
                os.makedirs(subfolder_path, mode=0o777, exist_ok=True)

            # Create Out-Sample folder with permissions
            out_sample_folder = f"Out_Sample_{out_sample_start}-{out_sample_end}"
            out_sample_path = os.path.join(strategy_path, out_sample_folder)
            os.makedirs(out_sample_path, mode=0o777, exist_ok=True)

            # Store paths
            paths[f"period_{i}"] = {
                "in_sample": in_sample_path,
                "out_sample": out_sample_path
            }

    except Exception as e:
        print(f"Error creating folder structure: {str(e)}")
        raise

    return paths, strategy_path

def save_results(Config: Any, strategy_path: str, all_params: Dict[str, Any]) -> None:
    """Save the results of the walk-forward analysis."""
    try:
        json_path = os.path.join(strategy_path, f"{Config.Strategy_name}_all_params.json")
        csv_path = os.path.join(strategy_path, f"{Config.Strategy_name}_all_params.csv")

        with open(json_path, 'w') as f:
            json.dump(all_params, f, indent=4)
        
        # Create DataFrame with explicit dtypes to save memory
        pd.DataFrame(all_params).T.to_csv(csv_path)

        Config.logger.info(f"Results saved to {strategy_path}")
    except Exception as e:
        Config.logger.error(f"Error saving results: {str(e)}")
        Config.logger.error(traceback.format_exc())

def analyze_parameter_frequency(Config, strategy_path, all_Top_results):
    try:
        if not all_Top_results:
            Config.logger.warning("No results to analyze")
            return {}
        
        combined_df = param_freq_sheet.combine_results(all_Top_results)
        total_periods = param_freq_sheet.get_total_periods(combined_df)
        parameter_analysis = param_freq_sheet.analyze_filter_pair(Config, combined_df, total_periods)
        
        param_freq_sheet.save_analysis_to_json(parameter_analysis, strategy_path, Config)
        
        summary_df = param_freq_sheet.create_summary_df(parameter_analysis, Config)
        safe_excel_path = param_freq_sheet.save_summary_to_excel(summary_df, strategy_path, Config, all_Top_results)
        
        Config.logger.info(f"Parameter frequency analysis saved to {safe_excel_path}")
        return parameter_analysis
        
    except Exception as e:
        Config.logger.error(f"Error in analyze_parameter_frequency: {str(e)}")
        Config.logger.error(traceback.format_exc())
        return {}

def generate_sampled_combinations(Config) -> List:
    """
    Generates and samples parameter combinations, returning only the sample.
    The total validated count is stored on Config.total_validated_combinations.
    """
    start_time = time.time()
    try:
        logger = Config.logger

        if hasattr(Config, 'RANDOM_SEED') and Config.RANDOM_SEED is not None:
            logger.info(f"Using fixed RANDOM_SEED: {Config.RANDOM_SEED} for reproducible sampling.")
            np.random.seed(Config.RANDOM_SEED)
        else:
            logger.info("No RANDOM_SEED found. Sampling will be non-deterministic.")
        
        logger.info(f"Initial unique_params for sampling: {Config.unique_params}")
        logger.info(f"Requested sample size: {Config.iterations_sample_size}")
        
        # USE UNIVERSAL CONFIG LOADER - This prevents multiple imports!
        if hasattr(Config, 'generate_all_combinations') and callable(Config.generate_all_combinations):
            logger.info(f"Calling dedicated function '{Config.generate_all_combinations.__name__}' to generate all combinations...")
            param_combinations_source = Config.generate_all_combinations()
            logger.info(f"Generated {len(param_combinations_source):,} combinations from function.")
        else:
            # Fallback for older configs or different structures
            logger.info("Using universal_config_loader to get combinations...")
            param_combinations_source = get_param_combinations()
            logger.info(f"Using universal cached combinations ({len(param_combinations_source):,} items)")
        
        if not param_combinations_source:
            logger.error("Parameter combinations source is empty. Cannot generate samples.")
            return []
        
        # Convert to list of tuples for set operations later
        initial_combinations_from_config = [tuple(comb) for comb in param_combinations_source]
        logger.info(f"Total available combinations before validation: {len(initial_combinations_from_config):,}")

        # MEMORY OPTIMIZATION: Clear the source after conversion
        param_combinations_source = None
        gc.collect()  # Force garbage collection
        logger.info("Cleared param_combinations_source from memory")

        # Strategy-specific parameter validation
        validated_combinations_tuples = []
        if hasattr(Config, 'parameter_validation_function') and callable(Config.parameter_validation_function):
            logger.info(f"Applying strategy-specific parameter validation: {Config.parameter_validation_function.__name__}")
            
            iterated_param_names = [p_name for p_name in Config.unique_params if p_name != 'pair']
            
            for comb_tuple in initial_combinations_from_config:
                if len(iterated_param_names) == len(comb_tuple):
                    params_dict = dict(zip(iterated_param_names, comb_tuple))
                    if Config.parameter_validation_function(params_dict):
                        validated_combinations_tuples.append(comb_tuple)

            # FIX: Use the correct variable 'validated_combinations_tuples' and store its length on Config
            Config.total_validated_combinations = len(validated_combinations_tuples)
            
            logger.info(f"Combinations after strategy-specific validation: {len(validated_combinations_tuples):,}")
            
            # MEMORY OPTIMIZATION: Clear initial_combinations_after validation
            initial_combinations_from_config = None
            gc.collect()  # Force garbage collection
            logger.info("Cleared initial_combinations_from_config from memory")
            
            if not validated_combinations_tuples:
                logger.warning("All parameter combinations failed strategy-specific validation.")
                return []
        else:
            logger.info("No strategy-specific parameter validation function found in Config. Using all combinations.")
            validated_combinations_tuples = initial_combinations_from_config
            # ALSO ADD IT HERE: Store the count for the 'else' case as well
            Config.total_validated_combinations = len(validated_combinations_tuples)
            initial_combinations_from_config = None  # Clear reference
        
        if not validated_combinations_tuples:
            logger.warning("No combinations available after validation (or initial list was empty).")
            return []

        available_params_tuples = validated_combinations_tuples
        
        # Checkpoint processing (existing code unchanged)
        checkpoint_files = glob.glob(os.path.join(Config.checkpoints_path, '*_checkpoint.pkl'))
        logger.info(f"Found {len(checkpoint_files)} checkpoint files in {Config.checkpoints_path}")
        
        processed_params_set: Set[tuple] = set()
        iterated_param_names_for_checkpoint = [p_name for p_name in Config.unique_params if p_name != 'pair']

        for file in checkpoint_files:
            try:
                checkpoint_data = Engine.load_checkpoint(file, Config) 
                if 'results' in checkpoint_data and checkpoint_data['results']:
                    for result_dict_item in checkpoint_data['results']:
                        if all(p_name in result_dict_item for p_name in iterated_param_names_for_checkpoint):
                            param_tuple = tuple(result_dict_item[p_name] for p_name in iterated_param_names_for_checkpoint)
                            processed_params_set.add(param_tuple)
            except Exception as e:
                logger.error(f"Error processing checkpoint file {file}: {str(e)}")
                continue
        
        available_params_set = set(available_params_tuples)
        valid_processed_params_set = processed_params_set.intersection(available_params_set)
        unprocessed_params_list = list(available_params_set - valid_processed_params_set)
        
        # MEMORY OPTIMIZATION: Clear large sets after use
        available_params_tuples = None
        available_params_set = None
        processed_params_set = None
        gc.collect()  # Force garbage collection
        logger.info("Cleared large parameter sets from memory")
        
        logger.info(f"Found {len(valid_processed_params_set):,} previously processed unique combinations from checkpoints.")
        logger.info(f"Found {len(unprocessed_params_list):,} unprocessed unique parameter combinations.")
        
        # Sampling logic
        total_needed = Config.iterations_sample_size
        final_sampled_list = []
        
        if valid_processed_params_set:
            processed_list_for_sampling = list(valid_processed_params_set)
            num_processed_to_keep = min(len(processed_list_for_sampling), total_needed)
            if num_processed_to_keep > 0:
                chosen_indices_proc = np.random.choice(len(processed_list_for_sampling), size=num_processed_to_keep, replace=False)
                final_sampled_list.extend([processed_list_for_sampling[i] for i in chosen_indices_proc])

        remaining_needed = total_needed - len(final_sampled_list)
        if remaining_needed > 0 and unprocessed_params_list:
            num_unprocessed_to_add = min(remaining_needed, len(unprocessed_params_list))
            if num_unprocessed_to_add > 0:
                chosen_indices_unproc = np.random.choice(len(unprocessed_params_list), size=num_unprocessed_to_add, replace=False)
                final_sampled_list.extend([unprocessed_params_list[i] for i in chosen_indices_unproc])
        
        # CRITICAL MEMORY OPTIMIZATION: Clear all large intermediate data structures
        valid_processed_params_set = None
        unprocessed_params_list = None
        gc.collect()  # Force garbage collection
        
        logger.info(f"Final sampled combinations count: {len(final_sampled_list):,}")
        logger.info(f"Target sample size was: {Config.iterations_sample_size}")
        logger.info(f"Memory cleanup completed - only sampled combinations retained")
        logger.info(f"Time to generate sampled_combinations: {time.time() - start_time:.2f} seconds")
        
        if len(final_sampled_list) < Config.iterations_sample_size:
             logger.warning(
                f"Could only generate {len(final_sampled_list)} unique combinations, "
                f"less than requested {Config.iterations_sample_size}. "
                f"This might be due to all validated combinations being processed or limited unique validated combinations."
            )
        
        return [list(item) for item in final_sampled_list]
        
    except Exception as e:
        logger_ref = Config.logger if hasattr(Config, 'logger') and Config.logger else logging.getLogger()
        logger_ref.error(f"Error in generate_sampled_combinations: {str(e)}\n{traceback.format_exc()}")
        return []

def generate_wfa_periods(start_year: int, end_year: int, in_sample_years: int, 
                        out_sample_years: int, method: str = 'loddy') -> List[Dict]:
    """
    Generate the in-sample and out-of-sample periods for walk-forward analysis.
    Years are stored as integers for efficiency.
    """
    try:
        current_year = start_year
        periods = []

        while current_year + in_sample_years + out_sample_years <= end_year:
            # In-sample period
            in_sample_start = current_year
            in_sample_end = current_year + in_sample_years - 1

            # Out-of-sample period
            out_sample_start = in_sample_end + 1
            out_sample_end = out_sample_start + out_sample_years - 1

            periods.append({
                'in_sample': (in_sample_start, in_sample_end),
                'out_sample': (out_sample_start, out_sample_end)
            })

            # Move to the next in-sample start year
            if method.lower() == 'loddy':
                current_year = out_sample_start
            elif method.lower() == 'original':
                current_year += out_sample_years
            else:
                raise ValueError("Invalid method. Choose 'loddy' or 'original'.")

        # Check if there's room for one more period (only for Loddy's method)
        if method.lower() == 'loddy' and current_year + in_sample_years <= end_year:
            in_sample_start = current_year
            in_sample_end = current_year + in_sample_years - 1
            out_sample_start = in_sample_end + 1
            out_sample_end = min(out_sample_start + out_sample_years - 1, end_year)

            periods.append({
                'in_sample': (in_sample_start, in_sample_end),
                'out_sample': (out_sample_start, out_sample_end)
            })

        return periods
    except Exception as e:
        print(f"Error generating WFA periods: {str(e)}")
        print(traceback.format_exc())
        return []

# ADD this function for memory monitoring
def log_memory_usage():
    """Log current memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    return f"Memory usage: {memory_mb:.1f} MB"

def walk_forward_analysis(Config, start_year: int, end_year: int, in_sample_years: int, out_sample_years: int) -> None:
    try:
        Config = get_config()  # Ensure Config is properly initialized
        base_path = getattr(Config, 'base_path', "../backtested_data")
        default_method = 'Original' if 'OG' in Config.Strategy_name else 'Loddy'
        walk_forward_method = getattr(Config, 'walk_forward_method', default_method)
        periods = generate_wfa_periods(start_year, end_year, in_sample_years, out_sample_years, method=walk_forward_method)
        folder_structure, strategy_path = create_wfa_folder_structure(Config.Strategy_name, base_path, periods)
        
        # Conditional Logger Initialization
        if getattr(Config, 'DEBUG_LOGGING', False):
            log_level_str = Config.log_config.get('level', 'INFO').upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            # Use a sub-directory within strategy_path for LogWrapper logs
            wrapper_log_dir = os.path.join(strategy_path, "debug_logs")
            os.makedirs(wrapper_log_dir, exist_ok=True, mode=0o777) # Ensure directory is writable

            Config.logger = LogWrapper.get_instance(
                name=f"{Config.Strategy_name}_WFA_MP", # Differentiate name
                log_dir=wrapper_log_dir,
                level=log_level
            )
            Config.logger.info(f"Using LogWrapper (multiprocessing-safe) at level {log_level_str}.")
        else:
            # Use the existing simple logger setup (assuming cf.setup_logging)
            Config.logger = cf.setup_logging(strategy_path, f'{Config.Strategy_name}.log')
            # Basic check to ensure cf.setup_logging returned a logger
            if hasattr(Config.logger, 'info'):
                 Config.logger.info("Using standard cf.setup_logging (non-multiprocessing optimized).")
            else:
                # Fallback if cf.setup_logging is problematic
                print(f"Warning: cf.setup_logging for {Config.Strategy_name} did not return a valid logger. Using basic console logger.")
                Config.logger = logging.getLogger(f"{Config.Strategy_name}_WFA_Simple")
                # Ensure level is set if falling back
                log_level_str = Config.log_config.get('level', 'INFO').upper()
                log_level = getattr(logging, log_level_str, logging.INFO)
                Config.logger.setLevel(log_level)
                if not Config.logger.handlers: # Add a handler if none exist
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    Config.logger.addHandler(handler)
                Config.logger.info(f"Fell back to basic console logger at level {log_level_str}.")

        for i, period in enumerate(periods, 1):
            if i == 1:
                current_paths = folder_structure[f"period_{i}"]
                set_paths(Config, current_paths['in_sample'])
                break
                
        Config.sampled_combinations = generate_sampled_combinations(Config)
        total_combinations_count = getattr(Config, 'total_validated_combinations', 0)
        vs.visualize_parameter_distribution(Config, strategy_path)
        Config.logger.info("-" * 80)
        Config.logger.info(f"Walk-Forward Analysis started for {Config.Strategy_name} \n with {Config.No_of_processes} prallel processes and {len(Config.sampled_combinations)} combinations from a total of {total_combinations_count:,} combinations\n From Year {Config.start_year} to {Config.end_year}.")
        Config.logger.info(f"Using {len(Config.sampled_combinations):,} sampled combinations")
        Config.logger.info(f"Memory after sampling: {log_memory_usage()}")
        
        all_params = {}
        all_Top_results = []

        for i, period in enumerate(periods, 1):
            in_sample_start, in_sample_end = period['in_sample']
            out_sample_start, out_sample_end = period['out_sample']

            current_run = {
                'in_sample': f"IS-{str(in_sample_start)[-2:]}-{str(in_sample_end)[-2:]}",
                'out_sample': f"OS-{str(out_sample_start)[-2:]}-{str(out_sample_end)[-2:]}"
            }

            Config.logger.info("*"*75)
            Config.logger.info(f"Starting in-sample period {current_run['in_sample']} ({i}/{len(periods)})")
            current_paths = folder_structure[f"period_{i}"]

            try:
                period_params, Top_results = run_wfa_period(Config, current_run, current_paths, in_sample_start, in_sample_end, out_sample_start, out_sample_end)
                
                if period_params:
                    all_params[f"period_{i}"] = period_params
                if Top_results:
                    # Add current_run information to Top_results - vectorized operation
                    for result in Top_results:
                        result['in_sample_period'] = current_run['in_sample']
                        result['out_sample_period'] = current_run['out_sample']
                        result['period_number'] = i
                    all_Top_results.extend(Top_results)
            except Exception as e:
                Config.logger.error(f"Error in period {i}: {str(e)}")
                Config.logger.error(traceback.format_exc())
                continue  # Skip to next period on error
            finally:
                Config.logger.info(f"Completed ({i}/{len(periods)}) periods. {cf.log_memory_usage()}\n")

        if all_params:
            save_results(Config, strategy_path, all_params)
        if all_Top_results:
            _ = analyze_parameter_frequency(Config, strategy_path, all_Top_results)
            Config.logger.info("Parameter frequency analysis completed")
        else:
            Config.logger.warning("No valid results found across all periods")

        Config.logger.info("Walk-Forward Analysis completed successfully.\n\n\n")
    except Exception as e:
        Config.logger.error(f"Error in walk_forward_analysis: {str(e)}")
        Config.logger.error(traceback.format_exc())
    
def run_wfa_period(Config, current_run, current_paths, in_sample_start, in_sample_end, out_sample_start, out_sample_end):
    try:
        period_params = {}
        Top_results = []

        # Run in-sample period
        in_sample_results = run_in_sample(Config, current_run, current_paths, in_sample_start, in_sample_end)
        
        if in_sample_results is None or not in_sample_results:
            Config.logger.warning(f"No valid in-sample results for period {current_run['in_sample']}")
            Config.logger.warning("Consider lowering filter criteria or checking data quality")
            return {}, []  # Skip to next period

        valid_filters = False  # Flag to track if any filter produced results
        
        for filter_set, filter_data in in_sample_results.items():
            if filter_set == "No_Filter":
                Config.logger.warning(f"All filters failed for period {current_run['in_sample']} - skipping to next period")
                Config.logger.warning("Consider adjusting filter criteria to be less stringent")
                return {}, []  # Skip to next period
                
            try:
                if not filter_data:
                    Config.logger.warning(f"Empty filter data for {filter_set} - skipping filter")
                    continue
                
                # Extract filter number for out-sample processing
                filter_number = filter_set.split('_')[-1]
                corresponding_os_filter = f"OS_Filter_{filter_number}"
                
                OS_best_params, out_sample_results = run_out_of_sample(
                    Config, current_run, current_paths,
                    out_sample_start, out_sample_end,
                    corresponding_os_filter, filter_data
                )

                if OS_best_params is None or (isinstance(out_sample_results, pd.DataFrame) and out_sample_results.empty):
                    Config.logger.warning(f"No valid results for filter set {filter_set} - trying next filter")
                    continue

                valid_filters = True  # At least one filter produced results
                
                # Process valid results
                period_params[current_run['in_sample']] = period_params.get(current_run['in_sample'], {})
                period_params[current_run['in_sample']][filter_set] = {
                    "period_type": "In_Sample",
                    "params": filter_data
                }

                if isinstance(out_sample_results, pd.DataFrame) and not out_sample_results.empty:
                    # Add filter information directly without loops
                    out_sample_results['IS_Filter'] = filter_set
                    out_sample_results['OS_Filter'] = corresponding_os_filter
                    Top_results.append(out_sample_results)

                for os_filter_set, os_params in OS_best_params.items():
                    period_params[current_run['out_sample']] = period_params.get(current_run['out_sample'], {})
                    period_params[current_run['out_sample']][os_filter_set] = {
                        "period_type": "Out_Sample",
                        "params": os_params
                    }

            except Exception as e:
                Config.logger.error(f"Error processing filter set {filter_set}: {str(e)}")
                Config.logger.error(traceback.format_exc())
                continue

        if not valid_filters:
            Config.logger.warning(f"No valid results from any filter for period {current_run['in_sample']}")
            Config.logger.warning("Consider adjusting filter criteria to be less stringent")
            return {}, []  # Skip to next period

        return period_params, Top_results
    except Exception as e:
        Config.logger.error(f"Error in run_wfa_period: {str(e)}")
        Config.logger.error(traceback.format_exc())
        return {}, []

def run_in_sample(Config, current_run, current_paths, start_year, end_year):
    try:
        Config.start_year, Config.end_year = start_year, end_year
        Config.In_Sample_Run = True
        Config.current_period = current_run['in_sample']
        set_paths(Config, current_paths['in_sample'])

        Config.logger.info("Running all pairs") if Config.run_all_pairs_in_sample else Config.logger.info(f"IS_our_curr: {Config.IS_our_curr}")

        results = Engine.run_Strategy(Config)
        
        if results is None:
            return None
        
        try:
            IS_unique_values, _ = Analyzer.analyze_allresults(Config)
            
            if not isinstance(IS_unique_values, dict) or not IS_unique_values:
                Config.logger.warning(f"WFA No valid best parameters found for in-sample period {Config.current_period}")
                return None
            return IS_unique_values
        
        except Exception as e:
            Config.logger.error(f"Error in run_in_sample analysis: {str(e)}")
            Config.logger.error(traceback.format_exc())
            return None
    except Exception as e:
        Config.logger.error(f"Error in run_in_sample: {str(e)}")
        Config.logger.error(traceback.format_exc())
        return None

def run_out_of_sample(Config, current_run, current_paths, start_year, end_year, filter_set, filter_data):
    try:
        Config.start_year, Config.end_year = start_year, end_year
        Config.In_Sample_Run = False
        Config.current_period = current_run['out_sample']
        Config.current_filter_set = filter_set
        set_paths(Config, current_paths['out_sample'], filter_set)
        Config.logger.info("*"*35)
        Config.logger.info(f"Starting out-sample period {Config.current_period} - {filter_set}")

        out_sample_combinations = prepare_out_sample_combinations(Config, filter_data)
        all_results = []
        period_params = {}

        # Collect all pairs and their specific parameter combinations
        pairs_with_combinations = {pair: param_combinations for pair, param_combinations in out_sample_combinations.items()}

        # Update Config to include specific parameter combinations for each pair
        Config.pairs_with_combinations = pairs_with_combinations

        # Run all pairs in parallel
        results = Engine.run_Strategy(Config)
        
        if results is None:
            return None, pd.DataFrame()

        try:
            OS_best_params, OS_top_n_df = Analyzer.analyze_allresults(Config)
            if OS_best_params is not None and not OS_top_n_df.empty:
                all_results.append(OS_top_n_df)
                for os_filter_set, os_params in OS_best_params.items():
                    period_params[Config.current_period] = period_params.get(Config.current_period, {})
                    period_params[Config.current_period][os_filter_set] = {
                        "period_type": "Out_Sample",
                        "params": os_params
                    }

            # Use pd.concat with specified dtypes to optimize memory
            if all_results:
                return period_params, pd.concat(all_results, ignore_index=True)
            return period_params, pd.DataFrame()
        
        except Exception as e:
            Config.logger.error(f"Error in out-sample analysis: {str(e)}")
            Config.logger.error(traceback.format_exc())
            return None, pd.DataFrame()
    except Exception as e:
        Config.logger.error(f"Error in run_out_of_sample: {str(e)}")
        Config.logger.error(traceback.format_exc())
        return None, pd.DataFrame()

def set_paths(Config, base_path, filter_set=None):
    try:
        if filter_set:
            base_path = os.path.join(base_path, filter_set)
        Config.checkpoints_path = os.path.join(base_path, 'checkpoints')
        Config.final_results_path = os.path.join(base_path, 'final_results')
        Config.trade_res_path = os.path.join(base_path, 'trades')
        os.makedirs(Config.checkpoints_path, exist_ok=True)
        os.makedirs(Config.final_results_path, exist_ok=True)
        os.makedirs(Config.trade_res_path, exist_ok=True)
    except Exception as e:
        Config.logger.error(f"Error setting paths: {str(e)}")
        Config.logger.error(traceback.format_exc())

def prepare_out_sample_combinations(Config, filter_data):
    try:
        out_sample_combinations = {}  # Use a dictionary to store combinations for each pair
        
        # Vectorize the parameter extraction for efficiency
        for pair, param_list in filter_data.items():
            # Pre-allocate a list of the right size
            pair_combinations = []
            
            # Extract parameters in a single step if possible
            if param_list:
                # Create parameter tuples directly
                for param_set in param_list:
                    param_tuple = tuple(param_set[param] for param in Config.unique_params if param != 'pair')
                    pair_combinations.append(param_tuple)
                    
            out_sample_combinations[pair] = pair_combinations
        
        Config.logger.info(f"Currencies for out-sample: {list(filter_data.keys())}")
        
        return out_sample_combinations
    except Exception as e:
        Config.logger.error(f"Error preparing out sample combinations: {str(e)}")
        Config.logger.error(traceback.format_exc())
        return {}
