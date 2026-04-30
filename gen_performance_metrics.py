'''Developed on V2 for: 1. combining the AllMetrics class together to receive the columns as needed
2. Getting % values in decimals and then updating excel to show them as %
3. Separated the number and (%) format
4. removed un necessary metrics
5. made names more precise
6. corrected some calculations
7. Added Score based Acc size
8. Added Normalized prices with $100 start for Score and Dollar based values
9. Removed Trade Risk and Initial deposit columns from dataframe
10.

'''

import numpy as np

class UnaffectedMetrics:
    def __init__(self, df_T1):
        try:
            self.df_T1 = df_T1
            
            # Count metrics - use int for counts
            self.total_trades = df_T1.shape[0]
            self.total_short_trades = df_T1[df_T1.SIGNAL == -1].shape[0]
            self.total_long_trades = df_T1[df_T1.SIGNAL == 1].shape[0]
            self.No_of_winning_Trades = df_T1[df_T1.Dv_QC > 0].shape[0]
            self.No_of_losing_Trades = df_T1[df_T1.Dv_QC < 0].shape[0]
            
            # Win/loss counts by direction
            try:
                self.No_of_winning_long_trades = df_T1[(df_T1['SIGNAL'] == 1) & (df_T1['Dv_QC'] > 0)].shape[0]
                self.No_of_winning_short_trades = df_T1[(df_T1['SIGNAL'] == -1) & (df_T1['Dv_QC'] > 0)].shape[0]
                self.No_of_losing_long_trades = df_T1[(df_T1['SIGNAL'] == 1) & (df_T1['Dv_QC'] <= 0)].shape[0]
                self.No_of_losing_short_trades = df_T1[(df_T1['SIGNAL'] == -1) & (df_T1['Dv_QC'] <= 0)].shape[0]
            except Exception as e:
                print(f"[UnaffectedMetrics] Error calculating win/loss counts: {e}")
                self.No_of_winning_long_trades = 0
                self.No_of_winning_short_trades = 0
                self.No_of_losing_long_trades = 0
                self.No_of_losing_short_trades = 0
            
            # Percentages - use float32 for all percentages
            try:
                self.short_trades_pct = np.float32(self.total_short_trades / self.total_trades) if self.total_trades > 0 else np.float32(0)
                self.long_trades_pct = np.float32(self.total_long_trades / self.total_trades) if self.total_trades > 0 else np.float32(0)
                self.win_pct = np.float32(self.No_of_winning_Trades / self.total_trades) if self.total_trades > 0 else np.float32(0)
                self.loss_pct = np.float32(self.No_of_losing_Trades / self.total_trades) if self.total_trades > 0 else np.float32(0)
                self.winning_long_trades_pct = np.float32(self.No_of_winning_long_trades / self.total_long_trades) if self.total_long_trades > 0 else np.float32(0)
                self.winning_short_trades_pct = np.float32(self.No_of_winning_short_trades / self.total_short_trades) if self.total_short_trades > 0 else np.float32(0)
                self.losing_long_trades_pct = np.float32(self.No_of_losing_long_trades / self.total_long_trades) if self.total_long_trades > 0 else np.float32(0)
                self.losing_short_trades_pct = np.float32(self.No_of_losing_short_trades / self.total_short_trades) if self.total_short_trades > 0 else np.float32(0)
            except Exception as e:
                print(f"[UnaffectedMetrics] Error calculating percentage metrics: {e}")
                # Set defaults
                self.short_trades_pct = np.float32(0)
                self.long_trades_pct = np.float32(0)
                self.win_pct = np.float32(0)
                self.loss_pct = np.float32(0)
                self.winning_long_trades_pct = np.float32(0)
                self.winning_short_trades_pct = np.float32(0)
                self.losing_long_trades_pct = np.float32(0) 
                self.losing_short_trades_pct = np.float32(0)
            
            # Largest and average trade values - use float32
            try:
                wins_df = df_T1[df_T1['Dv_QC'] > 0]
                losses_df = df_T1[df_T1['Dv_QC'] < 0]
                
                self.largest_profitable_trade = np.float32(wins_df['Dv_QC'].max()) if not wins_df.empty else np.float32(0)
                self.largest_loss_trade = np.float32(losses_df['Dv_QC'].min()) if not losses_df.empty else np.float32(0)
                self.avg_profit_trade = np.float32(wins_df['Dv_QC'].mean()) if not wins_df.empty else np.float32(0)
                self.avg_loss_trade = np.float32(losses_df['Dv_QC'].mean()) if not losses_df.empty else np.float32(0)
            except Exception as e:
                print(f"[UnaffectedMetrics] Error calculating trade value metrics: {e}")
                self.largest_profitable_trade = np.float32(0)
                self.largest_loss_trade = np.float32(0)
                self.avg_profit_trade = np.float32(0)
                self.avg_loss_trade = np.float32(0)
            
            # Calculate win/loss streaks
            try:
                # Create binary columns for wins and losses as int8 (smaller than bool)
                self.df_T1['Win'] = (self.df_T1['Dv_QC'] > 0).astype(np.int8)
                self.df_T1['Loss'] = (self.df_T1['Dv_QC'] < 0).astype(np.int8)
                
                # Calculate streaks
                win_cumsum = self.df_T1['Win'].cumsum()
                win_reset = win_cumsum - win_cumsum.where(~self.df_T1['Win'].astype(bool)).ffill().fillna(0)
                
                loss_cumsum = self.df_T1['Loss'].cumsum()
                loss_reset = loss_cumsum - loss_cumsum.where(~self.df_T1['Loss'].astype(bool)).ffill().fillna(0)
                
                # Store results as float32
                self.max_consecutive_wins = np.float32(win_reset.max())
                self.max_consecutive_losses = np.float32(loss_reset.max())
                self.avg_consecutive_wins = np.float32(win_reset[win_reset > 0].mean()) if (win_reset > 0).any() else np.float32(0)
                self.avg_consecutive_losses = np.float32(loss_reset[loss_reset > 0].mean()) if (loss_reset > 0).any() else np.float32(0)
                
                # Drop intermediate columns immediately in batch
                self.df_T1.drop(['Win', 'Loss'], axis=1, inplace=True)
                
            except Exception as e:
                print(f"[UnaffectedMetrics] Error calculating streak metrics: {e}")
                self.max_consecutive_wins = np.float32(0)
                self.max_consecutive_losses = np.float32(0)
                self.avg_consecutive_wins = np.float32(0)
                self.avg_consecutive_losses = np.float32(0)
            
            # Expected payoff calculation as an attribute
            try:
                self.expected_payoff = np.float32(
                    ((self.win_pct * self.avg_profit_trade) + (self.loss_pct * self.avg_loss_trade)) / 100
                )
            except Exception as e:
                print(f"[UnaffectedMetrics] Error calculating expected payoff: {e}")
                self.expected_payoff = np.float32(0)
                
        except Exception as e:
            print(f"[UnaffectedMetrics] Unhandled error in initialization: {e}")
            # Set default values for all attributes
            self.total_trades = 0
            self.win_pct = np.float32(0)
            self.loss_pct = np.float32(0)
            # ...other defaults as needed

class ScoreAndPipsMetrics:
    def __init__(self, df_T1, Trade_risk, Initial_deposit, logger):
        try:
            self.logger = logger
            self.df_T1 = df_T1
            self.Trade_risk = np.float32(Trade_risk)
            self.Initial_deposit = np.float32(Initial_deposit)

            # Initialize masks with default values to avoid 'unbound' errors
            win_mask = self.df_T1['score'] > 0
            loss_mask = self.df_T1['score'] <= 0
            long_mask = self.df_T1['SIGNAL'] == 1
            short_mask = self.df_T1['SIGNAL'] == -1
            pips_win_mask = self.df_T1['pips_gained'] > 0
            pips_loss_mask = self.df_T1['pips_gained'] <= 0

            # Score-based calculations with float32
            try:
                self.df_T1['score_gains'] = (self.df_T1['score'] * self.Trade_risk).astype(np.float32)
                self.df_T1['SbCum_res'] = self.df_T1['score_gains'].cumsum().astype(np.float32)
                self.df_T1["SbAcc_Sz"] = (self.Initial_deposit + self.df_T1["SbCum_res"]).astype(np.float32)
                normalization_factor = np.float32(100.0 / self.Initial_deposit)
                self.df_T1["SbAcc_Normalised"] = self.df_T1["SbAcc_Sz"].mul(normalization_factor).astype(np.float32)
            except Exception as e:
                print(f"[ScoreAndPipsMetrics] Error in score-based calculations: {e}")
                if logger: logger.error(f"Error in score-based calculations: {e}", exc_info=True)

            # Profit/loss calculations
            try:
                # Create masks for different conditions to avoid repetitive calculations
                win_mask = self.df_T1['score'] > 0
                loss_mask = self.df_T1['score'] <= 0
                long_mask = self.df_T1['SIGNAL'] == 1
                short_mask = self.df_T1['SIGNAL'] == -1
                
                # Gross profit/loss calculations
                self.Gross_loss_Score_based = np.float32((df_T1[loss_mask]['score'].sum()) * self.Trade_risk)
                self.Gross_profit_Score_based = np.float32((df_T1[win_mask]['score'].sum()) * self.Trade_risk)
                self.Net_profit_Score_based = np.float32(self.Gross_profit_Score_based + self.Gross_loss_Score_based)
                self.Ending_Acc_Value_Score_based = np.float32(Initial_deposit + self.Net_profit_Score_based)

                # Score calculations
                self.Net_score = np.float32(df_T1.score.sum())
                self.Net_Long_Score = np.float32(df_T1[long_mask].score.sum())
                self.Net_Short_Score = np.float32(df_T1[short_mask].score.sum())

                # Profit calculations with risk
                self.short_profit_with_n_risk = np.float32(self.Net_Short_Score * self.Trade_risk)
                self.long_profit_with_n_risk = np.float32(self.Net_Long_Score * self.Trade_risk)

                # Loss calculations with risk
                self.short_loss_with_n_risk = np.float32((self.df_T1[short_mask & loss_mask]['score'].sum()) * self.Trade_risk)
                self.long_loss_with_n_risk = np.float32((self.df_T1[long_mask & loss_mask]['score'].sum()) * self.Trade_risk)
            except Exception as e:
                print(f"[ScoreAndPipsMetrics] Error in profit/loss calculations: {e}")
                if logger: logger.error(f"Error in profit/loss calculations: {e}", exc_info=True)
                # Set safe defaults
                self.Gross_loss_Score_based = np.float32(0)
                self.Gross_profit_Score_based = np.float32(0)
                self.Net_profit_Score_based = np.float32(0)
                self.Ending_Acc_Value_Score_based = self.Initial_deposit
                self.Net_score = np.float32(0)
                self.Net_Long_Score = np.float32(0)
                self.Net_Short_Score = np.float32(0)
                self.short_profit_with_n_risk = np.float32(0)
                self.long_profit_with_n_risk = np.float32(0)
                self.short_loss_with_n_risk = np.float32(0)
                self.long_loss_with_n_risk = np.float32(0)

            # Pips calculations
            try:
                # Create conditions for win/loss on pips
                pips_win_mask = self.df_T1['pips_gained'] > 0
                pips_loss_mask = self.df_T1['pips_gained'] <= 0
                
                # Calculate pips metrics
                self.Net_pips_gained = np.float32(self.df_T1.pips_gained.sum())
                self.long_pips_gained = np.float32(self.df_T1[long_mask].pips_gained.sum())
                self.short_pips_gained = np.float32(self.df_T1[short_mask].pips_gained.sum())
                self.gross_pips_won = np.float32(self.df_T1[pips_win_mask]['pips_gained'].sum())
                self.gross_pips_lost = np.float32(self.df_T1[pips_loss_mask]['pips_gained'].sum())
                self.pips_lost_from_short_trades = np.float32(self.df_T1[short_mask & pips_loss_mask]['pips_gained'].sum())
                self.pips_lost_from_long_trades = np.float32(self.df_T1[long_mask & pips_loss_mask]['pips_gained'].sum())
            except Exception as e:
                print(f"[ScoreAndPipsMetrics] Error in pips calculations: {e}")
                if logger: logger.error(f"Error in pips calculations: {e}", exc_info=True)
                # Set safe defaults for pips metrics
                self.Net_pips_gained = np.float32(0)
                self.long_pips_gained = np.float32(0)
                self.short_pips_gained = np.float32(0)
                self.gross_pips_won = np.float32(0)
                self.gross_pips_lost = np.float32(0)
                self.pips_lost_from_short_trades = np.float32(0)
                self.pips_lost_from_long_trades = np.float32(0)

            # Additional metrics
            try:
                self.profit_factor_Score_based = (
                    np.float32(abs(self.Gross_profit_Score_based / self.Gross_loss_Score_based))
                    if self.Gross_loss_Score_based != 0 else np.float32(float('inf'))
                )
            except Exception as e:
                print(f"[ScoreAndPipsMetrics] Error calculating profit factor: {e}")
                if logger: logger.error(f"Error calculating profit factor: {e}", exc_info=True)
                self.profit_factor_Score_based = np.float32(0)

            # Clean up any intermediate columns that are not needed
            # (In this case, we keep all columns since they might be needed for Excel export)
                
        except Exception as e:
            print(f"[ScoreAndPipsMetrics] Unhandled error in initialization: {e}")
            if logger: logger.error(f"Unhandled error in initialization: {e}", exc_info=True)
            # Set safe defaults for critical attributes
            self.Net_profit_Score_based = np.float32(0)
            self.Ending_Acc_Value_Score_based = self.Initial_deposit
    
    def calculate_score_based_cagr(self, start_year, end_year):
        try:
            if self.Ending_Acc_Value_Score_based > 0 and self.Initial_deposit > 0:
                number_of_years = np.float32(end_year - start_year + 1)
                cagr = np.float32((((self.Ending_Acc_Value_Score_based / self.Initial_deposit) ** (1 / number_of_years)) - 1) * 100)
                return cagr
            else:
                return np.nan
        except Exception as e:
            print(f"[ScoreAndPipsMetrics.calculate_score_based_cagr] Error: {e}")
            if self.logger: self.logger.error(f"Error in calculating Sb_CAGR: {e}", exc_info=True)
            return np.nan
    
    @staticmethod
    def calculate_max_drawdown(df, column_name):
        try:
            cumulative = df[column_name].cumsum()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max).astype(np.float32)
            max_drawdown_value = np.float32(drawdown.min())
            
            # Safely handle potential division by zero
            if cumulative.idxmin() in running_max and running_max[cumulative.idxmin()] != 0:
                max_drawdown_pct = np.float32(max_drawdown_value / running_max[cumulative.idxmin()])
            else:
                max_drawdown_pct = np.float32(0)
                
            return max_drawdown_value, max_drawdown_pct
        except Exception as e:
            print(f"[ScoreAndPipsMetrics.calculate_max_drawdown] Error: {e}")
            return np.float32(0), np.float32(0)


class DollarValueMetrics:
    def __init__(self, df_T1, Initial_deposit):
        try:
            self.df_T1 = df_T1
            self.Initial_deposit = np.float32(Initial_deposit)
            
            try:
                # Use float32 throughout for memory efficiency
                win_mask = self.df_T1['Dv_QC'] > 0
                loss_mask = self.df_T1['Dv_QC'] <= 0
                
                self.Gross_loss_Dvbased = np.float32(self.df_T1[loss_mask]['Dv_QC'].sum())
                self.Gross_profit_Dvbased = np.float32(self.df_T1[win_mask]['Dv_QC'].sum())
                self.Net_profit_Dvbased = np.float32(self.Gross_profit_Dvbased + self.Gross_loss_Dvbased)
                self.Ending_Acc_Value_Dvbased = np.float32(self.Initial_deposit + self.Net_profit_Dvbased)
            except Exception as e:
                print(f"[DollarValueMetrics] Error calculating profit/loss metrics: {e}")
                # Set safe defaults
                self.Gross_loss_Dvbased = np.float32(0)
                self.Gross_profit_Dvbased = np.float32(0)
                self.Net_profit_Dvbased = np.float32(0)
                self.Ending_Acc_Value_Dvbased = self.Initial_deposit
            
            try:
                # Calculate profit factor with safety check for division by zero
                if self.Gross_loss_Dvbased != 0:
                    self.profit_factor_Dvbased = np.float32(abs(self.Gross_profit_Dvbased / self.Gross_loss_Dvbased))
                else:
                    self.profit_factor_Dvbased = np.float32(float('inf'))
            except Exception as e:
                print(f"[DollarValueMetrics] Error calculating profit factor: {e}")
                self.profit_factor_Dvbased = np.float32(0)
        except Exception as e:
            print(f"[DollarValueMetrics] Unhandled error in initialization: {e}")
            # Set safe defaults for all attributes
            self.Gross_loss_Dvbased = np.float32(0)
            self.Gross_profit_Dvbased = np.float32(0) 
            self.Net_profit_Dvbased = np.float32(0)
            self.Ending_Acc_Value_Dvbased = np.float32(Initial_deposit)
            self.profit_factor_Dvbased = np.float32(0)
    
    
class AccountMetrics:
    def __init__(self, df_T1, Initial_deposit, logger):
        try:
            self.logger = logger
            self.df_T1 = df_T1
            self.Initial_deposit = np.float32(Initial_deposit)
            
            # Add columns to df_T1 using float32 for memory efficiency
            try:
                self.df_T1["DvCum_res"] = self.df_T1['Dv_QC'].cumsum().astype(np.float32)
            except Exception as e:
                print(f"[AccountMetrics] Error adding DvCum_res: {e}")
                if logger: logger.error(f"Error adding DvCum_res: {e}", exc_info=True)
                
            try:
                self.df_T1["DvAcc_Sz"] = (self.Initial_deposit + self.df_T1["DvCum_res"]).astype(np.float32)
                self.df_T1["Acc_Sz_lag1"] = (self.Initial_deposit + self.df_T1["DvCum_res"].shift(1).fillna(0)).astype(np.float32)
                normalization_factor = np.float32(100.0 / self.Initial_deposit)
                self.df_T1["DvAcc_Normalised"] = self.df_T1["DvAcc_Sz"].mul(normalization_factor).astype(np.float32)
            except Exception as e:
                print(f"[AccountMetrics] Error calculating account sizes: {e}")
                if logger: logger.error(f"Error calculating account sizes: {e}", exc_info=True)

            # Calculate log returns
            try:
                valid_values = (self.df_T1["DvAcc_Sz"] > 0) & (self.df_T1["Acc_Sz_lag1"] > 0)
                self.df_T1["L_ret"] = np.nan
                self.df_T1.loc[valid_values, "L_ret"] = np.log(
                    self.df_T1.loc[valid_values, "DvAcc_Sz"] / self.df_T1.loc[valid_values, "Acc_Sz_lag1"]
                ).astype(np.float32)
            except Exception as e:
                print(f"[AccountMetrics] Error in log return calculation: {e}")
                if logger: logger.error(f"Error in log return calculation: {e}", exc_info=True)

            # Cumulative returns
            try:
                self.df_T1["Cum_L_ret"] = self.df_T1["L_ret"].cumsum().apply(np.exp).astype(np.float32)
                self.df_T1['Cum_Long'] = (self.df_T1['Dv_QC'] * (self.df_T1['SIGNAL'] == 1)).cumsum().astype(np.float32)
                self.df_T1['Cum_Short'] = (self.df_T1['Dv_QC'] * (self.df_T1['SIGNAL'] == -1)).cumsum().astype(np.float32)
            except Exception as e:
                print(f"[AccountMetrics] Error calculating cumulative values: {e}")
                if logger: logger.error(f"Error calculating cumulative values: {e}", exc_info=True)

            # Final account value metrics
            try:
                self.final_value_of_OneD = np.float32(self.df_T1["DvAcc_Sz"].iloc[-1] / self.Initial_deposit)
                self.final_account_value = np.float32(self.df_T1['DvAcc_Sz'].iloc[-1])
            except Exception as e:
                print(f"[AccountMetrics] Error calculating final values: {e}")
                if logger: logger.error(f"Error calculating final values: {e}", exc_info=True)
                self.final_value_of_OneD = np.float32(0)
                self.final_account_value = np.float32(0)

            # Growth percentage
            try:
                self.absolute_growth_pct = np.nan
                if self.final_account_value > 0 and self.Initial_deposit > 0:
                    self.absolute_growth_pct = np.float32(np.log(self.final_account_value / self.Initial_deposit))
                else:
                    self.absolute_growth_pct = np.nan_to_num(self.absolute_growth_pct, nan=np.nan)
            except Exception as e:
                print(f"[AccountMetrics] Error calculating growth percentage: {e}")
                if logger: logger.error(f"Error calculating growth percentage: {e}", exc_info=True)
                self.absolute_growth_pct = np.nan

            # Statistical metrics
            try:
                lret_std = self.df_T1['L_ret'].std()
                self.sharpe_ratio = np.nan
                if lret_std != 0:
                    self.sharpe_ratio = np.float32((self.df_T1['L_ret'].mean() / lret_std) * np.sqrt(252))
                self.sharpe_ratio = np.nan_to_num(self.sharpe_ratio, nan=np.nan)
                
                self.avg_ret_per_trade_pct = np.float32(self.df_T1['L_ret'].mean())
                self.std_dev_pct = np.float32(lret_std)
                
                self.avg_win_return_pct = np.float32(self.df_T1[self.df_T1['Dv_QC'] > 0]['L_ret'].mean())
                self.avg_loss_return_pct = np.float32(self.df_T1[self.df_T1['Dv_QC'] < 0]['L_ret'].mean())
                
                self.win_loss_ratio = np.float32(abs(self.avg_win_return_pct / self.avg_loss_return_pct) if self.avg_loss_return_pct != 0 else float('inf'))
            except Exception as e:
                print(f"[AccountMetrics] Error calculating statistical metrics: {e}")
                if logger: logger.error(f"Error calculating statistical metrics: {e}", exc_info=True)
                self.sharpe_ratio = np.nan
                self.avg_ret_per_trade_pct = np.nan
                self.std_dev_pct = np.nan
                self.avg_win_return_pct = np.nan
                self.avg_loss_return_pct = np.nan
                self.win_loss_ratio = np.nan

            # Drawdown calculation
            try:
                self.df_T1['Cumulative_Max'] = self.df_T1['DvAcc_Sz'].cummax().astype(np.float32)
                self.df_T1['Drawdown'] = (self.df_T1['DvAcc_Sz'] - self.df_T1['Cumulative_Max']).astype(np.float32)
                self.max_drawdown_Acc_Sz = np.float32(self.df_T1['Drawdown'].min())
                
                # Clean up intermediate columns
                self.df_T1.drop(['Cumulative_Max', 'Drawdown'], axis=1, inplace=True)
            except Exception as e:
                print(f"[AccountMetrics] Error in drawdown calculation: {e}")
                if logger: logger.error(f"Error in drawdown calculation: {e}", exc_info=True)
                self.max_drawdown_Acc_Sz = np.float32(0)
                
        except Exception as e:
            print(f"[AccountMetrics] Unhandled error in initialization: {e}")
            if logger: logger.error(f"Unhandled error in initialization: {e}", exc_info=True)
            raise
        
    def add_cumulative_columns(self, column_name):
        try:
            trade_results = self.df_T1[column_name].astype(np.float32)
            self.df_T1['Cum_Wins'] = (trade_results * (trade_results > 0)).cumsum().astype(np.float32)
            self.df_T1['Cum_Loss'] = (trade_results * (trade_results < 0)).abs().cumsum().astype(np.float32)
            return self.df_T1
        except Exception as e:
            print(f"[AccountMetrics.add_cumulative_columns] Error: {e}")
            if self.logger: self.logger.error(f"Error in add_cumulative_columns: {e}", exc_info=True)
            return self.df_T1
    
    def calculate_Dv_based_cagr(self, start_year, end_year):
        try:
            if self.final_account_value > 0 and self.Initial_deposit > 0:
                number_of_years = np.float32(end_year - start_year + 1)
                cagr = np.float32((((self.final_account_value / self.Initial_deposit) ** (1 / number_of_years)) - 1) * 100)
                return cagr
            else:
                return np.nan
        except Exception as e:
            print(f"[AccountMetrics.calculate_Dv_based_cagr] Error: {e}")
            if self.logger: self.logger.error(f"Error in calculate_Dv_based_cagr: {e}", exc_info=True)
            return np.nan


class AllMetrics:
    def __init__(self, df_T1, Trade_risk, Initial_deposit, start_year, end_year, logger):
        self.unaffected_metrics = UnaffectedMetrics(df_T1)
        self.account_metrics = AccountMetrics(df_T1, Initial_deposit, logger)
        self.score_and_pips_metrics = ScoreAndPipsMetrics(df_T1, Trade_risk, Initial_deposit, logger)
        self.dollar_value_metrics = DollarValueMetrics(df_T1, Initial_deposit)
        self.start_year = start_year
        self.end_year = end_year

    
    def get_metrics_dict(self):
        metrics_dict = {
        #----------------------------------------------- 'Table 1.1: Deciding Results' -----------------------------------------------#
        "Ending Account Value Dvbased": self.dollar_value_metrics.Ending_Acc_Value_Dvbased,
        "Ending Account Value Score Based": self.score_and_pips_metrics.Ending_Acc_Value_Score_based,
        "Net Profit Dv_QC": self.dollar_value_metrics.Net_profit_Dvbased,
        "Net Profit Score Based": self.score_and_pips_metrics.Net_profit_Score_based,
        "Net Pips Gained": self.score_and_pips_metrics.Net_pips_gained,
        "Total Trades": self.unaffected_metrics.total_trades,
        "Net Score": self.score_and_pips_metrics.Net_score,
        "Final value of 1 Dollar Invested": self.account_metrics.final_value_of_OneD,
        
        #--------------------------------------------- 'Table 1.2: Deciding Ratios' ------------------------------------------------#
        "Win Percentage": self.unaffected_metrics.win_pct,
        "Mo Win/Loss Ratio": self.account_metrics.win_loss_ratio,
        "Absolute Growth Percentage": self.account_metrics.absolute_growth_pct,
        "Max Drawdown Account Size": self.account_metrics.max_drawdown_Acc_Sz,
        "(mean) Average Return Per Trade Percentage": self.account_metrics.avg_ret_per_trade_pct,
        "(std) Standard Deviation Percentage": self.account_metrics.std_dev_pct,
        "Sharpe Ratio": self.account_metrics.sharpe_ratio,
        
        #---------------------------------------------- 'Table 1.3: IMP Metrics' ---------------------------------------------------#
        "Expected Payoff Percentage": self.unaffected_metrics.expected_payoff,
        "Profit Factor Score Based": self.score_and_pips_metrics.profit_factor_Score_based,
        "Profit Factor Dvbased": self.dollar_value_metrics.profit_factor_Dvbased,
        "CAGR Score Based": self.score_and_pips_metrics.calculate_score_based_cagr(self.start_year, self.end_year),
        "CAGR Dvbased": self.account_metrics.calculate_Dv_based_cagr(self.start_year, self.end_year),
        
        #-------------------------------------------- 'Un-affected Metrics' ---------------------------------------------------------#
        "Total Short Trades": self.unaffected_metrics.total_short_trades,
        "Short Trades Percentage": self.unaffected_metrics.short_trades_pct,
        "Total Long Trades": self.unaffected_metrics.total_long_trades,
        "Long Trades Percentage": self.unaffected_metrics.long_trades_pct,
        "Total Winning Trades": self.unaffected_metrics.No_of_winning_Trades,
        "Total Losing Trades": self.unaffected_metrics.No_of_losing_Trades,
        "Loss Percentage": self.unaffected_metrics.loss_pct,
        "Winning Long Trades": self.unaffected_metrics.No_of_winning_long_trades,
        "Winning Long Trades Percentage": self.unaffected_metrics.winning_long_trades_pct,
        "Winning Short Trades": self.unaffected_metrics.No_of_winning_short_trades,
        "Winning Short Trades Percentage": self.unaffected_metrics.winning_short_trades_pct,
        "Losing Long Trades": self.unaffected_metrics.No_of_losing_long_trades,
        "Losing Long Trades Percentage": self.unaffected_metrics.losing_long_trades_pct,
        "Losing Short Trades": self.unaffected_metrics.No_of_losing_short_trades,
        "Losing Short Trades Percentage": self.unaffected_metrics.losing_short_trades_pct,
        "Largest Profit Trade": self.unaffected_metrics.largest_profitable_trade,
        "Largest Loss Trade": self.unaffected_metrics.largest_loss_trade,
        "Average Profit Trade": self.unaffected_metrics.avg_profit_trade,
        "Average Loss Trade": self.unaffected_metrics.avg_loss_trade,
        "Max Consecutive Wins": self.unaffected_metrics.max_consecutive_wins,
        "Max Consecutive Losses": self.unaffected_metrics.max_consecutive_losses,
        "Average Consecutive Wins": self.unaffected_metrics.avg_consecutive_wins,
        "Average Consecutive Losses": self.unaffected_metrics.avg_consecutive_losses,
        "Average Win Return Percentage": self.account_metrics.avg_win_return_pct,
        "Average Loss Return Percentage": self.account_metrics.avg_loss_return_pct,
        
        #--------------------------------------------- 'Score Based Metrics' --------------------------------------------------------#
        "Short Profit Score Based": self.score_and_pips_metrics.short_profit_with_n_risk,
        "Short Loss Score Based": self.score_and_pips_metrics.short_loss_with_n_risk,
        "Net Short Score": self.score_and_pips_metrics.Net_Short_Score,
        "Long Profit Score Based": self.score_and_pips_metrics.long_profit_with_n_risk,
        "Long Loss Score Based": self.score_and_pips_metrics.long_loss_with_n_risk,
        "Net Long Score": self.score_and_pips_metrics.Net_Long_Score,
        "Gross Loss Score Based": self.score_and_pips_metrics.Gross_loss_Score_based,
        "Gross Profit Score Based": self.score_and_pips_metrics.Gross_profit_Score_based,
        
        #--------------------------------------------- 'Pips Based Metrics' ---------------------------------------------------------#
        "Long Pips Gained": self.score_and_pips_metrics.long_pips_gained,
        "Long Pips Lost": self.score_and_pips_metrics.pips_lost_from_long_trades,
        "Short Pips Gained": self.score_and_pips_metrics.short_pips_gained,
        "Short Pips Lost": self.score_and_pips_metrics.pips_lost_from_short_trades,
        "Gross Pips Won": self.score_and_pips_metrics.gross_pips_won,
        "Gross Pips Lost": self.score_and_pips_metrics.gross_pips_lost,
            
        }
        return metrics_dict
