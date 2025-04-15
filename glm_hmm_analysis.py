import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact
from photometry_analysis import calculate_sem, save_figure, process_session, plot_session_results, check_saved_pooled_results, save_pooled_results, analyze_pooled_data, load_filtered_behavior_data, ensure_directory_exists



########################################################################FOR PRESENTATION: Currently some legens disabled, enable them again before plotting. 

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global parameters
base_dir = "/Volumes/ogma/delab/lab-members/joanna/photometry/preprocess"
output_dir = "/Volumes/ogma/delab/lab-members/joschua/photometry_analysis"
sampling_rate = 120
pre_cue_time = 3
post_cue_time = 5
pre_cue_samples = int(pre_cue_time * sampling_rate)
post_cue_samples = int(post_cue_time * sampling_rate)
total_window_samples = pre_cue_samples + post_cue_samples

PARQUET_PATH = "/Volumes/ogma/delab/matchingpennies/matchingpennies_datatable.parquet"
CODE_VERSION = "1.1.1"  # Increment this when making analysis changes --> will force recomputation of all data
_SESSION_CACHE = {}

def analyze_behavioral_states(subjid="All", win_loss=False, specific_subjects=None, behavior_df=None, condition=None):
    """
    Analyze behavioral states from GLM-HMM analysis across trials.
    
    Parameters:
    -----------
    subjid : str
        Subject ID to analyze, or "All" for cross-subject analysis
    win_loss : bool, optional (default=False)
        If True, separate analysis by previous trial outcome (win vs loss)
    specific_subjects : list, optional
        List of subject IDs to include if subjid="All"
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    condition : str, optional (default=None)
        Condition to analyze: "current_trial" for win/loss, "previous_trial" for previous win/loss,
        or None for state only (ignores win_loss parameter if set to None)
        
    Returns:
    --------
    dict: Analysis results
    """
    if subjid == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Make sure behavior_df is loaded
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
        
        # Container for subject-level results
        subject_results = []
        
        # Process each subject individually
        for subject in specific_subjects:
            print(f"Processing subject {subject} for behavioral state analysis...")
            try:
                # Get individual subject results by calling original function
                subject_result = analyze_behavioral_states_single(subject, win_loss, behavior_df=behavior_df, condition=condition)
                if subject_result:
                    subject_results.append(subject_result)
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
        
        if not subject_results:
            print("No valid subjects found for cross-subject analysis")
            return None
            
        # Average across subjects
        if condition == "current_trial" and win_loss:
            # For win/loss analysis we need to collect data separately for win and loss trials
            stochastic_win_data = []
            stochastic_loss_data = []
            biased_win_data = []
            biased_loss_data = []
            
            time_axis = None
            
            # Extract data from each subject's results
            for result in subject_results:
                if time_axis is None and 'time_axis' in result:
                    time_axis = result['time_axis']
                
                # Extract win/loss data for each state
                if ('stochastic' in result and 
                    result['stochastic'] is not None and 
                    'win_avg' in result['stochastic'] and 
                    result['stochastic']['win_avg'] is not None):
                    stochastic_win_data.append(result['stochastic']['win_avg'])
                
                if ('stochastic' in result and 
                    result['stochastic'] is not None and 
                    'loss_avg' in result['stochastic'] and 
                    result['stochastic']['loss_avg'] is not None):
                    stochastic_loss_data.append(result['stochastic']['loss_avg'])
                    
                if ('biased' in result and 
                    result['biased'] is not None and 
                    'win_avg' in result['biased'] and 
                    result['biased']['win_avg'] is not None):
                    biased_win_data.append(result['biased']['win_avg'])
                    
                if ('biased' in result and 
                    result['biased'] is not None and 
                    'loss_avg' in result['biased'] and 
                    result['biased']['loss_avg'] is not None):
                    biased_loss_data.append(result['biased']['loss_avg'])
            
            if time_axis is None or (not stochastic_win_data and not stochastic_loss_data and 
                                    not biased_win_data and not biased_loss_data):
                print("No valid win/loss data found for cross-subject analysis")
                return None
                
            # Create the plot
            plt.figure(figsize=(12, 7))
            
            # Define colors and styles for win/loss conditions
            colors = {
                'stochastic_win': 'darkgreen',
                'stochastic_loss': 'lightgreen',
                'biased_win': 'purple',
                'biased_loss': 'plum'
            }
            
            linestyles = {
                'stochastic_win': '-',
                'stochastic_loss': '--',
                'biased_win': '-',
                'biased_loss': '--'
            }
            
            # Plot stochastic win data if available
            if stochastic_win_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(stochastic_win_data, axis=0)
                sem = np.std(stochastic_win_data, axis=0) / np.sqrt(len(stochastic_win_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['stochastic_win'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['stochastic_win'],
                        linestyle=linestyles['stochastic_win'],
                        linewidth=2,
                        label=f"Stochastic Win")
                        
            # Plot stochastic loss data if available
            if stochastic_loss_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(stochastic_loss_data, axis=0)
                sem = np.std(stochastic_loss_data, axis=0) / np.sqrt(len(stochastic_loss_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['stochastic_loss'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['stochastic_loss'],
                        linestyle=linestyles['stochastic_loss'],
                        linewidth=2,
                        label=f"Stochastic Loss")
                        
            # Plot biased win data if available
            if biased_win_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(biased_win_data, axis=0)
                sem = np.std(biased_win_data, axis=0) / np.sqrt(len(biased_win_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['biased_win'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['biased_win'],
                        linestyle=linestyles['biased_win'],
                        linewidth=2,
                        label=f"Biased Win")
                        
            # Plot biased loss data if available
            if biased_loss_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(biased_loss_data, axis=0)
                sem = np.std(biased_loss_data, axis=0) / np.sqrt(len(biased_loss_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['biased_loss'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['biased_loss'],
                        linestyle=linestyles['biased_loss'],
                        linewidth=2,
                        label=f"Biased Loss")
            
            # Add reference lines
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add labels and title
            plt.xlabel('Time (s) after first lick', fontsize=24)
            plt.ylabel('z-ΔF/F', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.title(f'Photometry by Behavioral State and Outcome', fontsize=14)
            plt.xlim([-pre_cue_time, post_cue_time])
            #plt.legend(loc='upper right', fontsize=24)
            plt.tight_layout()
            
            # Save the figure
            save_figure(plt.gcf(), "all_subjects", "pooled", "behavioral_states_winloss")
            
            plt.show()
            
            return {
                'subjid': 'All',
                'specific_subjects': specific_subjects,
                'win_loss': win_loss,
                'condition': condition,
                'n_subjects': len(subject_results),
                'stochastic_win_data': stochastic_win_data,
                'stochastic_loss_data': stochastic_loss_data,
                'biased_win_data': biased_win_data,
                'biased_loss_data': biased_loss_data,
                'time_axis': time_axis
            }
        
        elif condition == "previous_trial" and win_loss:
            # For previous trial outcome analysis
            stochastic_prev_win_data = []
            stochastic_prev_loss_data = []
            biased_prev_win_data = []
            biased_prev_loss_data = []
            
            time_axis = None
            
            # Extract data from each subject's results
            for result in subject_results:
                if time_axis is None and 'time_axis' in result:
                    time_axis = result['time_axis']
                
                # Extract previous win/loss data for each state
                if ('stochastic' in result and 
                    result['stochastic'] is not None and 
                    'prev_win_avg' in result['stochastic'] and 
                    result['stochastic']['prev_win_avg'] is not None):
                    stochastic_prev_win_data.append(result['stochastic']['prev_win_avg'])
                
                if ('stochastic' in result and 
                    result['stochastic'] is not None and 
                    'prev_loss_avg' in result['stochastic'] and 
                    result['stochastic']['prev_loss_avg'] is not None):
                    stochastic_prev_loss_data.append(result['stochastic']['prev_loss_avg'])
                    
                if ('biased' in result and 
                    result['biased'] is not None and 
                    'prev_win_avg' in result['biased'] and 
                    result['biased']['prev_win_avg'] is not None):
                    biased_prev_win_data.append(result['biased']['prev_win_avg'])
                    
                if ('biased' in result and 
                    result['biased'] is not None and 
                    'prev_loss_avg' in result['biased'] and 
                    result['biased']['prev_loss_avg'] is not None):
                    biased_prev_loss_data.append(result['biased']['prev_loss_avg'])
            
            if time_axis is None or (not stochastic_prev_win_data and not stochastic_prev_loss_data and 
                                    not biased_prev_win_data and not biased_prev_loss_data):
                print("No valid previous outcome data found for cross-subject analysis")
                return None
                
            # Create the plot
            plt.figure(figsize=(12, 7))
            
            # Define colors and styles for previous win/loss conditions
            colors = {
                'stochastic_prev_win': 'darkgreen',
                'stochastic_prev_loss': 'lightgreen',
                'biased_prev_win': 'purple',
                'biased_prev_loss': 'plum'
            }
            
            linestyles = {
                'stochastic_prev_win': '-',
                'stochastic_prev_loss': '--',
                'biased_prev_win': '-',
                'biased_prev_loss': '--'
            }
            
            # Plot stochastic previous win data if available
            if stochastic_prev_win_data:
                avg = np.mean(stochastic_prev_win_data, axis=0)
                sem = np.std(stochastic_prev_win_data, axis=0) / np.sqrt(len(stochastic_prev_win_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['stochastic_prev_win'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['stochastic_prev_win'],
                        linestyle=linestyles['stochastic_prev_win'],
                        linewidth=2,
                        label=f"Stochastic Previous Win")
                        
            # Plot stochastic previous loss data if available
            if stochastic_prev_loss_data:
                avg = np.mean(stochastic_prev_loss_data, axis=0)
                sem = np.std(stochastic_prev_loss_data, axis=0) / np.sqrt(len(stochastic_prev_loss_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['stochastic_prev_loss'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['stochastic_prev_loss'],
                        linestyle=linestyles['stochastic_prev_loss'],
                        linewidth=2,
                        label=f"Stochastic Previous Loss")
                        
            # Plot biased previous win data if available
            if biased_prev_win_data:
                avg = np.mean(biased_prev_win_data, axis=0)
                sem = np.std(biased_prev_win_data, axis=0) / np.sqrt(len(biased_prev_win_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['biased_prev_win'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['biased_prev_win'],
                        linestyle=linestyles['biased_prev_win'],
                        linewidth=2,
                        label=f"Biased Previous Win")
                        
            # Plot biased previous loss data if available
            if biased_prev_loss_data:
                avg = np.mean(biased_prev_loss_data, axis=0)
                sem = np.std(biased_prev_loss_data, axis=0) / np.sqrt(len(biased_prev_loss_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['biased_prev_loss'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['biased_prev_loss'],
                        linestyle=linestyles['biased_prev_loss'],
                        linewidth=2,
                        label=f"Biased Previous Loss")
            
            # Add reference lines
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add labels and title
            plt.xlabel('Time (s) after first lick', fontsize=24)
            plt.ylabel('z-ΔF/F', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.title(f'Photometry by Behavioral State and Previous Outcome', fontsize=26)
            plt.xlim([-pre_cue_time, post_cue_time])
            #plt.legend(loc='upper right', fontsize=24)
            plt.tight_layout()
            
            # Save the figure
            save_figure(plt.gcf(), "all_subjects", "pooled", "behavioral_states_prev_outcome")
            
            plt.show()
            
            return {
                'subjid': 'All',
                'specific_subjects': specific_subjects,
                'win_loss': win_loss,
                'condition': condition,
                'n_subjects': len(subject_results),
                'stochastic_prev_win_data': stochastic_prev_win_data,
                'stochastic_prev_loss_data': stochastic_prev_loss_data,
                'biased_prev_win_data': biased_prev_win_data,
                'biased_prev_loss_data': biased_prev_loss_data,
                'time_axis': time_axis
            }
            
        else:
            # For regular state analysis (not win-loss)
            # Extract data for each state from all subjects
            stochastic_data = []
            biased_data = []
            time_axis = None
            
            # Extract data from each subject's results
            for result in subject_results:
                if time_axis is None and 'time_axis' in result:
                    time_axis = result['time_axis']
                
                # Extract average data for each state
                if ('stochastic' in result and 
                    result['stochastic'] is not None and 
                    'avg' in result['stochastic'] and 
                    result['stochastic']['avg'] is not None):
                    stochastic_data.append(result['stochastic']['avg'])
                
                if ('biased' in result and 
                    result['biased'] is not None and 
                    'avg' in result['biased'] and 
                    result['biased']['avg'] is not None):
                    biased_data.append(result['biased']['avg'])
            
            if time_axis is None or (not stochastic_data and not biased_data):
                print("No valid data found for cross-subject analysis")
                return None
            
            # Create the plot
            plt.figure(figsize=(12, 7))
            
            # Define colors for states
            colors = {
                'stochastic': 'darkgreen',
                'biased': 'purple'
            }
            
            # Plot stochastic data if available
            if stochastic_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(stochastic_data, axis=0)
                sem = np.std(stochastic_data, axis=0) / np.sqrt(len(stochastic_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['stochastic'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['stochastic'],
                        linewidth=2,
                        label=f"Stochastic")
            
            # Plot biased data if available
            if biased_data:
                # Calculate mean and SEM across subjects
                avg = np.mean(biased_data, axis=0)
                sem = np.std(biased_data, axis=0) / np.sqrt(len(biased_data))
                
                plt.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors['biased'], alpha=0.3)
                plt.plot(time_axis, avg,
                        color=colors['biased'],
                        linewidth=2,
                        label=f"Biased")
            
            # Add reference lines
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add labels and title
            plt.xlabel('Time (s) after first lick', fontsize=24)
            plt.ylabel('z-ΔF/F', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.title(f'Photometry by Behavioral State', fontsize=26)
            plt.xlim([-pre_cue_time, post_cue_time])
            #plt.legend(loc='upper right', fontsize=24)
            plt.tight_layout()
            
            # Save the figure
            save_figure(plt.gcf(), "all_subjects", "pooled", "behavioral_states_combined")
            
            plt.show()
            
            return {
                'subjid': 'All',
                'specific_subjects': specific_subjects,
                'win_loss': win_loss,
                'condition': condition,
                'n_subjects': len(subject_results),
                'stochastic_data': stochastic_data,
                'biased_data': biased_data,
                'time_axis': time_axis
            }
    
    else:
        # Original single-subject analysis (call original function)
        return analyze_behavioral_states_single(subjid, win_loss, behavior_df=behavior_df, condition=condition)

def analyze_behavioral_states_single(subject_id, win_loss=False, threshold=0.8, behavior_df=None, condition=None):
    """
    Analyze photometry signals based on behavioral states (stochastic vs biased)

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    win_loss : bool, optional (default=False)
        If True, further subdivides each state group into win and loss trials
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    condition : str, optional (default=None)
        Condition to analyze: "current_trial" for win/loss, "previous_trial" for previous win/loss,
        or None for state only (ignores win_loss parameter if set to None)

    Returns:
    --------
    dict: Analysis results including state-specific photometry signals
    """
    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])


    required_columns = ['p_stochastic', 'p_leftbias', 'p_rightbias']
    if not all(col in behavior_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in behavior_df.columns]
        print(f"Error: Missing required columns in parquet data: {missing_cols}")
        return None

    # Store results based on analysis type
    state_data = {
        'stochastic': {
            'data': [],
            'win_data': [],
            'loss_data': [],
            'prev_win_data': [],
            'prev_loss_data': []
        },
        'biased': {
            'data': [],
            'win_data': [],
            'loss_data': [],
            'prev_win_data': [],
            'prev_loss_data': []
        }
    }

    time_axis = None
    total_non_m_trials = 0  # Track ALL non-missed trials regardless of state

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        # Get photometry data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Skip sessions with too few trials
        if session_result and len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Get reward outcomes for non-missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]

        # Count ALL non-missed trials for this session
        total_non_m_trials += len(non_m_indices)

        # Get state probabilities for each trial
        # We need to map the photometry trials to the behavioral data
        behavior_data = session_result['behavioral_data']
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'

        # Get original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Create previous reward array (shifted)
        # First trial has no previous trial, assign -1 (will be filtered out)
        prev_rewards = np.zeros_like(reward_outcomes)
        prev_rewards[0] = -1  # No previous trial for first trial
        prev_rewards[1:] = reward_outcomes[:-1]

        # Create a mapping from filtered trial indices to state probabilities
        trial_states = []

        # For each valid photometry trial, determine its state
        for i, orig_idx in enumerate(session_result["valid_trials"]):
            if orig_idx not in session_result["non_m_trials"]:
                continue

            # Get the trial index in the behavioral data
            if orig_idx < len(non_miss_mask) and non_miss_mask[orig_idx]:
                # Find where this trial appears in the non-missed sequence
                filtered_idx = np.where(orig_non_miss_indices == orig_idx)[0]

                if len(filtered_idx) > 0:
                    # This gives the position of this trial within non-missed trials
                    trial_pos = filtered_idx[0]

                    # Use this position to look up the state in session_df if within bounds
                    if trial_pos < len(session_df):
                        # Get state data from parquet
                        p_stochastic = session_df.iloc[trial_pos]['p_stochastic']
                        p_leftbias = session_df.iloc[trial_pos]['p_leftbias']
                        p_rightbias = session_df.iloc[trial_pos]['p_rightbias']

                        # Determine state based on threshold
                        if p_stochastic >= threshold:
                            state = 'stochastic'
                        elif p_leftbias >= threshold or p_rightbias >= threshold:
                            state = 'biased'
                        else:
                            state = 'uncertain'

                        trial_states.append(state)
                    else:
                        trial_states.append('uncertain')
                else:
                    trial_states.append('uncertain')
            else:
                trial_states.append('uncertain')

        # For each valid trial, add its photometry data to the appropriate group
        for i, (state, reward) in enumerate(zip(trial_states, reward_outcomes)):
            if state == 'uncertain':
                continue

            # Get photometry data for this trial
            photometry_data = session_result['plotting_data'][i]

            # Add to overall state data
            state_data[state]['data'].append(photometry_data)

            # Add to win/loss subgroups if analyzing current trial outcomes
            if condition == "current_trial" or (win_loss and condition is None):
                if reward == 1:  # Win
                    state_data[state]['win_data'].append(photometry_data)
                else:  # Loss
                    state_data[state]['loss_data'].append(photometry_data)
                    
            # Add to previous win/loss subgroups if analyzing previous trial outcomes
            if condition == "previous_trial" or (win_loss and condition is None):
                # Skip first trial (no previous outcome)
                non_m_idx = non_m_indices[i] if i < len(non_m_indices) else None
                if non_m_idx is not None and non_m_idx < len(prev_rewards):
                    prev_outcome = prev_rewards[non_m_idx]
                    if prev_outcome != -1:  # Not the first trial
                        if prev_outcome == 1:  # Previous Win
                            state_data[state]['prev_win_data'].append(photometry_data)
                        else:  # Previous Loss
                            state_data[state]['prev_loss_data'].append(photometry_data)

    # Check if we found any valid trials
    stochastic_count = len(state_data['stochastic']['data'])
    biased_count = len(state_data['biased']['data'])
    uncertain_count = total_non_m_trials - stochastic_count - biased_count

    if stochastic_count + biased_count == 0:
        print(f"No trials meeting state criteria found for {subject_id}")
        return None

    # Calculate percentages for reporting based on ALL non-missed trials
    stochastic_pct = (stochastic_count / total_non_m_trials) * 100
    biased_pct = (biased_count / total_non_m_trials) * 100
    uncertain_pct = (uncertain_count / total_non_m_trials) * 100

    print(f"\nFound {stochastic_count} stochastic trials ({stochastic_pct:.1f}%) and "
          f"{biased_count} biased trials ({biased_pct:.1f}%) "
          f"out of {total_non_m_trials} total non-missed trials")

    if uncertain_count > 0:
        print(f"Note: {uncertain_count} trials ({uncertain_pct:.1f}%) did not meet the threshold "
              f"criteria of {threshold} for any state and were excluded")

    # Convert lists to numpy arrays
    for state in ['stochastic', 'biased']:
        if state_data[state]['data']:
            state_data[state]['data'] = np.array(state_data[state]['data'])
        if state_data[state]['win_data']:
            state_data[state]['win_data'] = np.array(state_data[state]['win_data'])
        if state_data[state]['loss_data']:
            state_data[state]['loss_data'] = np.array(state_data[state]['loss_data'])
        if state_data[state]['prev_win_data']:
            state_data[state]['prev_win_data'] = np.array(state_data[state]['prev_win_data'])
        if state_data[state]['prev_loss_data']:
            state_data[state]['prev_loss_data'] = np.array(state_data[state]['prev_loss_data'])

    # Calculate averages and SEMs
    for state in ['stochastic', 'biased']:
        if len(state_data[state]['data']) > 0:
            state_data[state]['avg'] = np.mean(state_data[state]['data'], axis=0)
            state_data[state]['sem'] = calculate_sem(state_data[state]['data'], axis=0)
        else:
            state_data[state]['avg'] = None
            state_data[state]['sem'] = None

        if len(state_data[state]['win_data']) > 0:
            state_data[state]['win_avg'] = np.mean(state_data[state]['win_data'], axis=0)
            state_data[state]['win_sem'] = calculate_sem(state_data[state]['win_data'], axis=0)
        else:
            state_data[state]['win_avg'] = None
            state_data[state]['win_sem'] = None

        if len(state_data[state]['loss_data']) > 0:
            state_data[state]['loss_avg'] = np.mean(state_data[state]['loss_data'], axis=0)
            state_data[state]['loss_sem'] = calculate_sem(state_data[state]['loss_data'], axis=0)
        else:
            state_data[state]['loss_avg'] = None
            state_data[state]['loss_sem'] = None
            
        if len(state_data[state]['prev_win_data']) > 0:
            state_data[state]['prev_win_avg'] = np.mean(state_data[state]['prev_win_data'], axis=0)
            state_data[state]['prev_win_sem'] = calculate_sem(state_data[state]['prev_win_data'], axis=0)
        else:
            state_data[state]['prev_win_avg'] = None
            state_data[state]['prev_win_sem'] = None
            
        if len(state_data[state]['prev_loss_data']) > 0:
            state_data[state]['prev_loss_avg'] = np.mean(state_data[state]['prev_loss_data'], axis=0)
            state_data[state]['prev_loss_sem'] = calculate_sem(state_data[state]['prev_loss_data'], axis=0)
        else:
            state_data[state]['prev_loss_avg'] = None
            state_data[state]['prev_loss_sem'] = None

    # Create plot based on analysis type
    plt.figure(figsize=(12, 7))

    if condition == "current_trial" and win_loss:
        # Plot win/loss separated for each state
        colors = {
            'stochastic_win': 'darkgreen',
            'stochastic_loss': 'lightgreen',
            'biased_win': 'purple',
            'biased_loss': 'plum'
        }

        linestyles = {
            'stochastic_win': '-',
            'stochastic_loss': '--',
            'biased_win': '-',
            'biased_loss': '--'
        }

        # Plot stochastic wins
        if state_data['stochastic']['win_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['stochastic']['win_avg'] - state_data['stochastic']['win_sem'],
                             state_data['stochastic']['win_avg'] + state_data['stochastic']['win_sem'],
                             color=colors['stochastic_win'], alpha=0.3)
            plt.plot(time_axis, state_data['stochastic']['win_avg'],
                     color=colors['stochastic_win'],
                     linestyle=linestyles['stochastic_win'],
                     linewidth=2,
                     label=f"Stochastic Win (n={len(state_data['stochastic']['win_data'])})")

        # Plot stochastic losses
        if state_data['stochastic']['loss_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['stochastic']['loss_avg'] - state_data['stochastic']['loss_sem'],
                             state_data['stochastic']['loss_avg'] + state_data['stochastic']['loss_sem'],
                             color=colors['stochastic_loss'], alpha=0.3)
            plt.plot(time_axis, state_data['stochastic']['loss_avg'],
                     color=colors['stochastic_loss'],
                     linestyle=linestyles['stochastic_loss'],
                     linewidth=2,
                     label=f"Stochastic Loss (n={len(state_data['stochastic']['loss_data'])})")

        # Plot biased wins
        if state_data['biased']['win_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['biased']['win_avg'] - state_data['biased']['win_sem'],
                             state_data['biased']['win_avg'] + state_data['biased']['win_sem'],
                             color=colors['biased_win'], alpha=0.3)
            plt.plot(time_axis, state_data['biased']['win_avg'],
                     color=colors['biased_win'],
                     linestyle=linestyles['biased_win'],
                     linewidth=2,
                     label=f"Biased Win (n={len(state_data['biased']['win_data'])})")

        # Plot biased losses
        if state_data['biased']['loss_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['biased']['loss_avg'] - state_data['biased']['loss_sem'],
                             state_data['biased']['loss_avg'] + state_data['biased']['loss_sem'],
                             color=colors['biased_loss'], alpha=0.3)
            plt.plot(time_axis, state_data['biased']['loss_avg'],
                     color=colors['biased_loss'],
                     linestyle=linestyles['biased_loss'],
                     linewidth=2,
                     label=f"Biased Loss (n={len(state_data['biased']['loss_data'])})")
                     
        plot_title = f'Photometry by Behavioral State and Outcome: {subject_id}'
        save_suffix = "behavioral_states_winloss"
        
    elif condition == "previous_trial" and win_loss:
        # Plot previous win/loss separated for each state
        colors = {
            'stochastic_prev_win': 'darkgreen',
            'stochastic_prev_loss': 'lightgreen',
            'biased_prev_win': 'purple',
            'biased_prev_loss': 'plum'
        }

        linestyles = {
            'stochastic_prev_win': '-',
            'stochastic_prev_loss': '--',
            'biased_prev_win': '-',
            'biased_prev_loss': '--'
        }

        # Plot stochastic previous wins
        if state_data['stochastic']['prev_win_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['stochastic']['prev_win_avg'] - state_data['stochastic']['prev_win_sem'],
                             state_data['stochastic']['prev_win_avg'] + state_data['stochastic']['prev_win_sem'],
                             color=colors['stochastic_prev_win'], alpha=0.3)
            plt.plot(time_axis, state_data['stochastic']['prev_win_avg'],
                     color=colors['stochastic_prev_win'],
                     linestyle=linestyles['stochastic_prev_win'],
                     linewidth=2,
                     label=f"Stochastic Previous Win (n={len(state_data['stochastic']['prev_win_data'])})")

        # Plot stochastic previous losses
        if state_data['stochastic']['prev_loss_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['stochastic']['prev_loss_avg'] - state_data['stochastic']['prev_loss_sem'],
                             state_data['stochastic']['prev_loss_avg'] + state_data['stochastic']['prev_loss_sem'],
                             color=colors['stochastic_prev_loss'], alpha=0.3)
            plt.plot(time_axis, state_data['stochastic']['prev_loss_avg'],
                     color=colors['stochastic_prev_loss'],
                     linestyle=linestyles['stochastic_prev_loss'],
                     linewidth=2,
                     label=f"Stochastic Previous Loss (n={len(state_data['stochastic']['prev_loss_data'])})")

        # Plot biased previous wins
        if state_data['biased']['prev_win_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['biased']['prev_win_avg'] - state_data['biased']['prev_win_sem'],
                             state_data['biased']['prev_win_avg'] + state_data['biased']['prev_win_sem'],
                             color=colors['biased_prev_win'], alpha=0.3)
            plt.plot(time_axis, state_data['biased']['prev_win_avg'],
                     color=colors['biased_prev_win'],
                     linestyle=linestyles['biased_prev_win'],
                     linewidth=2,
                     label=f"Biased Previous Win (n={len(state_data['biased']['prev_win_data'])})")

        # Plot biased previous losses
        if state_data['biased']['prev_loss_avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['biased']['prev_loss_avg'] - state_data['biased']['prev_loss_sem'],
                             state_data['biased']['prev_loss_avg'] + state_data['biased']['prev_loss_sem'],
                             color=colors['biased_prev_loss'], alpha=0.3)
            plt.plot(time_axis, state_data['biased']['prev_loss_avg'],
                     color=colors['biased_prev_loss'],
                     linestyle=linestyles['biased_prev_loss'],
                     linewidth=2,
                     label=f"Biased Previous Loss (n={len(state_data['biased']['prev_loss_data'])})")
                     
        plot_title = f'Photometry by Behavioral State and Previous Outcome: {subject_id}'
        save_suffix = "behavioral_states_prev_outcome"
        
    else:
        # Plot overall state averages without separating by outcome
        colors = {
            'stochastic': 'darkgreen',
            'biased': 'purple'
        }

        # Plot stochastic trials
        if state_data['stochastic']['avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['stochastic']['avg'] - state_data['stochastic']['sem'],
                             state_data['stochastic']['avg'] + state_data['stochastic']['sem'],
                             color=colors['stochastic'], alpha=0.3)
            plt.plot(time_axis, state_data['stochastic']['avg'],
                     color=colors['stochastic'],
                     linewidth=2,
                     label=f"Stochastic (n={len(state_data['stochastic']['data'])})")

        # Plot biased trials
        if state_data['biased']['avg'] is not None:
            plt.fill_between(time_axis,
                             state_data['biased']['avg'] - state_data['biased']['sem'],
                             state_data['biased']['avg'] + state_data['biased']['sem'],
                             color=colors['biased'], alpha=0.3)
            plt.plot(time_axis, state_data['biased']['avg'],
                     color=colors['biased'],
                     linewidth=2,
                     label=f"Biased (n={len(state_data['biased']['data'])})")
                     
        plot_title = f'Photometry by Behavioral State: {subject_id}'
        save_suffix = "behavioral_states_combined"

    # Add reference lines
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add labels and title
    plt.xlabel('Time (s) after first lick', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylabel('z-ΔF/F', fontsize=24)
    plt.yticks(fontsize=24)

    plt.title(plot_title, fontsize=26)
    plt.xlim([-pre_cue_time, post_cue_time])
    #plt.legend(loc='upper right', fontsize=12)

    # Add threshold info and classification stats to the plot
    plt.figtext(0.01, 0.01,
                f"State threshold: {threshold} | Stochastic: {stochastic_pct:.1f}%, Biased: {biased_pct:.1f}%, Uncertain: {uncertain_pct:.1f}%",
                fontsize=9)

    plt.tight_layout()

    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", save_suffix)

    plt.show()

    return {
        'subject_id': subject_id,
        'threshold': threshold,
        'time_axis': time_axis,
        'stochastic': state_data['stochastic'],
        'biased': state_data['biased'],
        'stochastic_count': stochastic_count, 
        'biased_count': biased_count,
        'uncertain_count': uncertain_count,
        'total_non_m_trials': total_non_m_trials,
        'stochastic_pct': stochastic_pct,
        'biased_pct': biased_pct,
        'uncertain_pct': uncertain_pct
    }

def plot_state_probabilities(subject_id, session_date=None, behavior_df=None):
    """
    Plot the state probabilities (p_stochastic, p_leftbias, p_rightbias) across trials
    for all sessions of a subject or for a specific session.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, plot all sessions as subplots.

    Returns:
    --------
    None
    """
    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])


    valid_sessions = []
    for session in sessions:
        session_result = process_session(subject_id, session)
        if not session_result:
            continue
            
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue
        
        valid_sessions.append(session)
    
    # Update the sessions list to only include valid sessions
    sessions = valid_sessions
    
    if len(sessions) == 0:
        print(f"No sessions with at least 100 valid trials found for {subject_id}")
        return None

    # Determine which sessions to plot
    if session_date:
        # Plot single session
        if session_date not in sessions:
            print(f"Session {session_date} does not meet the 100 valid trials threshold")
            return None
            
        sessions = [session_date]
        session_data = subject_data[subject_data['date'] == session_date]
        if session_data.empty:
            print(f"No data found for session {session_date}")
            return None
    else:
        # Plot all valid sessions
        sessions = sorted(set(subject_data['date'].unique()) & set(sessions))

    # Setup the figure
    if len(sessions) == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax]  # Make it iterable for consistent code
    else:
        # Calculate rows and columns for subplots
        n_sessions = len(sessions)
        n_cols = min(3, n_sessions)
        n_rows = (n_sessions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()  # Flatten to make indexing easier

    # Set a color scheme
    colors = {
        'p_stochastic': 'green',
        'p_leftbias': 'red',
        'p_rightbias': 'blue'
    }

    label_names = {
        'p_stochastic': 'Stochastic',
        'p_leftbias': 'Left Bias',
        'p_rightbias': 'Right Bias'
    }

    # Plot each session
    for i, session in enumerate(sessions):
        if i >= len(axes):
            print(f"Warning: Not enough subplots for all sessions. Showing first {len(axes)} sessions.")
            break

        ax = axes[i]

        session_data = subject_data[subject_data['date'] == session]
        if session_data.empty:
            ax.text(0.5, 0.5, f"No data for {session}", ha='center', va='center')
            continue

        # Sort by trial number if available
        if 'trialnum' in session_data.columns:
            session_data = session_data.sort_values('trialnum')
            x_values = session_data['trialnum'].values
            x_label = 'Trial Number'
        else:
            x_values = np.arange(len(session_data))
            x_label = 'Trial Index'

        # Plot each probability
        for prob, color in colors.items():
            if prob in session_data.columns:
                ax.plot(x_values, session_data[prob], color=color, linewidth=2, label=label_names[prob])

        # Add reference lines and formatting
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f"{subject_id} - {session}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Add win/loss markers
        if 'reward' in session_data.columns:
            win_trials = session_data[session_data['reward'] == 1]
            loss_trials = session_data[session_data['reward'] == 0]

            if 'trialnum' in session_data.columns:
                win_x = win_trials['trialnum'].values
                loss_x = loss_trials['trialnum'].values
            else:
                win_indices = session_data.index[session_data['reward'] == 1]
                loss_indices = session_data.index[session_data['reward'] == 0]
                win_x = np.arange(len(session_data))[np.isin(session_data.index, win_indices)]
                loss_x = np.arange(len(session_data))[np.isin(session_data.index, loss_indices)]

            # Add small tick marks at bottom for wins/losses
            ax.scatter(win_x, np.zeros_like(win_x) - 0.02, marker='|', color='green', alpha=0.5, s=20)
            ax.scatter(loss_x, np.zeros_like(loss_x) - 0.02, marker='|', color='red', alpha=0.5, s=20)

            # Add small legend for win/loss markers
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='green', marker='|', linestyle='None'),
                            Line2D([0], [0], color='red', marker='|', linestyle='None')]
            ax.legend(custom_lines, ['Win', 'Loss'], loc='lower right', fontsize=8)

            # Re-add the main legend
            handles, labels = [], []
            for prob, color in colors.items():
                if prob in session_data.columns:
                    handles.append(Line2D([0], [0], color=color, linewidth=2))
                    labels.append(label_names[prob])
            ax.legend(handles, labels, loc='upper right')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add overall title
    if len(sessions) == 1:
        fig.suptitle(f"State Probabilities for {subject_id} - {sessions[0]}", fontsize=14)
    else:
        fig.suptitle(f"State Probabilities for {subject_id} Across Sessions", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the figure
    if len(sessions) == 1:
        save_figure(fig, subject_id, sessions[0], "state_probabilities")
    else:
        save_figure(fig, subject_id, "pooled", "state_probabilities_all_sessions")

    plt.show()

    return None

def analyze_state_occupation(behavior_df=None, subjid="All", save_fig=True, smooth_window=4):
    """
    Calculate and plot the percentage of trials spent in each behavioral state 
    (stochastic, left_biased, right_biased) across sessions with smoothing option.
    
    Parameters:
    -----------
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavioral dataframe to analyze
    subjid : str or list, optional
        Subject ID to analyze, or "All" for group analysis
    save_fig : bool, optional
        Whether to save the generated figure
    smooth_window : int, optional (default=4)
        Number of sessions to use for smoothing window (0 for no smoothing)
    
    Returns:
    --------
    dict: Dictionary with state occupation data by subject and session
    """
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            print(f"Loaded behavior data from parquet file: {len(behavior_df)} rows")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None
            
    # Check for required columns
    required_columns = ['p_stochastic', 'p_leftbias', 'p_rightbias', 'subjid', 'date']
    if not all(col in behavior_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in behavior_df.columns]
        print(f"Error: Missing required columns: {missing}")
        return None
        
    # Set up default subject list for "All"
    if subjid == "All":
        subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
        print(f"Using default subject list: {subjects}")
    elif isinstance(subjid, list):
        subjects = subjid
    else:
        subjects = [subjid]
        
    print(f"Analyzing state occupation for {len(subjects)} subjects")
    
    # Threshold for state assignment
    threshold = 0.8
    
    # Store results for individual subjects
    subject_results = {}
    max_sessions = 0
    
    # Process each subject
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_data = behavior_df[behavior_df['subjid'] == subject_id]
        
        # Sort sessions chronologically
        sessions = sorted(subject_data['date'].unique())
        session_states = []
        
        # Process each session
        for i, session_date in enumerate(sessions):
            session_df = subject_data[subject_data['date'] == session_date]
            
            # Skip sessions with too few trials
            if len(session_df) < 100:
                print(f"Skipping {subject_id}/{session_date}, fewer than 100 trials ({len(session_df)}).")
                continue
                
            # Count trials in each state
            state_counts = {
                'stochastic': 0,
                'left_biased': 0,
                'right_biased': 0,
                'uncertain': 0
            }
            
            # Classify each trial by state
            for _, trial in session_df.iterrows():
                p_stochastic = trial['p_stochastic']
                p_leftbias = trial['p_leftbias']
                p_rightbias = trial['p_rightbias']
                
                if p_stochastic >= threshold:
                    state_counts['stochastic'] += 1
                elif p_leftbias >= threshold:
                    state_counts['left_biased'] += 1
                elif p_rightbias >= threshold:
                    state_counts['right_biased'] += 1
                else:
                    state_counts['uncertain'] += 1
            
            # Calculate percentages excluding uncertain trials
            classified_trials = state_counts['stochastic'] + state_counts['left_biased'] + state_counts['right_biased']
            
            if classified_trials > 0:
                pct_stochastic = state_counts['stochastic'] / classified_trials
                pct_left_biased = state_counts['left_biased'] / classified_trials
                pct_right_biased = state_counts['right_biased'] / classified_trials
                
                session_states.append({
                    'session_num': i + 1,  # 1-indexed session number
                    'session_date': session_date,
                    'stochastic': pct_stochastic,
                    'left_biased': pct_left_biased,
                    'right_biased': pct_right_biased,
                    'total_trials': len(session_df),
                    'classified_trials': classified_trials,
                    'uncertain_trials': state_counts['uncertain']
                })
        
        # Store results for this subject
        if session_states:
            subject_results[subject_id] = session_states
            max_sessions = max(max_sessions, len(session_states))
    
    # Function to apply moving average smoothing
    def smooth_data(data, window_size):
        if window_size <= 1:
            return data  # No smoothing
        
        # Apply centered moving average
        smoothed = []
        for i in range(len(data)):
            # Calculate window bounds
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            
            # Calculate average of window
            window_data = data[start:end]
            smoothed.append(sum(window_data) / len(window_data))
        
        return smoothed
    
    # Plot individual subject results
    for subject_id, sessions in subject_results.items():
        if subjid != "All":  # Only plot individual subjects if not group analysis
            plt.figure(figsize=(10, 6))
            
            # Prepare data for plotting
            session_nums = [s['session_num'] for s in sessions]
            stochastic_pcts = [s['stochastic'] for s in sessions]
            left_biased_pcts = [s['left_biased'] for s in sessions]
            right_biased_pcts = [s['right_biased'] for s in sessions]
            
            # Apply smoothing if requested
            if smooth_window > 1 and len(session_nums) > smooth_window:
                stochastic_smooth = smooth_data(stochastic_pcts, smooth_window)
                left_biased_smooth = smooth_data(left_biased_pcts, smooth_window)
                right_biased_smooth = smooth_data(right_biased_pcts, smooth_window)
                
                # Plot both raw data (thin transparent lines) and smoothed data
                plt.plot(session_nums, stochastic_pcts, '-', color='green', alpha=0.3, linewidth=1)
                plt.plot(session_nums, left_biased_pcts, '-', color='red', alpha=0.3, linewidth=1)
                plt.plot(session_nums, right_biased_pcts, '-', color='blue', alpha=0.3, linewidth=1)
                
                # Plot smoothed data - use same x-axis as raw data
                plt.plot(session_nums, stochastic_smooth, '-', color='green', linewidth=2.5, label='Stochastic')
                plt.plot(session_nums, left_biased_smooth, '-', color='red', linewidth=2.5, label='Left Biased')
                plt.plot(session_nums, right_biased_smooth, '-', color='blue', linewidth=2.5, label='Right Biased')
            else:
                # Plot raw data only
                plt.plot(session_nums, stochastic_pcts, '-', color='green', linewidth=2, label='Stochastic')
                plt.plot(session_nums, left_biased_pcts, '-', color='red', linewidth=2, label='Left Biased')
                plt.plot(session_nums, right_biased_pcts, '-', color='blue', linewidth=2, label='Right Biased')
            
            plt.xlabel('Session Number', fontsize=12)
            plt.ylabel('State Occupation (proportion)', fontsize=12)
            smoothing_info = f" (Smoothed, window={smooth_window})" if smooth_window > 1 else ""
            plt.title(f'State Occupation Across Sessions - {subject_id}{smoothing_info}', fontsize=14)
            
            plt.xlim(0.5, len(sessions) + 0.5)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Add a note about uncertain trials
            avg_uncertain = np.mean([s['uncertain_trials']/s['total_trials'] for s in sessions])
            plt.figtext(0.1, 0.01, f"Average uncertain trials: {avg_uncertain:.1%} (excluded from percentages)", 
                       fontsize=10, ha='left')
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.98])
            
            # Save figure
            if save_fig:
                figure_path = os.path.join(output_dir, subject_id, "pooled")
                ensure_directory_exists(figure_path)
                fig_file = os.path.join(figure_path, f"state_occupation{'' if smooth_window <= 1 else f'_smooth{smooth_window}'}.png")
                
                try:
                    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
                    print(f"Saved figure to {fig_file}")
                except Exception as e:
                    print(f"Error saving figure: {e}")
            
            plt.show()
    
    # Group analysis for all subjects
    if subjid == "All":
        print("\nGenerating group analysis...")
        
        # Prepare data structure for group averages
        sessions_data = [[] for _ in range(max_sessions + 1)]  # +1 for 1-indexed sessions
        
        # Collect data from all subjects by session number
        for subject_id, sessions in subject_results.items():
            for session in sessions:
                session_num = session['session_num']
                if session_num <= max_sessions:
                    sessions_data[session_num].append({
                        'stochastic': session['stochastic'],
                        'left_biased': session['left_biased'],
                        'right_biased': session['right_biased']
                    })
        
        # Calculate group averages where we have at least 3 subjects
        x_values = []
        stochastic_means = []
        stochastic_sems = []
        left_biased_means = []
        left_biased_sems = []
        right_biased_means = []
        right_biased_sems = []
        
        for session_num, session_data in enumerate(sessions_data):
            if session_num == 0:
                continue  # Skip placeholder at index 0
                
            if len(session_data) >= 3:  # Only include if we have at least 3 subjects
                x_values.append(session_num)
                
                # Calculate means and SEMs for each state
                stochastic_values = [s['stochastic'] for s in session_data]
                left_biased_values = [s['left_biased'] for s in session_data]
                right_biased_values = [s['right_biased'] for s in session_data]
                
                stochastic_means.append(np.mean(stochastic_values))
                stochastic_sems.append(np.std(stochastic_values) / np.sqrt(len(stochastic_values)))
                
                left_biased_means.append(np.mean(left_biased_values))
                left_biased_sems.append(np.std(left_biased_values) / np.sqrt(len(left_biased_values)))
                
                right_biased_means.append(np.mean(right_biased_values))
                right_biased_sems.append(np.std(right_biased_values) / np.sqrt(len(right_biased_values)))
        
        # Plot group results if we have data
        if x_values:
            plt.figure(figsize=(12, 7))
            
            # Plot individual subject data as thin grey lines
            for subject_id, sessions in subject_results.items():
                session_nums = [s['session_num'] for s in sessions]
                stochastic_pcts = [s['stochastic'] for s in sessions]
                left_biased_pcts = [s['left_biased'] for s in sessions]
                right_biased_pcts = [s['right_biased'] for s in sessions]
                
                plt.plot(session_nums, stochastic_pcts, '-', color='gray', linewidth=0.8, alpha=0.3)
                plt.plot(session_nums, left_biased_pcts, '-', color='gray', linewidth=0.8, alpha=0.3)
                plt.plot(session_nums, right_biased_pcts, '-', color='gray', linewidth=0.8, alpha=0.3)
            
            # Apply smoothing to group data if requested
            if smooth_window > 1 and len(x_values) > smooth_window:
                # Smooth the means
                stochastic_smooth = smooth_data(stochastic_means, smooth_window)
                left_biased_smooth = smooth_data(left_biased_means, smooth_window)
                right_biased_smooth = smooth_data(right_biased_means, smooth_window)
                
                # Smooth the SEMs (using same window size)
                stochastic_sems_smooth = smooth_data(stochastic_sems, smooth_window)
                left_biased_sems_smooth = smooth_data(left_biased_sems, smooth_window)
                right_biased_sems_smooth = smooth_data(right_biased_sems, smooth_window)
                
                # Plot raw means with thin lines
                plt.plot(x_values, stochastic_means, '-', color='green', linewidth=1, alpha=0.3)
                plt.plot(x_values, left_biased_means, '-', color='red', linewidth=1, alpha=0.3)
                plt.plot(x_values, right_biased_means, '-', color='blue', linewidth=1, alpha=0.3)
                
                # Plot smoothed lines - use the same x-values for both raw and smoothed data
                plt.plot(x_values, stochastic_smooth, '-', color='green', linewidth=2.5, label='Stochastic')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(stochastic_smooth, stochastic_sems_smooth)],
                                [m + s for m, s in zip(stochastic_smooth, stochastic_sems_smooth)],
                                color='green', alpha=0.2)
                
                plt.plot(x_values, left_biased_smooth, '-', color='red', linewidth=2.5, label='Left Biased')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(left_biased_smooth, left_biased_sems_smooth)],
                                [m + s for m, s in zip(left_biased_smooth, left_biased_sems_smooth)],
                                color='red', alpha=0.2)
                
                plt.plot(x_values, right_biased_smooth, '-', color='blue', linewidth=2.5, label='Right Biased')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(right_biased_smooth, right_biased_sems_smooth)],
                                [m + s for m, s in zip(right_biased_smooth, right_biased_sems_smooth)],
                                color='blue', alpha=0.2)
            else:
                # Plot non-smoothed group means with shaded error areas
                plt.plot(x_values, stochastic_means, '-', color='green', linewidth=2.5, label='Stochastic')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(stochastic_means, stochastic_sems)],
                                [m + s for m, s in zip(stochastic_means, stochastic_sems)],
                                color='green', alpha=0.2)
                
                plt.plot(x_values, left_biased_means, '-', color='red', linewidth=2.5, label='Left Biased')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(left_biased_means, left_biased_sems)],
                                [m + s for m, s in zip(left_biased_means, left_biased_sems)],
                                color='red', alpha=0.2)
                
                plt.plot(x_values, right_biased_means, '-', color='blue', linewidth=2.5, label='Right Biased')
                plt.fill_between(x_values, 
                                [m - s for m, s in zip(right_biased_means, right_biased_sems)],
                                [m + s for m, s in zip(right_biased_means, right_biased_sems)],
                                color='blue', alpha=0.2)
            
            plt.xlabel('Session Number', fontsize=14)
            plt.ylabel('State Occupation (proportion)', fontsize=14)
            smoothing_info = f" (Smoothed, window={smooth_window})" if smooth_window > 1 else ""
            plt.title(f'Group State Occupation Across Sessions (n={len(subjects)}){smoothing_info}', fontsize=16)
            
            plt.xlim(0.5, max(x_values) + 0.5)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            if save_fig:
                figure_path = os.path.join(output_dir, "group_analysis")
                ensure_directory_exists(figure_path)
                fig_file = os.path.join(figure_path, f"group_state_occupation{'' if smooth_window <= 1 else f'_smooth{smooth_window}'}.png")
                
                try:
                    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
                    print(f"Saved figure to {fig_file}")
                except Exception as e:
                    print(f"Error saving figure: {e}")
            
            plt.show()
        else:
            print("Not enough data for group analysis")
    
    # Prepare final results to return
    results = {
        'subjects': subjects,
        'subject_results': subject_results,
        'smooth_window': smooth_window
    }
    
    if subjid == "All":
        if x_values:
            if smooth_window > 1 and len(x_values) > smooth_window:
                results['group_data'] = {
                    'x': x_values,
                    'stochastic_mean': stochastic_smooth,
                    'stochastic_sem': stochastic_sems_smooth,
                    'left_biased_mean': left_biased_smooth,
                    'left_biased_sem': left_biased_sems_smooth,
                    'right_biased_mean': right_biased_smooth,
                    'right_biased_sem': right_biased_sems_smooth
                }
            else:
                results['group_data'] = {
                    'x': x_values,
                    'stochastic_mean': stochastic_means,
                    'stochastic_sem': stochastic_sems,
                    'left_biased_mean': left_biased_means,
                    'left_biased_sem': left_biased_sems,
                    'right_biased_mean': right_biased_means,
                    'right_biased_sem': right_biased_sems
                }
    
    return results

def analyze_previous_outcome_effect_by_state(subject_id, threshold=0.8, specific_subjects=None, behavior_df=None):
    """
    Analyze photometry signals based on previous and current trial outcomes,
    separated by behavioral state (stochastic vs biased).

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject, or "All" for cross-subject analysis
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file

    Returns:
    --------
    dict: Analysis results including state-specific previous outcome effects
    """
    if subject_id == "All":
        if specific_subjects is None:
            # Default list of subjects
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Make sure behavior_df is loaded
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
        
        # Container for subject-level results
        subject_results = []
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for previous outcome effect analysis...")
            try:
                # Process individual subject
                subj_result = analyze_previous_outcome_effect_by_state_single(subj, threshold, behavior_df=behavior_df)
                if subj_result:
                    subject_results.append(subj_result)
            except Exception as e:
                print(f"Error processing subject {subj}: {e}")
        
        if not subject_results:
            print("No valid subjects found for cross-subject analysis")
            return None
        
        # Store time axis from first subject
        time_axis = subject_results[0]['time_axis'] if 'time_axis' in subject_results[0] else None
        
        # Prepare containers for cross-subject data
        cross_subject_data = {
            'stochastic': {
                'prev_win_curr_win': [],  # Each subject's average will be one entry
                'prev_win_curr_loss': [],
                'prev_loss_curr_win': [],
                'prev_loss_curr_loss': []
            },
            'biased': {
                'prev_win_curr_win': [],
                'prev_win_curr_loss': [],
                'prev_loss_curr_win': [],
                'prev_loss_curr_loss': []
            }
        }
        
        # Group data by subject
        for result in subject_results:
            if result and 'condition_data' in result:
                for state in ['stochastic', 'biased']:
                    for condition in cross_subject_data[state].keys():
                        if (state in result['condition_data'] and 
                            condition in result['condition_data'][state] and 
                            'avg' in result['condition_data'][state][condition] and
                            result['condition_data'][state][condition]['avg'] is not None and
                            len(result['condition_data'][state][condition]['avg']) > 0):
                            # Add this subject's average to our collection
                            cross_subject_data[state][condition].append(result['condition_data'][state][condition]['avg'])
        
        # Check if we have enough data to create plots
        if time_axis is None:
            print("No time axis found for cross-subject analysis")
            return None
        
        # Calculate cross-subject averages and SEMs
        cross_subject_averages = {
            'stochastic': {},
            'biased': {}
        }
        
        for state in ['stochastic', 'biased']:
            for condition in cross_subject_data[state].keys():
                if cross_subject_data[state][condition]:
                    avg = np.mean(cross_subject_data[state][condition], axis=0)
                    sem = np.std(cross_subject_data[state][condition], axis=0) / np.sqrt(len(cross_subject_data[state][condition]))
                    count = len(cross_subject_data[state][condition])
                    
                    cross_subject_averages[state][condition] = {
                        'avg': avg,
                        'sem': sem,
                        'count': count
                    }
                else:
                    cross_subject_averages[state][condition] = {
                        'avg': None,
                        'sem': None,
                        'count': 0
                    }
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        
        # Define colors and labels
        colors = {
            'prev_win_curr_win': '#117733',  # green
            'prev_win_curr_loss': '#DDCC77', # yellow
            'prev_loss_curr_win': '#4477AA', # blue
            'prev_loss_curr_loss': '#CC6677' # red
        }
        
        condition_labels = {
            'prev_win_curr_win': 'Win→Win',
            'prev_win_curr_loss': 'Win→Loss',
            'prev_loss_curr_win': 'Loss→Win',
            'prev_loss_curr_loss': 'Loss→Loss'
        }
        
        # Find global min/max for consistent y scaling
        y_min = float('inf')
        y_max = float('-inf')
        
        for state in ['stochastic', 'biased']:
            for condition in cross_subject_averages[state].keys():
                if cross_subject_averages[state][condition]['avg'] is not None:
                    avg = cross_subject_averages[state][condition]['avg']
                    sem = cross_subject_averages[state][condition]['sem']
                    y_min = min(y_min, np.min(avg - sem))
                    y_max = max(y_max, np.max(avg + sem))
        
        # Add some padding to y limits
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        # Plot each state in its subplot
        for i, state in enumerate(['stochastic', 'biased']):
            ax = axes[i]
            
            for condition, color in colors.items():
                if cross_subject_averages[state][condition]['avg'] is not None:
                    avg = cross_subject_averages[state][condition]['avg']
                    sem = cross_subject_averages[state][condition]['sem']
                    count = cross_subject_averages[state][condition]['count']
                    
                    # Only plot if we have data
                    if len(avg) > 0 and count > 0:
                        ax.fill_between(time_axis,
                                        avg - sem,
                                        avg + sem,
                                        color=color, alpha=0.3)
                        ax.plot(time_axis, avg,
                                color=color, linewidth=2,
                                label=f'{condition_labels[condition]} (n={count} subjects)')
            
            # Add vertical line at cue onset
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Set the same y-limits for both subplots
            ax.set_ylim(y_min, y_max)
            
            # Labels and formatting
            ax.set_xlabel('Time (s)', fontsize=12)
            if i == 0:
                ax.set_ylabel('ΔF/F', fontsize=12)
            
            ax.set_title(f'{state.capitalize()} State', fontsize=14)
            ax.set_xlim([-pre_cue_time, post_cue_time])
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.2)
        
        # Add main title
        plt.suptitle(f'Cross-Subject Previous Outcome Effect by Behavioral State (n={len(subject_results)} subjects)',
                     fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the subtitle
        
        # Save the figure
        save_figure(fig, "all_subjects", "pooled", f"previous_outcome_by_state_thresh{threshold}")
        
        plt.show()
        
        # Return analysis results
        return {
            'subject_id': 'All',
            'specific_subjects': specific_subjects,
            'n_subjects': len(subject_results),
            'threshold': threshold,
            'time_axis': time_axis,
            'cross_subject_averages': cross_subject_averages,
            'cross_subject_data': cross_subject_data,
        }
    
    else:
        # Original single-subject analysis
        return analyze_previous_outcome_effect_by_state_single(subject_id, threshold, behavior_df=behavior_df)


def analyze_previous_outcome_effect_by_state_single(subject_id, threshold=0.8, behavior_df=None):
    """
    Analyze photometry signals based on previous and current trial outcomes,
    separated by behavioral state (stochastic vs biased).

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state

    Returns:
    --------
    dict: Analysis results including state-specific previous outcome effects
    """
    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
            
    # Store data by state and condition
    state_data = {
        'stochastic': {
            'prev_win_curr_win': [],
            'prev_win_curr_loss': [],
            'prev_loss_curr_win': [],
            'prev_loss_curr_loss': []
        },
        'biased': {
            'prev_win_curr_win': [],
            'prev_win_curr_loss': [],
            'prev_loss_curr_win': [],
            'prev_loss_curr_loss': []
        }
    }

    time_axis = None
    session_dates = []

    # Process sessions in chronological order
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        # Get photometry data
        session_result = process_session(subject_id, session_date)
        if not session_result:
            continue

        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        session_dates.append(session_date)

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'

        # Get original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Extract data for valid trials
        session_plots = session_result['plotting_data']
        curr_rewards = session_result["reward_outcomes"][non_m_indices]

        # Create previous reward array (shifted)
        # First trial has no previous trial, assign -1 (will be filtered out)
        prev_rewards = np.zeros_like(curr_rewards)
        prev_rewards[0] = -1  # No previous trial for first trial
        prev_rewards[1:] = curr_rewards[:-1]

        # Create trial state list
        trial_states = []
        valid_indices = []

        # For each valid photometry trial, determine its state
        for i, orig_idx in enumerate(session_result["valid_trials"]):
            if orig_idx not in session_result["non_m_trials"]:
                continue

            # Get the trial index in the behavioral data
            if orig_idx < len(non_miss_mask) and non_miss_mask[orig_idx]:
                # Find where this trial appears in the non-missed sequence
                filtered_idx = np.where(orig_non_miss_indices == orig_idx)[0]

                if len(filtered_idx) > 0:
                    # This gives the position of this trial within non-missed trials
                    trial_pos = filtered_idx[0]

                    # Use this position to look up the state in session_df if within bounds
                    if trial_pos < len(session_df):
                        # Get state data from parquet
                        p_stochastic = session_df.iloc[trial_pos]['p_stochastic']
                        p_leftbias = session_df.iloc[trial_pos]['p_leftbias']
                        p_rightbias = session_df.iloc[trial_pos]['p_rightbias']

                        # Determine state based on threshold
                        if p_stochastic >= threshold:
                            state = 'stochastic'
                        elif p_leftbias >= threshold or p_rightbias >= threshold:
                            state = 'biased'
                        else:
                            state = 'uncertain'

                        trial_states.append(state)
                        valid_indices.append(i)
                    else:
                        continue
                else:
                    continue
            else:
                continue

        # Now group by previous and current outcome and state
        for idx, valid_idx in enumerate(valid_indices):
            state = trial_states[idx]
            if state == 'uncertain':
                continue

            # Get the index in the non_m_indices array
            index_in_non_m = np.where(non_m_indices == valid_idx)[0]
            if len(index_in_non_m) == 0:
                continue

            non_m_idx = index_in_non_m[0]

            # Skip first trial (no previous outcome)
            if prev_rewards[non_m_idx] == -1:
                continue

            # Get outcomes
            prev_outcome = prev_rewards[non_m_idx]
            curr_outcome = curr_rewards[non_m_idx]

            # Get photometry data
            photometry_data = session_plots[non_m_idx]

            # Categorize by outcomes
            if prev_outcome == 1 and curr_outcome == 1:
                state_data[state]['prev_win_curr_win'].append(photometry_data)
            elif prev_outcome == 1 and curr_outcome == 0:
                state_data[state]['prev_win_curr_loss'].append(photometry_data)
            elif prev_outcome == 0 and curr_outcome == 1:
                state_data[state]['prev_loss_curr_win'].append(photometry_data)
            elif prev_outcome == 0 and curr_outcome == 0:
                state_data[state]['prev_loss_curr_loss'].append(photometry_data)

    if not session_dates:
        print(f"No processed sessions found for subject {subject_id}")
        return None

    # Convert lists to numpy arrays for each condition in each state
    for state in ['stochastic', 'biased']:
        for condition in state_data[state]:
            if len(state_data[state][condition]) > 0:
                state_data[state][condition] = np.array(state_data[state][condition])

    # Calculate averages and SEM for each condition in each state
    condition_data = {}
    for state in ['stochastic', 'biased']:
        condition_data[state] = {}
        for condition, data in state_data[state].items():
            if isinstance(data, np.ndarray) and len(data) > 0:
                condition_data[state][condition] = {
                    'data': data,
                    'avg': np.mean(data, axis=0),
                    'sem': calculate_sem(data, axis=0),
                    'count': len(data)
                }
            else:
                condition_data[state][condition] = {
                    'data': np.array([]),
                    'avg': None,
                    'sem': None,
                    'count': 0
                }

    # Count trials per state
    stochastic_count = sum(condition_data['stochastic'][c]['count'] for c in condition_data['stochastic'])
    biased_count = sum(condition_data['biased'][c]['count'] for c in condition_data['biased'])
    classified_count = stochastic_count + biased_count
    
    # Count total non-missed trials from all sessions (including uncertain trials)
    total_non_m_trials = 0
    uncertain_count = 0

    for session_date in sessions:
            session_result = process_session(subject_id, session_date)
            if session_result and len(session_result['non_m_trials']) >= 10:
                # Count non-missed trials that weren't first trials 
                # (since we skip first trials due to no previous outcome)
                if len(session_result['non_m_trials']) > 0:
                    total_non_m_trials += len(session_result['non_m_trials']) - 1  # Subtract 1 for first trial
        
        # Calculate uncertain trials (those that didn't meet threshold criteria)
    uncertain_count = total_non_m_trials - classified_count
        
    if total_non_m_trials == 0:
            print("No valid trials found")
            return None
        
        # Calculate percentages based on total non-missed trials
    stochastic_pct = stochastic_count / total_non_m_trials * 100
    biased_pct = biased_count / total_non_m_trials * 100
    uncertain_pct = uncertain_count / total_non_m_trials * 100
        
    print(f"\nFound {stochastic_count} stochastic trials ({stochastic_pct:.1f}%) and "
        f"{biased_count} biased trials ({biased_pct:.1f}%) "
        f"out of {total_non_m_trials} total non-missed trials")
    
    if uncertain_count > 0:
        print(f"Note: {uncertain_count} trials ({uncertain_pct:.1f}%) did not meet the threshold "
              f"criteria of {threshold} for any state and were excluded")

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # Define colors and labels
    colors = {
        'prev_win_curr_win': '#117733',  # green
        'prev_win_curr_loss': '#DDCC77', # yellow
        'prev_loss_curr_win': '#4477AA', # blue
        'prev_loss_curr_loss': '#CC6677' # red
    }

    condition_labels = {
        'prev_win_curr_win': 'Win→Win',
        'prev_win_curr_loss': 'Win→Loss',
        'prev_loss_curr_win': 'Loss→Win',
        'prev_loss_curr_loss': 'Loss→Loss'
    }

    # Find global min/max for consistent y scaling
    y_min = float('inf')
    y_max = float('-inf')

    for state in ['stochastic', 'biased']:
        for condition in condition_data[state]:
            if condition_data[state][condition]['avg'] is not None:
                avg = condition_data[state][condition]['avg']
                sem = condition_data[state][condition]['sem']
                y_min = min(y_min, np.min(avg - sem))
                y_max = max(y_max, np.max(avg + sem))

    # Add some padding to y limits
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Plot each state in its subplot
    for i, state in enumerate(['stochastic', 'biased']):
        ax = axes[i]

        for condition, color in colors.items():
            if condition_data[state][condition]['avg'] is not None:
                avg = condition_data[state][condition]['avg']
                sem = condition_data[state][condition]['sem']
                count = condition_data[state][condition]['count']

                # Only plot if we have data
                if len(avg) > 0 and count > 0:
                    ax.fill_between(time_axis,
                                    avg - sem,
                                    avg + sem,
                                    color=color, alpha=0.3)
                    ax.plot(time_axis, avg,
                            color=color, linewidth=2,
                            label=f'{condition_labels[condition]} (n={count})')

        # Add vertical line at cue onset
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Set the same y-limits for both subplots
        ax.set_ylim(y_min, y_max)

        # Labels and formatting
        ax.set_xlabel('Time (s)', fontsize=12)
        if i == 0:
            ax.set_ylabel('ΔF/F', fontsize=12)

        state_percent = stochastic_pct if state == 'stochastic' else biased_pct
        state_count = stochastic_count if state == 'stochastic' else biased_count
        ax.set_title(f'{state.capitalize()} State ({state_count} trials, {state_percent:.1f}%)', fontsize=14)

        ax.set_xlim([-pre_cue_time, post_cue_time])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)

    # Add main title
    plt.suptitle(f'Previous Outcome Effect by Behavioral State: {subject_id} ({len(session_dates)} sessions)',
                 fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the subtitle

    # Save the figure
    save_figure(fig, subject_id, "pooled", f"previous_outcome_by_state_thresh{threshold}")

    plt.show()
    return {
        'subject_id': subject_id,
        'threshold': threshold,
        'time_axis': time_axis,
        'stochastic_count': stochastic_count,
        'biased_count': biased_count,
        'uncertain_count': uncertain_count,
        'total_non_m_trials': total_non_m_trials,
        'stochastic_pct': stochastic_pct,
        'biased_pct': biased_pct,
        'uncertain_pct': uncertain_pct,
        'condition_data': condition_data,
        'session_dates': session_dates
    }


def analyze_signal_quartiles_by_state(subject_id, signal_window='pre_cue', condition='loss', plot_verification=True, threshold=0.8, behavior_df=None):
    """
    Analyze trials based on photometry signal in specified time window for each behavioral state
    (stochastic, biased) and determine choice switching behavior.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    signal_window : str, optional (default='pre_cue')
        Time window to use for calculating average photometry signal:
        - 'pre_cue': -0.75s to -0.25s before lick (-0.75 to -0.25)
        - 'early_post': +1s to +2s after lick (1 to 2)
        - 'late_post': +3.5s to +4.5s after lick (3.5 to 4.5)
    condition : str, optional (default='loss')
        Which trial outcome to analyze:
        - 'loss': Analyze trials with no reward (0)
        - 'win': Analyze trials with reward (1)
    plot_verification : bool, optional (default=True)
        Whether to create verification plots showing the sorted quartiles
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state

    Returns:
    --------
    dict: Analysis results including quartile switch rates by state
    """
    # Define time windows based on the parameter
    time_windows = {
        'pre_cue': (-0.75, -0.25),
        'early_post': (1.0, 2.0),
        'late_post': (3.5, 4.5)
    }

    if signal_window not in time_windows:
        raise ValueError(f"Invalid signal_window. Choose from: {list(time_windows.keys())}")

    if condition not in ['win', 'loss']:
        raise ValueError("Condition must be either 'win' or 'loss'")

    # Get time window bounds
    window_start, window_end = time_windows[signal_window]

    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])

    # Determine analysis mode based on time window
    is_pre_cue_analysis = signal_window == 'pre_cue'

    # Store trial data and corresponding behavior for each state
    state_data = {
        'stochastic': {
            'trials_data': [],  # Photometry data for selected trials
            'trial_signals': [],  # Average signal in window for trials
            'choice_switches': [],  # Boolean: True if switch in choice
            'next_reward': []  # Boolean: True if next trial rewarded
        },
        'biased': {
            'trials_data': [],
            'trial_signals': [],
            'choice_switches': [],
            'next_reward': []
        }
    }

    time_axis = None  # Will be set from the first valid session

    # Set the reward value we're looking for based on condition
    target_reward = 1 if condition == 'win' else 0

    print(f"Processing {len(sessions)} sessions for {subject_id}")

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        # Process session
        session_result = process_session(subject_id, session_date)
        if not session_result:
            continue

        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        non_miss_rewards = rewards[non_miss_mask]
        non_miss_choices = choices[non_miss_mask]

        # Create mapping from filtered indices to original indices
        non_miss_indices = np.where(non_miss_mask)[0]
        filtered_to_orig = {i: non_miss_indices[i] for i in range(len(non_miss_indices))}

        # Get photometry data for valid trials
        valid_trials = session_result['valid_trials']
        epoched_data = session_result['epoched_data'][valid_trials]

        # Find time indices for the specified window
        window_idx_start = np.where(time_axis >= window_start)[0][0]
        window_idx_end = np.where(time_axis <= window_end)[0][-1]

        if is_pre_cue_analysis:
            # PRE-CUE ANALYSIS MODE
            # Look at trials AFTER a win/loss and analyze if the choice switched from the previous trial

            for i in range(1, len(non_miss_rewards)):  # Start from second trial
                prev_trial_idx = i - 1
                curr_trial_idx = i

                # Check if previous trial matches our target condition (win or loss)
                if non_miss_rewards[prev_trial_idx] != target_reward:
                    continue

                # We found a trial that follows our target condition
                orig_curr_idx = filtered_to_orig[curr_trial_idx]

                # Skip if we don't have photometry data for this trial
                if orig_curr_idx not in valid_trials:
                    continue

                # Get photometry data for this trial
                valid_idx = np.where(np.array(valid_trials) == orig_curr_idx)[0]
                if len(valid_idx) == 0:
                    continue

                curr_photometry = epoched_data[valid_idx[0]]

                # Calculate average signal in the specified window
                window_signal = np.mean(curr_photometry[window_idx_start:window_idx_end])

                # Determine if the current choice is different from previous (switched)
                choice_switched = (non_miss_choices[prev_trial_idx] != non_miss_choices[curr_trial_idx])

                # Get trial state from parquet data
                if i < len(session_df):
                    p_stochastic = session_df.iloc[i]['p_stochastic']
                    p_leftbias = session_df.iloc[i]['p_leftbias']
                    p_rightbias = session_df.iloc[i]['p_rightbias']

                    # Determine state based on threshold
                    if p_stochastic >= threshold:
                        state = 'stochastic'
                    elif p_leftbias >= threshold or p_rightbias >= threshold:
                        state = 'biased'
                    else:
                        # Skip trials without a clear state assignment
                        continue

                    # Store the data in the appropriate state
                    state_data[state]['trials_data'].append(curr_photometry)
                    state_data[state]['trial_signals'].append(window_signal)
                    state_data[state]['choice_switches'].append(choice_switched)
                    state_data[state]['next_reward'].append(non_miss_rewards[curr_trial_idx])  # Current trial's reward

        else:
            # POST-CUE ANALYSIS MODE
            # Look at current win/loss trials and analyze if the next choice switches

            for i in range(len(non_miss_rewards) - 1):  # Exclude the last trial (no next trial)
                curr_trial_idx = i
                next_trial_idx = i + 1

                # Skip if current trial doesn't match our target condition
                if non_miss_rewards[curr_trial_idx] != target_reward:
                    continue

                # We found a trial matching our condition
                orig_curr_idx = filtered_to_orig[curr_trial_idx]

                # Skip if we don't have photometry data for this trial
                if orig_curr_idx not in valid_trials:
                    continue

                # Get photometry data for this trial
                valid_idx = np.where(np.array(valid_trials) == orig_curr_idx)[0]
                if len(valid_idx) == 0:
                    continue

                curr_photometry = epoched_data[valid_idx[0]]

                # Calculate average signal in the specified window
                window_signal = np.mean(curr_photometry[window_idx_start:window_idx_end])

                # Determine if the next choice is different from current (switched)
                choice_switched = (non_miss_choices[curr_trial_idx] != non_miss_choices[next_trial_idx])

                # Get trial state from parquet data
                if i < len(session_df):
                    p_stochastic = session_df.iloc[i]['p_stochastic']
                    p_leftbias = session_df.iloc[i]['p_leftbias']
                    p_rightbias = session_df.iloc[i]['p_rightbias']

                    # Determine state based on threshold
                    if p_stochastic >= threshold:
                        state = 'stochastic'
                    elif p_leftbias >= threshold or p_rightbias >= threshold:
                        state = 'biased'
                    else:
                        # Skip trials without a clear state assignment
                        continue

                    # Store the data in the appropriate state
                    state_data[state]['trials_data'].append(curr_photometry)
                    state_data[state]['trial_signals'].append(window_signal)
                    state_data[state]['choice_switches'].append(choice_switched)
                    state_data[state]['next_reward'].append(non_miss_rewards[next_trial_idx])  # Next trial's reward

    # Create results structure
    results = {
        'subject_id': subject_id,
        'signal_window': signal_window,
        'condition': condition,
        'window_bounds': (window_start, window_end),
        'is_pre_cue_analysis': is_pre_cue_analysis,
        'stochastic': {},
        'biased': {}
    }

    # Process each state's data
    for state in ['stochastic', 'biased']:
        # Check if we found any valid trials
        if len(state_data[state]['trials_data']) == 0:
            print(f"No valid {condition} trials found for {state} state analysis ({signal_window}) for {subject_id}")
            results[state] = None
            continue

        # Convert lists to numpy arrays
        trials_data = np.array(state_data[state]['trials_data'])
        trial_signals = np.array(state_data[state]['trial_signals'])
        choice_switches = np.array(state_data[state]['choice_switches'])
        next_reward = np.array(state_data[state]['next_reward'])

        # Skip if too few trials
        if len(trial_signals) < 10:
            print(f"Skipping {state} state analysis - too few trials ({len(trial_signals)})")
            results[state] = None
            continue

        # Sort trials into quartiles based on signal
        quartile_labels = pd.qcut(trial_signals, 4, labels=False)

        # Calculate switch rate for each quartile
        quartile_switch_rates = []
        quartile_trial_counts = []
        quartile_reward_rates = []

        # Process each quartile
        for quartile in range(4):
            quartile_mask = quartile_labels == quartile
            trials_in_quartile = np.sum(quartile_mask)

            if trials_in_quartile > 0:
                # Calculate switch rate (% of trials where choice switched)
                switch_count = np.sum(choice_switches[quartile_mask])
                switch_rate = (switch_count / trials_in_quartile) * 100

                # Calculate reward rate for the relevant trial
                reward_rate = (np.sum(next_reward[quartile_mask]) / trials_in_quartile) * 100

                quartile_switch_rates.append(switch_rate)
                quartile_trial_counts.append(trials_in_quartile)
                quartile_reward_rates.append(reward_rate)
            else:
                quartile_switch_rates.append(0)
                quartile_trial_counts.append(0)
                quartile_reward_rates.append(0)

        # Store results for this state
        results[state] = {
            'trials_data': trials_data,
            'trial_signals': trial_signals,
            'quartile_labels': quartile_labels,
            'quartile_switch_rates': quartile_switch_rates,
            'quartile_trial_counts': quartile_trial_counts,
            'quartile_reward_rates': quartile_reward_rates,
        }

        # Print results for this state
        outcome_label = "Win" if condition == 'win' else "Loss"
        print(
            f"\n=== Analysis of {signal_window} Signal Quartiles ({outcome_label} trials) - {state.upper()} state: {subject_id} ===")
        print(f"Time window: ({window_start}s to {window_end}s)")
        print(f"Total trials analyzed: {len(trial_signals)}")

        if is_pre_cue_analysis:
            print(f"\nChoice Switch Rates by Pre-Cue Signal Quartile (after {outcome_label.lower()} trials):")
            print(f"(% where T0 choice differs from previous T-1 {outcome_label.lower()} trial)")
        else:
            print(f"\nChoice Switch Rates by Post-Cue Signal Quartile (for {outcome_label.lower()} trials):")
            print(f"(% where next T+1 choice differs from current T0 {outcome_label.lower()} trial)")

        for quartile in range(4):
            print(f"Quartile {quartile + 1}: {quartile_switch_rates[quartile]:.1f}% switch rate "
                  f"({quartile_trial_counts[quartile]} trials)")

        print("\nReward Rates by Signal Quartile:")
        for quartile in range(4):
            print(f"Quartile {quartile + 1}: {quartile_reward_rates[quartile]:.1f}% rewarded")

    # Create verification plot if requested
    if plot_verification:
        # Create a separate plot for each state if both have data
        for state in ['stochastic', 'biased']:
            if results[state] is None:
                continue

            # Create a single plot figure with just the photometry traces
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            colors = ['blue', 'green', 'orange', 'red']  # Colors for quartiles

            # Plot each quartile's average trace
            quartile_labels = results[state]['quartile_labels']
            trials_data = results[state]['trials_data']
            quartile_trial_counts = results[state]['quartile_trial_counts']

            for quartile in range(4):
                quartile_mask = quartile_labels == quartile
                if np.sum(quartile_mask) > 0:
                    quartile_data = trials_data[quartile_mask]
                    quartile_avg = np.mean(quartile_data, axis=0)
                    quartile_sem = calculate_sem(quartile_data, axis=0)

                    ax.fill_between(time_axis,
                                     quartile_avg - quartile_sem,
                                     quartile_avg + quartile_sem,
                                     color=colors[quartile], alpha=0.3)
                    ax.plot(time_axis, quartile_avg,
                             color=colors[quartile], linewidth=2,
                             label=f'Quartile {quartile + 1} (n={quartile_trial_counts[quartile]})')

            # Highlight the time window used for sorting
            ax.axvspan(window_start, window_end, color='gray', alpha=0.3, label='Sorting Window')

            # Add reference lines
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('ΔF/F', fontsize=12)

            if is_pre_cue_analysis:
                ax.set_title(
                    f'Trials After {outcome_label} Sorted by Pre-Cue Signal: {subject_id} - {state.upper()} state',
                    fontsize=14)
            else:
                ax.set_title(
                    f'{outcome_label} Trials Sorted by {signal_window} Signal: {subject_id} - {state.upper()} state',
                    fontsize=14)

            ax.legend(loc='upper right')
            ax.set_xlim([-pre_cue_time, post_cue_time])

            # Save the figure
            condition_str = condition  # 'win' or 'loss'
            mode_suffix = f"after_{condition_str}" if is_pre_cue_analysis else f"{condition_str}_trials"
            save_figure(fig, subject_id, "pooled", f"{mode_suffix}_{signal_window}_quartiles_{state}")

            plt.show()

    # Add statistical analysis if we have enough data for each state
    for state in ['stochastic', 'biased']:
        if results[state] is not None and len(results[state]['trial_signals']) >= 50:
            print(f"\nStatistical analysis for {state.upper()} state:")
            quartile_labels = results[state]['quartile_labels']
            choice_switches = np.array(state_data[state]['choice_switches'])
            quartile_trial_counts = results[state]['quartile_trial_counts']
            quartile_switch_rates = results[state]['quartile_switch_rates']

            # Build contingency table
            contingency_table = []
            for quartile in range(4):
                quartile_mask = quartile_labels == quartile
                trials_in_quartile = np.sum(quartile_mask)

                if trials_in_quartile > 0:
                    switch_count = np.sum(choice_switches[quartile_mask])
                    stay_count = trials_in_quartile - switch_count
                    contingency_table.append([switch_count, stay_count])
                else:
                    contingency_table.append([0, 0])

            contingency_table = np.array(contingency_table)

            # Chi-square test for independence
            try:
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
                print(f"Chi-squared test for independence:")
                print(f"  Chi2 = {chi2:.3f}, p = {p_chi2:.6f}, dof = {dof}")

                if p_chi2 < 0.05:
                    print("  Significant difference in switch rates across quartiles (p < 0.05)")
                else:
                    print("  No significant difference in switch rates across quartiles")

                # Check if any expected frequencies are too low for reliable chi-square
                if np.any(expected < 5):
                    print("  Warning: Some expected frequencies are < 5, chi-square may not be reliable")
                    print("  Consider using Fisher's exact test for pairwise comparisons")
            except Exception as e:
                print(f"  Error in chi-squared test: {e}")

            # Pairwise comparisons between highest and lowest quartiles
            try:
                # Compare quartile 1 (lowest signal) vs quartile 4 (highest signal)
                q1_data = contingency_table[0]
                q4_data = contingency_table[3]

                # Reshape to 2x2 for Fisher's exact test
                fisher_table = np.array([q1_data, q4_data])

                odds_ratio, p_fisher = fisher_exact(fisher_table)
                print(f"\nFisher's exact test: Quartile 1 vs Quartile 4")
                print(f"  Odds ratio = {odds_ratio:.3f}, p = {p_fisher:.6f}")

                if p_fisher < 0.05:
                    print("  Significant difference between quartile 1 and quartile 4 (p < 0.05)")
                    # Report which quartile has higher switch rate
                    if quartile_switch_rates[0] > quartile_switch_rates[3]:
                        print(
                            f"  Quartile 1 has a higher switch rate ({quartile_switch_rates[0]:.1f}%) than Quartile 4 ({quartile_switch_rates[3]:.1f}%)")
                    else:
                        print(
                            f"  Quartile 4 has a higher switch rate ({quartile_switch_rates[3]:.1f}%) than Quartile 1 ({quartile_switch_rates[0]:.1f}%)")
                else:
                    print("  No significant difference between quartile 1 and quartile 4")
            except Exception as e:
                print(f"  Error in Fisher's exact test: {e}")

            print("\nContingency Table (Switch vs Stay for each quartile):")
            print("             Switches  Stays   Total   Switch%")
            for i, (switch, stay) in enumerate(contingency_table):
                total = switch + stay
                percent = (switch / total * 100) if total > 0 else 0
                print(f"Quartile {i + 1}:  {switch:6d}   {stay:6d}   {total:5d}   {percent:.1f}%")

    return results


def analyze_loss_streaks_by_state(subject_id, threshold=0.8, split_biased=False, behavior_df=None):
    """
    Analyze loss streak counts separately for stochastic and biased states.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    split_biased : bool, optional (default=False)
        If True, split biased trials into two groups based on reward rate

    Returns:
    --------
    dict: Analysis results including streak counts for each state
    """
    print(f"Analyzing loss streaks by behavioral state for {subject_id}...")

    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
    
    # Create streak counters for each state
    streak_counts = {
        'stochastic': {
            '1_loss': 0,
            '2_loss': 0,
            '3_loss': 0,
            '4_loss': 0,
            '5plus_loss': 0
        },
        'biased': {
            '1_loss': 0,
            '2_loss': 0,
            '3_loss': 0,
            '4_loss': 0,
            '5plus_loss': 0
        }
    }

    # If splitting biased trials, add high and low reward rate categories
    if split_biased:
        streak_counts['biased_high_reward'] = {
            '1_loss': 0,
            '2_loss': 0,
            '3_loss': 0,
            '4_loss': 0,
            '5plus_loss': 0
        }
        streak_counts['biased_low_reward'] = {
            '1_loss': 0,
            '2_loss': 0,
            '3_loss': 0,
            '4_loss': 0,
            '5plus_loss': 0
        }

    session_level_data = {}

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        # Process session for photometry data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_rewards = rewards[non_miss_mask]
        filtered_choices = choices[non_miss_mask]

        # Get original indices for non-missed trials
        non_miss_indices = np.where(non_miss_mask)[0]

        # Skip if not enough trials after filtering
        if len(filtered_rewards) < 6:
            print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials")
            continue

        # Calculate moving average reward rate if splitting biased trials
        if split_biased:
            window_size = 20
            reward_rates = []
            overall_reward_rate = np.mean(filtered_rewards)

            # Calculate moving average reward rates
            for i in range(len(filtered_rewards)):
                if i < window_size:
                    # For early trials, use available data plus weighted overall rate
                    rewards_so_far = filtered_rewards[:i + 1]
                    available_weight = len(rewards_so_far) / window_size
                    rate = (np.sum(rewards_so_far) / len(rewards_so_far) * available_weight +
                            overall_reward_rate * (1 - available_weight))
                else:
                    # For later trials, use full window
                    window_rewards = filtered_rewards[i - window_size + 1:i + 1]
                    rate = np.mean(window_rewards)
                reward_rates.append(rate)

        # Store session level data for each trial
        session_level_data[session_date] = {
            'states': [],
            'reward_rates': [] if split_biased else None,
            'streaks': []
        }

        # Assign state to each trial
        for i in range(len(session_df)):
            if i < len(session_df):
                p_stochastic = session_df.iloc[i]['p_stochastic']
                p_leftbias = session_df.iloc[i]['p_leftbias']
                p_rightbias = session_df.iloc[i]['p_rightbias']

                # Determine state based on threshold
                if p_stochastic >= threshold:
                    state = 'stochastic'
                elif p_leftbias >= threshold or p_rightbias >= threshold:
                    state = 'biased'
                else:
                    state = 'uncertain'

                session_level_data[session_date]['states'].append(state)

                # Add reward rate if splitting biased trials
                if split_biased and i < len(reward_rates):
                    session_level_data[session_date]['reward_rates'].append(reward_rates[i])

        # Find trials that were losses followed by a win
        for i in range(len(filtered_rewards) - 1):
            # Check if current trial is a loss and next is a win
            if filtered_rewards[i] == 0 and filtered_rewards[i + 1] == 1:
                # This is a loss streak that ends with a win in the next trial

                # Count consecutive losses going backward from current trial
                loss_streak = 1  # Start with 1 (current trial is a loss)

                # Looking back in filtered space (no missed trials)
                for j in range(i - 1, -1, -1):
                    if filtered_rewards[j] == 0:
                        loss_streak += 1
                    else:
                        # Found a win, streak ends
                        break

                # Get the state for this trial
                if i < len(session_level_data[session_date]['states']):
                    state = session_level_data[session_date]['states'][i]

                    # Skip uncertain trials
                    if state == 'uncertain':
                        continue

                    # Get the streak category
                    if loss_streak == 1:
                        streak_cat = '1_loss'
                    elif loss_streak == 2:
                        streak_cat = '2_loss'
                    elif loss_streak == 3:
                        streak_cat = '3_loss'
                    elif loss_streak == 4:
                        streak_cat = '4_loss'
                    else:  # 5 or more consecutive losses
                        streak_cat = '5plus_loss'

                    # Record streak information
                    session_level_data[session_date]['streaks'].append({
                        'position': i,
                        'length': loss_streak,
                        'state': state,
                        'category': streak_cat
                    })

                    # Increment the appropriate counter based on state
                    if state == 'biased' and split_biased and session_level_data[session_date][
                        'reward_rates'] is not None:
                        # Get reward rate for this trial
                        reward_rate = session_level_data[session_date]['reward_rates'][i]

                        # We'll determine high/low based on the median across all biased trials later
                        # Just store the reward rate for now
                        session_level_data[session_date]['streaks'][-1]['reward_rate'] = reward_rate
                    else:
                        # Increment counter for standard state
                        streak_counts[state][streak_cat] += 1

    # If splitting biased trials, determine median reward rate and categorize biased trials
    if split_biased:
        # Collect all reward rates for biased trials
        all_biased_rates = []
        for session_date, data in session_level_data.items():
            for streak in data['streaks']:
                if streak['state'] == 'biased' and 'reward_rate' in streak:
                    all_biased_rates.append(streak['reward_rate'])

        # Calculate median reward rate for biased trials
        if all_biased_rates:
            median_rate = np.median(all_biased_rates)
            print(f"Median reward rate for biased trials: {median_rate:.4f}")

            # Categorize biased trials and update counts
            for session_date, data in session_level_data.items():
                for streak in data['streaks']:
                    if streak['state'] == 'biased' and 'reward_rate' in streak:
                        # Determine if high or low reward rate
                        if streak['reward_rate'] >= median_rate:
                            state_key = 'biased_high_reward'
                        else:
                            state_key = 'biased_low_reward'

                        # Increment the appropriate counter
                        streak_counts[state_key][streak['category']] += 1
        else:
            print("No biased trials with reward rates found")

    # Calculate total trials in each category for normalization
    state_totals = {}
    for state in streak_counts:
        state_totals[state] = sum(streak_counts[state].values())

    # Calculate percentages
    streak_percentages = {}
    for state in streak_counts:
        if state_totals[state] > 0:
            streak_percentages[state] = {
                cat: (count / state_totals[state] * 100)
                for cat, count in streak_counts[state].items()
            }
        else:
            streak_percentages[state] = {cat: 0 for cat in streak_counts[state]}

    # Create visualization
    fig, axes = plt.subplots(1, 2 if not split_biased else 3, figsize=(15, 7), sharey=True)

    # Define x-axis labels and positions
    x_labels = ['1 Loss', '2 Losses', '3 Losses', '4 Losses', '5+ Losses']
    x_pos = np.arange(len(x_labels))

    # Define colors for different states
    colors = {
        'stochastic': 'blue',
        'biased': 'orange',
        'biased_high_reward': 'green',
        'biased_low_reward': 'red'
    }

    # Plot stochastic state
    axes[0].bar(x_pos, [streak_counts['stochastic'][f'{i + 1}_loss'] for i in range(4)] +
                [streak_counts['stochastic']['5plus_loss']],
                color=colors['stochastic'], alpha=0.7)

    # Add counts as text on bars
    for i, count in enumerate([streak_counts['stochastic'][f'{i + 1}_loss'] for i in range(4)] +
                              [streak_counts['stochastic']['5plus_loss']]):
        if count > 0:
            axes[0].text(i, count + max(streak_counts['stochastic'].values()) * 0.05,
                         str(count), ha='center', va='bottom', fontsize=10)

    axes[0].set_title(f'Stochastic State (n={state_totals["stochastic"]})')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Plot biased state(s)
    if not split_biased:
        # Single biased plot
        axes[1].bar(x_pos, [streak_counts['biased'][f'{i + 1}_loss'] for i in range(4)] +
                    [streak_counts['biased']['5plus_loss']],
                    color=colors['biased'], alpha=0.7)

        # Add counts as text on bars
        for i, count in enumerate([streak_counts['biased'][f'{i + 1}_loss'] for i in range(4)] +
                                  [streak_counts['biased']['5plus_loss']]):
            if count > 0:
                axes[1].text(i, count + max(streak_counts['biased'].values()) * 0.05,
                             str(count), ha='center', va='bottom', fontsize=10)

        axes[1].set_title(f'Biased State (n={state_totals["biased"]})')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x_labels)
        axes[1].grid(True, axis='y', alpha=0.3)
    else:
        # Split biased plots
        # High reward rate
        axes[1].bar(x_pos, [streak_counts['biased_high_reward'][f'{i + 1}_loss'] for i in range(4)] +
                    [streak_counts['biased_high_reward']['5plus_loss']],
                    color=colors['biased_high_reward'], alpha=0.7)

        # Add counts as text on bars
        for i, count in enumerate([streak_counts['biased_high_reward'][f'{i + 1}_loss'] for i in range(4)] +
                                  [streak_counts['biased_high_reward']['5plus_loss']]):
            if count > 0:
                axes[1].text(i, count + max(streak_counts['biased_high_reward'].values()) * 0.05,
                             str(count), ha='center', va='bottom', fontsize=10)

        axes[1].set_title(f'Biased State - High Reward (n={state_totals["biased_high_reward"]})')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x_labels)
        axes[1].grid(True, axis='y', alpha=0.3)

        # Low reward rate
        axes[2].bar(x_pos, [streak_counts['biased_low_reward'][f'{i + 1}_loss'] for i in range(4)] +
                    [streak_counts['biased_low_reward']['5plus_loss']],
                    color=colors['biased_low_reward'], alpha=0.7)

        # Add counts as text on bars
        for i, count in enumerate([streak_counts['biased_low_reward'][f'{i + 1}_loss'] for i in range(4)] +
                                  [streak_counts['biased_low_reward']['5plus_loss']]):
            if count > 0:
                axes[2].text(i, count + max(streak_counts['biased_low_reward'].values()) * 0.05,
                             str(count), ha='center', va='bottom', fontsize=10)

        axes[2].set_title(f'Biased State - Low Reward (n={state_totals["biased_low_reward"]})')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(x_labels)
        axes[2].grid(True, axis='y', alpha=0.3)

    # Add overall title
    fig.suptitle(f'Loss Streak Distribution by Behavioral State: {subject_id}', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

    # Save the figure
    save_figure(fig, subject_id, "pooled", f"loss_streak_by_state{'_split_biased' if split_biased else ''}")

    plt.show()

def analyze_loss_streaks_by_state_photometry(subject_id, threshold=0.8, only_1_3=False, plot_verification=True, behavior_df=None, plot_trial='win'):
    """
    Analyze photometry signals during loss streaks separated by behavioral state (stochastic vs biased)

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject, or "All" for cross-subject analysis
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    only_1_3 : bool, optional (default=False)
        If True, group all streaks of 3+ losses together
        If False, maintain separate categories for 3, 4, and 5+ loss streaks
    plot_verification : bool, optional (default=True)
        Whether to create verification plots showing individual trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    plot_trial : str, optional (default='win')
        Which trial to analyze for each streak:
        - 'win': Plot the first win trial after the loss streak
        - 'loss': Plot the last loss trial in the streak

    Returns:
    --------
    dict: Analysis results including streak-specific photometry signals by state
    """
    if subject_id == "All":
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
        
        # Default list of subjects
        specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
        print(f"Using default subject list: {specific_subjects}")
        
        # Container for subject-level results
        subject_results = []
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for loss streak analysis...")
            try:
                # Process individual subject
                subj_result = analyze_loss_streaks_by_state_photometry_single(subj, threshold, only_1_3, 
                                                                             plot_verification=False, 
                                                                             behavior_df=behavior_df,
                                                                             plot_trial=plot_trial)
                if subj_result:
                    subject_results.append(subj_result)
            except Exception as e:
                print(f"Error processing subject {subj}: {e}")
        
        if not subject_results:
            print("No valid subjects found for cross-subject analysis")
            return None
        
        # Define streak categories based on only_1_3 parameter
        if only_1_3:
            streak_categories = ['1_loss', '3plus_loss']
        else:
            streak_categories = ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']
        
        # Store time axis from first subject
        time_axis = None
        for result in subject_results:
            if 'time_axis' in result and result['time_axis'] is not None:
                time_axis = result['time_axis']
                break
        
        if time_axis is None:
            print("No time axis found in subject results")
            return None
        
        # Collect streak averages for each subject by state and category
        cross_subject_data = {
            'stochastic': {cat: [] for cat in streak_categories},
            'biased': {cat: [] for cat in streak_categories}
        }
        
        for result in subject_results:
            if 'streak_averages' not in result:
                continue
                
            for state in ['stochastic', 'biased']:
                for cat in streak_categories:
                    if (cat in result['streak_averages'][state] and 
                        result['streak_averages'][state][cat]['avg'] is not None):
                        cross_subject_data[state][cat].append(result['streak_averages'][state][cat]['avg'])
        
        # Calculate cross-subject averages and SEMs
        cross_subject_averages = {
            'stochastic': {},
            'biased': {}
        }
        
        for state in ['stochastic', 'biased']:
            for cat in streak_categories:
                if cross_subject_data[state][cat]:
                    avg = np.mean(cross_subject_data[state][cat], axis=0)
                    sem = np.std(cross_subject_data[state][cat], axis=0) / np.sqrt(len(cross_subject_data[state][cat]))
                    count = len(cross_subject_data[state][cat])
                    
                    cross_subject_averages[state][cat] = {
                        'avg': avg,
                        'sem': sem,
                        'count': count
                    }
                else:
                    cross_subject_averages[state][cat] = {
                        'avg': None,
                        'sem': None,
                        'count': 0
                    }
        
        # Create the visualization
        if plot_verification:
            # Create a 2x1 subplot figure
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            
            # Define colors
            if only_1_3:
                colors = {
                    '1_loss': 'blue',
                    '2_loss': 'green',
                    '3plus_loss': 'red'
                }
                labels = {
                    '1_loss': '1 Loss',
                    '2_loss': '2 Losses',
                    '3plus_loss': '3+ Losses'
                }
            else:
                colors = {
                    '1_loss': 'blue',
                    '2_loss': 'green',
                    '3_loss': 'orange',
                    '4_loss': 'red',
                    '5plus_loss': 'purple'
                }
                labels = {
                    '1_loss': '1 Loss',
                    '2_loss': '2 Losses',
                    '3_loss': '3 Losses',
                    '4_loss': '4 Losses',
                    '5plus_loss': '5+ Losses'
                }
            
            # Find global min/max for consistent y scaling
            y_min = float('inf')
            y_max = float('-inf')
            
            for state in ['stochastic', 'biased']:
                for cat in streak_categories:
                    if cross_subject_averages[state][cat]['avg'] is not None:
                        avg = cross_subject_averages[state][cat]['avg']
                        sem = cross_subject_averages[state][cat]['sem']
                        y_min = min(y_min, np.min(avg - sem))
                        y_max = max(y_max, np.max(avg + sem))
            
            # Add some padding to y limits
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            # Plot each state in its subplot
            for i, state in enumerate(['stochastic', 'biased']):
                ax = axes[i]
                
                state_subject_count = sum(cross_subject_averages[state][cat]['count'] > 0 
                                         for cat in streak_categories)
                
                for cat in streak_categories:
                    if cross_subject_averages[state][cat]['avg'] is not None:
                        avg = cross_subject_averages[state][cat]['avg']
                        sem = cross_subject_averages[state][cat]['sem']
                        count = cross_subject_averages[state][cat]['count']
                        
                        # Only plot if we have data
                        if len(avg) > 0 and count > 0:
                            ax.fill_between(time_axis,
                                           avg - sem,
                                           avg + sem,
                                           color=colors[cat], alpha=0.3)
                            
                            # Adjust label based on plot_trial parameter
                            if plot_trial == 'win':
                                trial_label = f'{labels[cat]} → Win (n={count} subjects)'
                            else:  # loss
                                trial_label = f'{labels[cat]} (last loss) (n={count} subjects)'
                                
                            ax.plot(time_axis, avg,
                                   color=colors[cat], linewidth=2,
                                   label=trial_label)
                
                # Add vertical line at cue onset
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
                ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Set the same y-limits for both subplots
                ax.set_ylim(y_min, y_max)
                
                # Labels and formatting
                ax.set_xlabel('Time (s)', fontsize=12)
                if i == 0:
                    ax.set_ylabel('ΔF/F', fontsize=12)
                
                ax.set_title(f'{state.capitalize()} State', fontsize=14)
                ax.set_xlim([-pre_cue_time, post_cue_time])
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.2)
            
            # Add main title with appropriate description based on plot_trial
            trial_description = "Following" if plot_trial == 'win' else "During"
            plt.suptitle(
                f'Cross-Subject Photometry Response {trial_description} Loss Streaks by State (n={len(subject_results)} subjects)',
                fontsize=16, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the subtitle
            
            # Save the figure
            suffix = "simple" if only_1_3 else "detailed"
            trial_suffix = "win_after" if plot_trial == 'win' else "last_loss"
            save_figure(fig, "all_subjects", "pooled", f"loss_streak_photometry_by_state_{suffix}_{trial_suffix}")
            
            plt.show()
        
        return {
            'subject_id': 'All',
            'n_subjects': len(subject_results),
            'threshold': threshold,
            'only_1_3': only_1_3,
            'plot_trial': plot_trial,
            'time_axis': time_axis,
            'cross_subject_averages': cross_subject_averages,
            'cross_subject_data': cross_subject_data
        }
    
    else:
        # Call the single-subject analysis
        return analyze_loss_streaks_by_state_photometry_single(subject_id, threshold, only_1_3, plot_verification, behavior_df, plot_trial)


def analyze_loss_streaks_by_state_photometry_single(subject_id, threshold=0.8, only_1_3=False, plot_verification=True, behavior_df=None, plot_trial='win'):
    """
    Analyze photometry signals during loss streaks separated by behavioral state (stochastic vs biased)

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    only_1_3 : bool, optional (default=True)
        If True, group all streaks of 3+ losses together
        If False, maintain separate categories for 3, 4, and 5+ loss streaks
    plot_verification : bool, optional (default=True)
        Whether to create verification plots showing individual trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    plot_trial : str, optional (default='win')
        Which trial to analyze for each streak:
        - 'win': Plot the first win trial after the loss streak
        - 'loss': Plot the last loss trial in the streak

    Returns:
    --------
    dict: Analysis results including streak-specific photometry signals by state
    """
    print(f"Analyzing loss streak photometry signals by state for {subject_id}...")

    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
    
    # Define streak categories based on only_1_3 parameter
    if only_1_3:
        streak_categories = ['1_loss', '2_loss', '3plus_loss']
    else:
        streak_categories = ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']

    # Store photometry data by state and streak length
    streak_data = {
        'stochastic': {cat: [] for cat in streak_categories},
        'biased': {cat: [] for cat in streak_categories}
    }

    time_axis = None
    session_dates = []

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        # Process session for photometry data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        if time_axis is None:
            time_axis = session_result['time_axis']

        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        session_dates.append(session_date)

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_rewards = rewards[non_miss_mask]
        filtered_choices = choices[non_miss_mask]

        # Skip if not enough trials after filtering
        if len(filtered_rewards) < 6:
            print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials")
            continue

        # Get original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Get mapping to valid photometry trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Get state probability for each trial
        trial_states = []
        for i in range(len(session_df)):
            if i < len(session_df):
                p_stochastic = session_df.iloc[i]['p_stochastic']
                p_leftbias = session_df.iloc[i]['p_leftbias']
                p_rightbias = session_df.iloc[i]['p_rightbias']

                # Determine state based on threshold
                if p_stochastic >= threshold:
                    state = 'stochastic'
                elif p_leftbias >= threshold or p_rightbias >= threshold:
                    state = 'biased'
                else:
                    state = 'uncertain'
                trial_states.append(state)

        # Now find loss streaks followed by wins, where all trials are in the same state
        i = 0
        while i < len(filtered_rewards) - 1:  # Exclude last trial
            # Only consider streaks that end with a win
            if filtered_rewards[i] == 0 and i + 1 < len(filtered_rewards) and filtered_rewards[i + 1] == 1:
                # This is potentially the end of a loss streak

                # Count backwards to find streak length
                streak_length = 1  # Start with current loss
                start_idx = i

                # Check state of the current (last loss) trial
                if start_idx >= len(trial_states):
                    i += 1
                    continue

                streak_state = trial_states[start_idx]

                # Skip uncertain state trials
                if streak_state == 'uncertain':
                    i += 1
                    continue

                # Check backwards for consecutive losses in the same state
                for j in range(i - 1, -1, -1):
                    # Stop if we find a win or different state
                    if filtered_rewards[j] != 0 or j >= len(trial_states) or trial_states[j] != streak_state:
                        break
                    streak_length += 1
                    start_idx = j

                # Now we have the loss streak from start_idx to i, all in the same state
                # Get the streak category
                if only_1_3:
                    if streak_length == 1:
                        streak_cat = '1_loss'
                    elif streak_length == 2:
                        streak_cat = '2_loss'
                    else:  # 3 or more consecutive losses
                        streak_cat = '3plus_loss'
                else:
                    if streak_length == 1:
                        streak_cat = '1_loss'
                    elif streak_length == 2:
                        streak_cat = '2_loss'
                    elif streak_length == 3:
                        streak_cat = '3_loss'
                    elif streak_length == 4:
                        streak_cat = '4_loss'
                    else:  # 5 or more consecutive losses
                        streak_cat = '5plus_loss'

                # Depending on the plot_trial parameter, get either the win trial following the streak
                # or the last loss trial in the streak
                if plot_trial == 'win':
                    # Get photometry data for the win trial following the streak
                    win_trial_idx = i + 1

                    # Convert to original index
                    if win_trial_idx < len(orig_non_miss_indices):
                        orig_win_idx = orig_non_miss_indices[win_trial_idx]

                        # Check if this trial has photometry data
                        if orig_win_idx in session_result["valid_trials"]:
                            # Get index in photometry data
                            photo_idx = np.where(np.array(session_result["valid_trials"]) == orig_win_idx)[0]

                            if len(photo_idx) > 0:
                                photo_idx = photo_idx[0]
                                # Get the photometry data
                                trial_photometry = session_result['epoched_data'][photo_idx]

                                # Store in appropriate category
                                streak_data[streak_state][streak_cat].append(trial_photometry)
                else:  # plot_trial == 'loss'
                    # Get photometry data for the last loss trial in the streak
                    loss_trial_idx = i

                    # Convert to original index
                    if loss_trial_idx < len(orig_non_miss_indices):
                        orig_loss_idx = orig_non_miss_indices[loss_trial_idx]

                        # Check if this trial has photometry data
                        if orig_loss_idx in session_result["valid_trials"]:
                            # Get index in photometry data
                            photo_idx = np.where(np.array(session_result["valid_trials"]) == orig_loss_idx)[0]

                            if len(photo_idx) > 0:
                                photo_idx = photo_idx[0]
                                # Get the photometry data
                                trial_photometry = session_result['epoched_data'][photo_idx]

                                # Store in appropriate category
                                streak_data[streak_state][streak_cat].append(trial_photometry)

                # Move past this streak
                i = i + 2  # Move to trial after the win
            else:
                i += 1

    if not session_dates:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Convert all streak data to numpy arrays
    for state in ['stochastic', 'biased']:
        for cat in streak_categories:
            if streak_data[state][cat]:
                streak_data[state][cat] = np.array(streak_data[state][cat])

    # Calculate averages and SEMs
    streak_averages = {}
    for state in ['stochastic', 'biased']:
        streak_averages[state] = {}
        for cat in streak_categories:
            if isinstance(streak_data[state][cat], np.ndarray) and len(streak_data[state][cat]) > 0:
                streak_averages[state][cat] = {
                    'avg': np.mean(streak_data[state][cat], axis=0),
                    'sem': calculate_sem(streak_data[state][cat], axis=0),
                    'count': len(streak_data[state][cat])
                }
            else:
                streak_averages[state][cat] = {
                    'avg': None,
                    'sem': None,
                    'count': 0
                }

    # Create the visualization
    if plot_verification:
        # Create a 2x1 subplot figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        # Define colors
        if only_1_3:
            colors = {
                '1_loss': 'blue',
                '2_loss': 'green',
                '3plus_loss': 'red'
            }
            labels = {
                '1_loss': '1 Loss',
                '2_loss': '2 Losses',
                '3plus_loss': '3+ Losses'
            }
        else:
            colors = {
                '1_loss': 'blue',
                '2_loss': 'green',
                '3_loss': 'orange',
                '4_loss': 'red',
                '5plus_loss': 'purple'
            }
            labels = {
                '1_loss': '1 Loss',
                '2_loss': '2 Losses',
                '3_loss': '3 Losses',
                '4_loss': '4 Losses',
                '5plus_loss': '5+ Losses'
            }

        # Find global min/max for consistent y scaling
        y_min = float('inf')
        y_max = float('-inf')

        for state in ['stochastic', 'biased']:
            for cat in streak_categories:
                if streak_averages[state][cat]['avg'] is not None:
                    avg = streak_averages[state][cat]['avg']
                    sem = streak_averages[state][cat]['sem']
                    y_min = min(y_min, np.min(avg - sem))
                    y_max = max(y_max, np.max(avg + sem))

        # Add some padding to y limits
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Plot each state in its subplot
        for i, state in enumerate(['stochastic', 'biased']):
            ax = axes[i]

            state_trial_count = sum(streak_averages[state][cat]['count'] for cat in streak_categories)

            for cat in streak_categories:
                if only_1_3 and cat == '2_loss':
                    continue

                if streak_averages[state][cat]['avg'] is not None:
                    avg = streak_averages[state][cat]['avg']
                    sem = streak_averages[state][cat]['sem']
                    count = streak_averages[state][cat]['count']

                    # Only plot if we have data
                    if len(avg) > 0 and count > 0:
                        ax.fill_between(time_axis,
                                        avg - sem,
                                        avg + sem,
                                        color=colors[cat], alpha=0.3)
                                        
                        # Adjust label based on plot_trial parameter
                        if plot_trial == 'win':
                            trial_label = f'{labels[cat]} → Win (n={count})'
                        else:  # loss
                            trial_label = f'{labels[cat]} (last loss) (n={count})'
                            
                        ax.plot(time_axis, avg,
                                color=colors[cat], linewidth=2,
                                label=trial_label)

            # Add vertical line at cue onset
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Set the same y-limits for both subplots
            ax.set_ylim(y_min, y_max)

            # Labels and formatting
            ax.set_xlabel('Time (s)', fontsize=12)
            if i == 0:
                ax.set_ylabel('ΔF/F', fontsize=12)

            ax.set_title(f'{state.capitalize()} State ({state_trial_count} total streak trials)', fontsize=14)
            ax.set_xlim([-pre_cue_time, post_cue_time])
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.2)

        # Add main title with appropriate description based on plot_trial
        trial_description = "Following" if plot_trial == 'win' else "During"
        plt.suptitle(
            f'Photometry Response {trial_description} Loss Streaks by State: {subject_id} ({len(session_dates)} sessions)',
            fontsize=16, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the subtitle

        # Save the figure
        suffix = "simple" if only_1_3 else "detailed"
        trial_suffix = "win_after" if plot_trial == 'win' else "last_loss"
        save_figure(fig, subject_id, "pooled", f"loss_streak_photometry_by_state_{suffix}_{trial_suffix}")

        plt.show()

        # Create a separate figure for each state with individual trials
        for state in ['stochastic', 'biased']:
            # Skip if no data for this state
            if sum(streak_averages[state][cat]['count'] for cat in streak_categories) == 0:
                print(f"No data for {state} state, skipping individual trial plot")
                continue

            num_categories = sum(1 for cat in streak_categories if streak_averages[state][cat]['count'] > 0)
            if num_categories == 0:
                continue

            fig, axes = plt.subplots(1, num_categories, figsize=(6 * num_categories, 6), sharey=True)

            # Handle the case where there's only one subplot
            if num_categories == 1:
                axes = [axes]

            # Track which subplot we're on
            subplot_idx = 0

            for cat in streak_categories:
                if streak_averages[state][cat]['count'] == 0:
                    continue

                ax = axes[subplot_idx]
                subplot_idx += 1

                # Get data
                data = streak_data[state][cat]
                avg = streak_averages[state][cat]['avg']
                sem = streak_averages[state][cat]['sem']
                count = streak_averages[state][cat]['count']

                # Plot individual trials with alpha
                for trial in data:
                    ax.plot(time_axis, trial, color=colors[cat], alpha=0.2, linewidth=0.5)

                # Plot average with SEM
                ax.fill_between(time_axis,
                                avg - sem,
                                avg + sem,
                                color=colors[cat], alpha=0.4)
                                
                # Adjust label based on plot_trial parameter
                if plot_trial == 'win':
                    trial_label = f'Average (n={count})'
                    subplot_title = f'{labels[cat]} → Win'
                else:  # loss
                    trial_label = f'Average (n={count})'
                    subplot_title = f'{labels[cat]} (last loss)'
                
                ax.plot(time_axis, avg,
                        color=colors[cat], linewidth=2,
                        label=trial_label)

                # Add vertical line at cue onset
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
                ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

                # Labels and formatting
                ax.set_xlabel('Time (s)', fontsize=12)
                if subplot_idx == 1:
                    ax.set_ylabel('ΔF/F', fontsize=12)

                ax.set_title(subplot_title, fontsize=14)
                ax.set_xlim([-pre_cue_time, post_cue_time])
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.2)

            # Add main title with description based on plot_trial
            trial_description = "Following" if plot_trial == 'win' else "During"
            plt.suptitle(f'{state.capitalize()} State: Individual Trials {trial_description} Loss Streaks: {subject_id}',
                         fontsize=16, y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the subtitle

            # Save the figure
            suffix = "simple" if only_1_3 else "detailed"
            trial_suffix = "win_after" if plot_trial == 'win' else "last_loss"
            save_figure(fig, subject_id, "pooled", f"loss_streak_individual_trials_{state}_{suffix}_{trial_suffix}")

            plt.show()

    return {
        'subject_id': subject_id,
        'only_1_3': only_1_3,
        'threshold': threshold,
        'plot_trial': plot_trial,
        'time_axis': time_axis,
        'streak_data': streak_data,
        'streak_averages': streak_averages,
        'session_dates': session_dates
    }


def analyze_state_transitions(subject_id, window_size=10, min_stable_trials=3, split_bias=False, behavior_df=None):
    """
    Analyzes all state transitions for a subject by identifying every trial that reaches >0.8 probability
    for a state after a period of being in a different state.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    window_size : int, optional (default=10)
        Number of trials before and after transition to analyze
    min_stable_trials : int, optional (default=3)
        Minimum number of consecutive trials required with probability < threshold
        before considering a new transition to that state
    split_bias : bool, optional (default=False)
        If True, separates left and right bias states
        If False, treats them as a single "biased" state

    Returns:
    --------
    dict: Analysis results including state probabilities, reward rates and choice patterns around transitions
    """
    print(f"Analyzing all state transitions for {subject_id}...")

    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])

    # Define state types based on split_bias parameter
    if split_bias:
        state_types = ['stochastic', 'leftbias', 'rightbias']
    else:
        state_types = ['stochastic', 'biased']  # Combined biased state

    # Store data for each state transition
    transitions = {state: {
        'state_probs': [],  # Will store arrays of state probs around t0
        'choices': [],  # Will store arrays of choices around t0
        'rewards': [],  # Will store arrays of rewards around t0
        't0_choices': []  # Will store the choice at t0 for each transition
    } for state in state_types}

    # Define the threshold for state assignment
    threshold = 0.8
    total_transitions = 0
    session_count = 0

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")

        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]

        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue

        session_count += 1

        # Get behavioral data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        non_miss_choices = choices[non_miss_mask]
        non_miss_rewards = rewards[non_miss_mask]

        # Get original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Extract state probabilities as numpy arrays
        if len(session_df) > 0:
            probs = {
                'stochastic': session_df['p_stochastic'].values,
                'leftbias': session_df['p_leftbias'].values,
                'rightbias': session_df['p_rightbias'].values
            }

            # Add combined biased probability if needed
            if not split_bias:
                # Combined probability is the max of leftbias and rightbias for each trial
                probs['biased'] = np.maximum(probs['leftbias'], probs['rightbias'])

            # Track current state and how long it's been stable
            current_state = 'undefined'
            current_state_duration = 0

            # Track when each state was last active above threshold
            last_active = {state: -min_stable_trials - 1 for state in
                           state_types}  # Initialize to allow transitions at start

            # Find all transition points (t0) for each state
            for i in range(len(session_df)):
                # For early trials just track state
                if i < min_stable_trials:
                    # Update tracking of states for early trials
                    # Determine current state
                    new_state = 'uncertain'

                    if probs['stochastic'][i] >= threshold:
                        new_state = 'stochastic'
                        last_active['stochastic'] = i
                    elif not split_bias and probs['biased'][i] >= threshold:
                        new_state = 'biased'
                        last_active['biased'] = i
                    elif split_bias and probs['leftbias'][i] >= threshold:
                        new_state = 'leftbias'
                        last_active['leftbias'] = i
                    elif split_bias and probs['rightbias'][i] >= threshold:
                        new_state = 'rightbias'
                        last_active['rightbias'] = i

                    # Update state tracking
                    if new_state == current_state:
                        current_state_duration += 1
                    else:
                        current_state = new_state
                        current_state_duration = 1

                    continue

                # Process transitions for each state type
                for state_type in state_types:
                    # Check if this trial exceeds threshold for this state
                    if probs[state_type][i] >= threshold:
                        # Check if this is a transition
                        if current_state != state_type and (i - last_active[state_type]) > min_stable_trials:
                            # This is a transition to this state
                            # Will extract data for the available window around t0

                            # Get state probabilities within available window
                            windows = {}

                            # Define which states to include in the windows dict
                            if split_bias:
                                states_to_track = ['stochastic', 'leftbias', 'rightbias']
                            else:
                                states_to_track = ['stochastic', 'biased']

                            for state in states_to_track:
                                # Get as many trials as available within window bounds
                                start_idx = max(0, i - window_size)
                                end_idx = min(len(probs[state]), i + window_size + 1)

                                # Create array of proper length with NaN padding
                                window_data = np.full(2 * window_size + 1, np.nan)

                                # Fill in available data
                                data_slice = probs[state][start_idx:end_idx]
                                offset = window_size - (i - start_idx)  # Calculate offset in window array
                                window_data[offset:offset + len(data_slice)] = data_slice

                                windows[state] = window_data

                            # Store state probability window centered on transition
                            transitions[state_type]['state_probs'].append(windows)

                            # Handle rewards and choices
                            choice_window = [None] * (2 * window_size + 1)
                            reward_window = [None] * (2 * window_size + 1)
                            t0_choice = None

                            # Get available trials within window bounds
                            for rel_pos in range(-window_size, window_size + 1):
                                actual_idx = i + rel_pos
                                window_idx = rel_pos + window_size  # Convert to 0-based index in window array

                                if 0 <= actual_idx < len(session_df):
                                    # Trial exists in session
                                    # Find the corresponding indices in non-missed behavioral data
                                    if actual_idx < len(non_miss_choices):
                                        choice = non_miss_choices[actual_idx]
                                        reward = non_miss_rewards[actual_idx]

                                        choice_window[window_idx] = choice
                                        reward_window[window_idx] = reward

                                        # Record t0 choice
                                        if rel_pos == 0:
                                            t0_choice = choice

                            transitions[state_type]['choices'].append(choice_window)
                            transitions[state_type]['rewards'].append(reward_window)
                            if t0_choice is not None:
                                transitions[state_type]['t0_choices'].append(t0_choice)
                            else:
                                # If for some reason t0 choice is None, use placeholder
                                transitions[state_type]['t0_choices'].append("unknown")

                            total_transitions += 1

                        # Update tracking for this state
                        last_active[state_type] = i

                # Update the current state for this trial
                if probs['stochastic'][i] >= threshold:
                    new_state = 'stochastic'
                elif not split_bias and probs['biased'][i] >= threshold:
                    new_state = 'biased'
                elif split_bias and probs['leftbias'][i] >= threshold:
                    new_state = 'leftbias'
                elif split_bias and probs['rightbias'][i] >= threshold:
                    new_state = 'rightbias'
                else:
                    new_state = 'uncertain'

                # Update state tracking
                if new_state == current_state:
                    current_state_duration += 1
                else:
                    current_state = new_state
                    current_state_duration = 1

    print(f"\nAnalyzed {session_count} sessions, found {total_transitions} state transitions")

    # Check if we found any transitions
    if total_transitions == 0:
        print("No state transitions found. Try adjusting parameters or check the data.")
        return None

    # Process and visualize the results
    results = {
        'subject_id': subject_id,
        'window_size': window_size,
        'state_data': {},
        'transition_counts': {state: len(transitions[state]['state_probs']) for state in state_types}
    }

    # Print transition counts
    print(f"\nTransition counts by state:")
    for state in state_types:
        print(f"  {state.capitalize()}: {results['transition_counts'][state]}")

    # Define x-axis for plots (centered at t0)
    x_range = list(range(-window_size, window_size + 1))

    # Process each state transition type
    for state_type in state_types:
        if len(transitions[state_type]['state_probs']) == 0:
            print(f"No transitions found for {state_type} state")
            continue

        print(f"\nProcessing {len(transitions[state_type]['state_probs'])} transitions to {state_type} state")

        # Average state probabilities around transition
        avg_probs = {state: np.zeros(2 * window_size + 1) for state in state_types}

        # Count valid data points at each position to handle NaNs
        valid_counts = np.zeros((len(state_types), 2 * window_size + 1))
        state_idx = {state: i for i, state in enumerate(state_types)}

        # Sum up probabilities (skipping NaNs)
        for windows in transitions[state_type]['state_probs']:
            for prob_type in windows:
                # Mask for non-NaN values
                valid_mask = ~np.isnan(windows[prob_type])

                # Add to the sum only for non-NaN positions
                avg_probs[prob_type][valid_mask] += windows[prob_type][valid_mask]

                # Count valid positions
                valid_counts[state_idx[prob_type], valid_mask] += 1

        # Average (dividing by the number of valid data points at each position)
        for prob_type in avg_probs:
            idx = state_idx[prob_type]
            # Avoid division by zero by setting count=1 where count=0
            counts = np.maximum(valid_counts[idx], 1)
            avg_probs[prob_type] = avg_probs[prob_type] / counts

            # Handle positions with no valid data (set to NaN)
            avg_probs[prob_type][valid_counts[idx] == 0] = np.nan

        # Calculate reward percentages by position
        reward_pcts = np.zeros(2 * window_size + 1)
        reward_counts = np.zeros(2 * window_size + 1)

        for reward_window in transitions[state_type]['rewards']:
            for i, reward in enumerate(reward_window):
                if reward is not None:  # Skip None values
                    reward_pcts[i] += reward  # 1 for win, 0 for loss
                    reward_counts[i] += 1

        # Calculate percentage
        reward_pcts = np.divide(reward_pcts, reward_counts, out=np.zeros_like(reward_pcts),
                                where=reward_counts != 0) * 100

        # Calculate choice consistency with t0
        choice_match_pcts = np.zeros(2 * window_size + 1)
        choice_counts = np.zeros(2 * window_size + 1)

        for choice_window, t0_choice in zip(transitions[state_type]['choices'], transitions[state_type]['t0_choices']):
            if t0_choice is None or t0_choice == "unknown":
                continue

            # Middle position is t0 (should always match itself)
            middle_idx = window_size

            for i, choice in enumerate(choice_window):
                if choice is not None and choice != 'M':  # Skip None and missed trials
                    # Check if choice matches t0 choice
                    choice_match_pcts[i] += (choice == t0_choice)
                    choice_counts[i] += 1

        # Calculate percentage
        choice_match_pcts = np.divide(choice_match_pcts, choice_counts, out=np.zeros_like(choice_match_pcts),
                                      where=choice_counts != 0) * 100

        # Store processed data
        results['state_data'][state_type] = {
            'avg_probs': avg_probs,
            'reward_pcts': reward_pcts,
            'choice_match_pcts': choice_match_pcts,
            'valid_counts': valid_counts
        }

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle(
            f"Transitions to {state_type.upper()} State (n={len(transitions[state_type]['state_probs'])}) - {subject_id}",
            fontsize=16)

        # 1. Plot state probabilities
        ax1 = plt.subplot(3, 1, 1)

        # Define colors for each state probability line
        if split_bias:
            colors = {
                'stochastic': 'green',
                'leftbias': 'red',
                'rightbias': 'blue'
            }
        else:
            colors = {
                'stochastic': 'green',
                'biased': 'red'
            }

        # Plot each state probability line, handling NaNs
        for prob_type, color in colors.items():
            valid_mask = ~np.isnan(avg_probs[prob_type])
            if np.any(valid_mask):  # Only plot if there's any valid data
                x_valid = np.array(x_range)[valid_mask]
                y_valid = avg_probs[prob_type][valid_mask]
                ax1.plot(x_valid, y_valid, color=color, linewidth=2.5, label=prob_type.capitalize())

        # Highlight t0 with vertical black dotted line
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='t0 (Transition Trial)')

        # Highlight threshold with horizontal gray dotted line
        ax1.axhline(y=threshold, color='gray', linestyle=':', alpha=0.7, label=f'Threshold ({threshold})')

        ax1.set_ylabel('State Probability', fontsize=12)
        ax1.set_title(f'Average State Probabilities Around Transition', fontsize=14)
        ax1.set_xticks(range(-window_size, window_size + 1, 2))
        ax1.set_xticklabels([f't{x}' for x in range(-window_size, window_size + 1, 2)])
        ax1.set_xlim([-window_size, window_size])
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # 2. Plot reward rates
        ax2 = plt.subplot(3, 1, 2)

        # Plot reward rates, only for positions with data
        valid_mask = reward_counts > 0
        if np.any(valid_mask):
            x_valid = np.array(x_range)[valid_mask]
            y_valid = reward_pcts[valid_mask]
            ax2.plot(x_valid, y_valid, 'o-', color='purple', linewidth=2.5, label='Reward Rate')

        # Highlight t0 with vertical black dotted line
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='t0 (Transition Trial)')

        # Highlight 50% line with thin black line
        ax2.axhline(y=50, color='black', linestyle='-', linewidth=0.8, alpha=0.5, label='50% Reward')

        ax2.set_ylabel('Reward Rate (%)', fontsize=12)
        ax2.set_title(f'Reward Rates Around Transition', fontsize=14)
        ax2.set_xticks(range(-window_size, window_size + 1, 2))
        ax2.set_xticklabels([f't{x}' for x in range(-window_size, window_size + 1, 2)])
        ax2.set_xlim([-window_size, window_size])
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # 3. Plot choice consistency
        ax3 = plt.subplot(3, 1, 3)

        # Plot choice consistency, only for positions with data
        valid_mask = choice_counts > 0
        if np.any(valid_mask):
            x_valid = np.array(x_range)[valid_mask]
            y_valid = choice_match_pcts[valid_mask]
            ax3.plot(x_valid, y_valid, 'o-', color='orange', linewidth=2.5, label='Choice Match Rate')

        # Highlight t0 with vertical black dotted line
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='t0 (Transition Trial)')

        # Highlight 50% line with thin purple line
        ax3.axhline(y=50, color='purple', linestyle='-', linewidth=0.8, alpha=0.5, label='50% Match')

        ax3.set_xlabel('Trial Position Relative to Transition (t0)', fontsize=12)
        ax3.set_ylabel('Choice Match Rate (%)', fontsize=12)
        ax3.set_title(f'Percentage of Trials with Same Choice as t0', fontsize=14)
        ax3.set_xticks(range(-window_size, window_size + 1, 2))
        ax3.set_xticklabels([f't{x}' for x in range(-window_size, window_size + 1, 2)])
        ax3.set_xlim([-window_size, window_size])
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle

        # Save the figure
        save_figure(fig, subject_id, "pooled", f"transition_to_{state_type}_state")

        plt.show()

    # Print additional statistics
    print("\nDetailed statistics for each transition type:")

    for state_type in state_types:
        if len(transitions[state_type]['state_probs']) == 0:
            continue

        print(f"\n{state_type.upper()} STATE TRANSITIONS (n={len(transitions[state_type]['state_probs'])}):")

        # Reward rates at key positions
        reward_pcts = results['state_data'][state_type]['reward_pcts']
        reward_counts = np.zeros(2 * window_size + 1)
        for reward_window in transitions[state_type]['rewards']:
            for i, reward in enumerate(reward_window):
                if reward is not None:
                    reward_counts[i] += 1

        print("\nReward rates at key positions (n = number of valid trials):")
        for pos in [-3, -2, -1, 0, 1, 2, 3]:
            idx = window_size + pos
            if idx >= 0 and idx < len(reward_pcts) and reward_counts[idx] > 0:
                print(f"  t{pos}: {reward_pcts[idx]:.1f}% (n={int(reward_counts[idx])})")
            else:
                print(f"  t{pos}: No data")

        # Choice consistency with t0
        choice_match_pcts = results['state_data'][state_type]['choice_match_pcts']
        choice_counts = np.zeros(2 * window_size + 1)
        for choice_window, t0_choice in zip(transitions[state_type]['choices'], transitions[state_type]['t0_choices']):
            if t0_choice is None or t0_choice == "unknown":
                continue
            for i, choice in enumerate(choice_window):
                if choice is not None and choice != 'M':
                    choice_counts[i] += 1

        print("\nChoice match rates with t0 (n = number of valid trials):")
        for pos in [-3, -2, -1, 0, 1, 2, 3]:
            idx = window_size + pos
            if idx >= 0 and idx < len(choice_match_pcts) and choice_counts[idx] > 0:
                print(f"  t{pos}: {choice_match_pcts[idx]:.1f}% (n={int(choice_counts[idx])})")
            else:
                print(f"  t{pos}: No data")

    
    print("\n=== State Transition Speed Analysis ===")
    
    # Create dictionaries to store transition speed data
    transition_speeds = {}
    previous_states = {}
    
    # Initialize data structures based on state types
    for target_state in state_types:
        transition_speeds[target_state] = []
        previous_states[target_state] = {source: [] for source in state_types if source != target_state}
        if not split_bias and target_state != 'uncertain':
            previous_states[target_state]['uncertain'] = []
    
    # Process each session to track transition speeds
    for session_date in sessions:
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]
        
        if session_df.empty:
            continue
        
        # Extract state probabilities
        session_probs = {
            'stochastic': session_df['p_stochastic'].values,
            'leftbias': session_df['p_leftbias'].values,
            'rightbias': session_df['p_rightbias'].values
        }
        
        # Add combined biased probability if needed
        if not split_bias:
            session_probs['biased'] = np.maximum(session_probs['leftbias'], session_probs['rightbias'])
        
        # Determine state for each trial in session
        trial_states = []
        for i in range(len(session_df)):
            if session_probs['stochastic'][i] >= threshold:
                state = 'stochastic'
            elif not split_bias and session_probs['biased'][i] >= threshold:
                state = 'biased'
            elif split_bias and session_probs['leftbias'][i] >= threshold:
                state = 'leftbias'
            elif split_bias and session_probs['rightbias'][i] >= threshold:
                state = 'rightbias'
            else:
                state = 'uncertain'
            trial_states.append(state)
        
        # Find transitions
        for i in range(1, len(trial_states)):
            current_state = trial_states[i]
            
            # Skip uncertain states
            if current_state == 'uncertain':
                continue
            
            # Check if this is a new state (transition)
            if current_state != trial_states[i-1]:
                previous_state = trial_states[i-1]
                
                # Look backwards to find how far back different state was above threshold
                transition_distance = 1  # Start with 1 (previous trial)
                last_above_threshold_idx = i - 1
                
                # Count backward until we find a different state above threshold
                found_different_state = False
                
                # We're looking for last trial where any other state was above threshold
                for j in range(i-2, -1, -1):
                    # If we find a trial where current_state was already above threshold,
                    # this means we're just coming back to the same state after a brief dip
                    # We should ignore such cases for calculating genuine transition speeds
                    if trial_states[j] == current_state:
                        break
                    
                    # We're looking for last trial where a different state was above threshold
                    if trial_states[j] != 'uncertain' and trial_states[j] != current_state:
                        last_above_threshold_idx = j
                        found_different_state = True
                        break
                    
                    transition_distance += 1
                
                # Only count if we actually found a different state before
                if found_different_state:
                    transition_speeds[current_state].append(transition_distance)
                    
                    # Track which specific state we transitioned from
                    if previous_state != 'uncertain':
                        previous_states[current_state][previous_state].append(transition_distance)
    
    # Print transition speed analysis
    print("\nAverage trials between state transitions:")
    for target_state in state_types:
        if transition_speeds[target_state]:
            avg_speed = np.mean(transition_speeds[target_state])
            med_speed = np.median(transition_speeds[target_state])
            count = len(transition_speeds[target_state])
            print(f"  TO {target_state.upper()}: {avg_speed:.2f} trials avg, {med_speed:.1f} median (n={count})")
            
            # Print specific source state transition speeds if we have enough data
            print(f"    Broken down by previous state:")
            for source in previous_states[target_state]:
                if previous_states[target_state][source]:
                    source_avg = np.mean(previous_states[target_state][source])
                    source_med = np.median(previous_states[target_state][source])
                    source_count = len(previous_states[target_state][source])
                    print(f"      FROM {source.upper()}: {source_avg:.2f} trials avg, {source_med:.1f} median (n={source_count})")
    
    # Create transition speed distributions plot
    if sum(len(speeds) for speeds in transition_speeds.values()) > 0:
        fig, axes = plt.subplots(1, len(state_types), figsize=(5*len(state_types), 5), sharey=True)
        
        if len(state_types) == 1:
            axes = [axes]  # Make axes iterable if only one subplot
        
        for i, state in enumerate(state_types):
            if not transition_speeds[state]:
                axes[i].text(0.5, 0.5, f"No transitions to {state}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
                
            # Plot histogram of transition speeds
            axes[i].hist(transition_speeds[state], bins=range(1, max(transition_speeds[state])+2),
                       alpha=0.7, color='blue', edgecolor='black')
            axes[i].axvline(np.mean(transition_speeds[state]), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(transition_speeds[state]):.2f}')
            axes[i].axvline(np.median(transition_speeds[state]), color='green', linestyle='-', 
                          linewidth=2, label=f'Median: {np.median(transition_speeds[state]):.1f}')
            
            axes[i].set_title(f'Transition Speed to {state.capitalize()}')
            axes[i].set_xlabel('Trials Since Previous State Above Threshold')
            if i == 0:
                axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.suptitle(f'State Transition Speed Distributions: {subject_id}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        save_figure(fig, subject_id, "pooled", "state_transition_speeds")
        plt.show()
    
    return results


def analyze_reward_rate_distribution_by_state(subject_id, window_size=20, threshold=0.8, behavior_df=None):
    """
    Analyze the distribution of reward rates by behavioral state (stochastic vs biased).
    
    For each session, calculate a moving average reward rate for each trial using a window size.
    For early trials, weight in the overall session average. Group trials by their behavioral state
    and plot histograms of reward rate distributions for comparison.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    window_size : int, optional (default=20)
        Size of moving window for calculating reward rates
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
        
    Returns:
    --------
    dict: Analysis results including reward rate distributions by state
    """
    print(f"Analyzing reward rate distribution by behavioral state for {subject_id}...")
    
    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])

    # Store trials by state with their reward rates
    state_reward_rates = {
        'stochastic': [],
        'biased': []
    }
    
    # Track trials that will be excluded from the final calculation (first 10)
    exclude_first_n = 10
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]
        
        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue
        
        # Get photometry data (primarily for behavioral information)
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue
        
        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue
        
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_rewards = rewards[non_miss_mask]
        
        # Skip if not enough trials after filtering
        if len(filtered_rewards) < window_size:
            print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials")
            continue
        
        # Calculate overall session reward rate (for weighting early trials)
        overall_reward_rate = np.mean(filtered_rewards)
        
        # Calculate moving average reward rates for each trial
        reward_rates = []
        
        for i in range(len(filtered_rewards)):
            if i < window_size:
                # For early trials, use available data plus weighted overall rate
                rewards_so_far = filtered_rewards[:i + 1]
                available_weight = len(rewards_so_far) / window_size
                rate = (np.mean(rewards_so_far) * available_weight +
                        overall_reward_rate * (1 - available_weight))
            else:
                # For later trials, use full window
                window_rewards = filtered_rewards[i - window_size + 1:i + 1]
                rate = np.mean(window_rewards)
            
            reward_rates.append(rate)
        
        # Match state probabilities to filtered trials
        trial_states = []
        
        # For each trial, determine its state
        for i in range(len(session_df)):
            if i < len(session_df):
                p_stochastic = session_df.iloc[i]['p_stochastic']
                p_leftbias = session_df.iloc[i]['p_leftbias']
                p_rightbias = session_df.iloc[i]['p_rightbias']
                
                # Determine state based on threshold
                if p_stochastic >= threshold:
                    state = 'stochastic'
                elif p_leftbias >= threshold or p_rightbias >= threshold:
                    state = 'biased'
                else:
                    state = 'uncertain'
                
                trial_states.append(state)
            else:
                trial_states.append('uncertain')
        
        # Skip the first N trials as specified
        for i in range(exclude_first_n, len(reward_rates)):
            if i < len(trial_states):
                state = trial_states[i]
                
                # Only include trials with clear state assignment (not uncertain)
                if state == 'stochastic' or state == 'biased':
                    state_reward_rates[state].append(reward_rates[i])
    
    # Print the unique reward rate values to understand the distribution
    unique_values = {}
    for state in ['stochastic', 'biased']:
        if state_reward_rates[state]:
            # Round to 4 decimal places to account for floating point imprecision
            rounded_rates = np.round(state_reward_rates[state], 4)
            unique = np.unique(rounded_rates)
            unique_values[state] = sorted(unique)
            print(f"\nUnique reward rate values for {state}: {unique}")
    
    # Calculate statistics
    stats = {}
    for state in ['stochastic', 'biased']:
        if state_reward_rates[state]:
            rates = state_reward_rates[state]
            stats[state] = {
                'count': len(rates),
                'mean': np.mean(rates),
                'median': np.median(rates),
                'std': np.std(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        else:
            stats[state] = {
                'count': 0,
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None
            }
    
    # Calculate biased high/low split if we have biased trials
    biased_rates = state_reward_rates['biased']
    biased_median = None
    biased_high_rates = []
    biased_low_rates = []
    
    if len(biased_rates) > 0:
        biased_median = np.median(biased_rates)
        biased_high_rates = [r for r in biased_rates if r >= biased_median]
        biased_low_rates = [r for r in biased_rates if r < biased_median]
        
        # Add statistics for high/low biased subgroups
        stats['biased_high'] = {
            'count': len(biased_high_rates),
            'mean': np.mean(biased_high_rates) if len(biased_high_rates) > 0 else None,
            'median': np.median(biased_high_rates) if len(biased_high_rates) > 0 else None,
            'std': np.std(biased_high_rates) if len(biased_high_rates) > 0 else None,
            'min': np.min(biased_high_rates) if len(biased_high_rates) > 0 else None,
            'max': np.max(biased_high_rates) if len(biased_high_rates) > 0 else None
        }
        
        stats['biased_low'] = {
            'count': len(biased_low_rates),
            'mean': np.mean(biased_low_rates) if len(biased_low_rates) > 0 else None,
            'median': np.median(biased_low_rates) if len(biased_low_rates) > 0 else None,
            'std': np.std(biased_low_rates) if len(biased_low_rates) > 0 else None,
            'min': np.min(biased_low_rates) if len(biased_low_rates) > 0 else None,
            'max': np.max(biased_low_rates) if len(biased_low_rates) > 0 else None
        }
    
    # Print statistics
    print("\nReward Rate Distribution Statistics:")
    for state in ['stochastic', 'biased']:
        print(f"\n{state.upper()} State (n={stats[state]['count']} trials):")
        if stats[state]['count'] > 0:
            print(f"  Mean: {stats[state]['mean']:.3f}")
            print(f"  Median: {stats[state]['median']:.3f}")
            print(f"  Std Dev: {stats[state]['std']:.3f}")
            print(f"  Range: {stats[state]['min']:.3f} - {stats[state]['max']:.3f}")
    
    if biased_median is not None:
        print("\nBiased State Split at Median:")
        print(f"  Median Value: {biased_median:.3f}")
        print(f"  Biased High (≥{biased_median:.3f}): {len(biased_high_rates)} trials")
        print(f"  Biased Low (<{biased_median:.3f}): {len(biased_low_rates)} trials")
    
    # Create histogram visualization
    if sum(len(rates) for rates in state_reward_rates.values()) > 0:
        plt.figure(figsize=(12, 7))
        
        # Determine appropriate bins based on window size
        # For window_size=20, possible values are multiples of 0.05 (1/20)
        bin_step = 0.1/3
        
        # Create bins that center on the possible values
        # Add a small offset to ensure each possible value falls into a bin
        offset = bin_step / 3
        bin_edges = np.arange(0 - offset, 1.0 + bin_step + offset, bin_step)
        
        # Plot histograms with reduced alpha for better transparency
        # First stochastic state
        if len(state_reward_rates['stochastic']) > 0:
            plt.hist(state_reward_rates['stochastic'], bins=bin_edges, alpha=0.4, 
                    label=f"Stochastic (n={len(state_reward_rates['stochastic'])})",
                    color='green', edgecolor='black', linewidth=1)
        
        # Then biased state, split into high and low reward rate
        if biased_median is not None:
            # Plot biased low (red)
            if len(biased_low_rates) > 0:
                plt.hist(biased_low_rates, bins=bin_edges, alpha=0.4, 
                        label=f"Biased Low (<{biased_median:.3f}, n={len(biased_low_rates)})",
                        color='red', edgecolor='black', linewidth=1)
            
            # Plot biased high (blue)
            if len(biased_high_rates) > 0:
                plt.hist(biased_high_rates, bins=bin_edges, alpha=0.4, 
                        label=f"Biased High (≥{biased_median:.3f}, n={len(biased_high_rates)})",
                        color='blue', edgecolor='black', linewidth=1)
        else:
            # If no median calculated, just plot all biased trials in purple
            if len(state_reward_rates['biased']) > 0:
                plt.hist(state_reward_rates['biased'], bins=bin_edges, alpha=0.4, 
                        label=f"Biased (n={len(state_reward_rates['biased'])})",
                        color='purple', edgecolor='black', linewidth=1)
        if biased_median is not None:
            plt.axvline(biased_median, color='grey', linestyle='--', linewidth=1.5, alpha=1)
        # Add labels and title
        plt.xlabel('Current Reward Rate', fontsize=24)
        plt.ylabel('Trial Count', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f'Reward Rate Distribution by Behavioral State', fontsize=26)
        
        # Set x-axis ticks at 0.1 intervals only
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.xlim([0, 1])
        
        # Add legend
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        
        # Tighten layout
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), subject_id, "pooled", "reward_rate_distribution_by_state")
        
        plt.show()
    
    return {
        'subject_id': subject_id,
        'window_size': window_size,
        'threshold': threshold,
        'excluded_trials': exclude_first_n,
        'state_reward_rates': state_reward_rates,
        'stats': stats,
        'unique_values': unique_values
    }

def analyze_stay_switch_photometry(subject_id="All", state=False, behavior_df=None, specific_subjects=None, threshold=0.8):
    """
    Analyze photometry signals for trials where the current choice is the same as (stay)
    or different from (switch) the previous trial's choice, further subdivided by reward outcome.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject or "All" for cross-subject analysis
    state : bool, optional (default=False)
        If True, further subdivides the data by behavioral state (stochastic/biased)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state when state=True
        
    Returns:
    --------
    dict: Analysis results including photometry signals by stay/switch and reward outcome
    """
    if subject_id == "All":
        # Cross-subject analysis
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            
        # Default list of subjects
        if specific_subjects is None:
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store subject-level averages for each condition
        subject_condition_avgs = {
            'switch_rewarded': [],
            'switch_unrewarded': [],
            'stay_rewarded': [],
            'stay_unrewarded': []
        }
        
        # If state=True, also store state-specific averages
        state_subject_avgs = {}
        if state:
            state_subject_avgs = {
                'stochastic': {
                    'switch_rewarded': [],
                    'switch_unrewarded': [],
                    'stay_rewarded': [],
                    'stay_unrewarded': []
                },
                'biased': {
                    'switch_rewarded': [],
                    'switch_unrewarded': [],
                    'stay_rewarded': [],
                    'stay_unrewarded': []
                }
            }
        
        time_axis = None
        valid_subjects = []
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for stay/switch analysis...")
            
            # Process individual subject
            subj_result = analyze_stay_switch_photometry_single(subj, state, behavior_df, threshold)
            
            if subj_result:
                if time_axis is None:
                    time_axis = subj_result['time_axis']
                
                valid_subjects.append(subj)
                
                # Add each condition's average to the subject-level collection
                for condition in ['switch_rewarded', 'switch_unrewarded', 'stay_rewarded', 'stay_unrewarded']:
                    if subj_result['condition_data'][condition]['avg'] is not None:
                        subject_condition_avgs[condition].append(subj_result['condition_data'][condition]['avg'])
                
                # Add state-specific averages if state=True
                if state and 'state_data' in subj_result:
                    for state_type in ['stochastic', 'biased']:
                        if state_type in subj_result['state_data']:
                            for condition in ['switch_rewarded', 'switch_unrewarded', 'stay_rewarded', 'stay_unrewarded']:
                                if (subj_result['state_data'][state_type][condition]['avg'] is not None and
                                    len(subj_result['state_data'][state_type][condition]['avg']) > 0):
                                    state_subject_avgs[state_type][condition].append(
                                        subj_result['state_data'][state_type][condition]['avg'])
        
        if not valid_subjects:
            print("No valid subjects found for cross-subject analysis")
            return None
        
        # Create figures based on the data we have
        if time_axis is None or not any(subject_condition_avgs.values()):
            print("No valid data found for cross-subject analysis")
            return None
        
        # Create main plot for all conditions
        plt.figure(figsize=(12, 7))
        
        # Define colors and labels
        colors = {
            'switch_rewarded': 'blue',
            'switch_unrewarded': 'lightblue',
            'stay_rewarded': 'red',
            'stay_unrewarded': 'lightsalmon'
        }
        
        labels = {
            'switch_rewarded': 'Switch - Rewarded',
            'switch_unrewarded': 'Switch - Unrewarded',
            'stay_rewarded': 'Stay - Rewarded',
            'stay_unrewarded': 'Stay - Unrewarded'
        }
        
        # Calculate and plot the cross-subject average for each condition
        for condition, color in colors.items():
            if len(subject_condition_avgs[condition]) > 0:
                # Calculate mean and SEM across subjects
                condition_mean = np.mean(subject_condition_avgs[condition], axis=0)
                condition_sem = np.std(subject_condition_avgs[condition], axis=0) / np.sqrt(len(subject_condition_avgs[condition]))
                
                plt.fill_between(time_axis,
                               condition_mean - condition_sem,
                               condition_mean + condition_sem,
                               color=color, alpha=0.3)
                plt.plot(time_axis, condition_mean,
                       color=color, linewidth=2, 
                       label=f'{labels[condition]} (n={len(subject_condition_avgs[condition])} subjects)')
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Labels and formatting
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('z-ΔF/F', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f'LC Signal by Stay/Switch and Outcome', 
                 fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])
        #plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", "stay_switch_all_conditions")
        plt.show()
        
        # If state=True, create additional plots for each state and condition
        if state and all(any(state_subject_avgs[s][c] for c in state_subject_avgs[s]) for s in state_subject_avgs):
            # 1. Create one plot per state showing all 4 conditions
            for state_type in ['stochastic', 'biased']:
                plt.figure(figsize=(12, 7))
                
                for condition, color in colors.items():
                    if len(state_subject_avgs[state_type][condition]) > 0:
                        # Calculate mean and SEM across subjects
                        condition_mean = np.mean(state_subject_avgs[state_type][condition], axis=0)
                        condition_sem = np.std(state_subject_avgs[state_type][condition], axis=0) / np.sqrt(len(state_subject_avgs[state_type][condition]))
                        
                        plt.fill_between(time_axis,
                                        condition_mean - condition_sem,
                                        condition_mean + condition_sem,
                                        color=color, alpha=0.3)
                        plt.plot(time_axis, condition_mean,
                                color=color, linewidth=2, 
                                label=f'{labels[condition]} (n={len(state_subject_avgs[state_type][condition])} subjects)')
                
                # Add vertical line at cue onset
                plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                plt.xlabel('Time (s) after first lick', fontsize=24)
                plt.ylabel('z-ΔF/F', fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.title(f'{state_type.capitalize()} State: LC Signal by Stay/Switch and Outcome (All Subjects)', 
                        fontsize=26)
                plt.xlim([-pre_cue_time, post_cue_time])
                #plt.legend(loc='upper right', fontsize=12)
                plt.tight_layout()
                
                # Save the figure
                save_figure(plt.gcf(), "all_subjects", "pooled", f"stay_switch_all_conditions_{state_type}")
                plt.show()
            
            # 2. Create one plot per condition showing both states
            for condition in ['switch_rewarded', 'switch_unrewarded', 'stay_rewarded', 'stay_unrewarded']:
                plt.figure(figsize=(12, 7))
                
                for state_type, color in [('stochastic', 'green'), ('biased', 'purple')]:
                    if len(state_subject_avgs[state_type][condition]) > 0:
                        # Calculate mean and SEM across subjects
                        state_mean = np.mean(state_subject_avgs[state_type][condition], axis=0)
                        state_sem = np.std(state_subject_avgs[state_type][condition], axis=0) / np.sqrt(len(state_subject_avgs[state_type][condition]))
                        
                        plt.fill_between(time_axis,
                                        state_mean - state_sem,
                                        state_mean + state_sem,
                                        color=color, alpha=0.3)
                        plt.plot(time_axis, state_mean,
                                color=color, linewidth=2, 
                                label=f'{state_type.capitalize()} (n={len(state_subject_avgs[state_type][condition])} subjects)')
                
                # Add vertical line at cue onset
                plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
                plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                plt.xlabel('Time (s) after first lick', fontsize=24)
                plt.ylabel('z-ΔF/F', fontsize=24)
                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.title(f'{labels[condition]}: LC Signal by State (All Subjects)', 
                        fontsize=26)
                plt.xlim([-pre_cue_time, post_cue_time])
                #plt.legend(loc='upper right', fontsize=12)
                plt.tight_layout()
                
                # Save the figure
                save_figure(plt.gcf(), "all_subjects", "pooled", f"stay_switch_{condition}_by_state")
                plt.show()
        
        # Return analysis results
        result = {
            'subject_id': 'All',
            'specific_subjects': valid_subjects,
            'subject_condition_avgs': subject_condition_avgs,
            'time_axis': time_axis
        }
        
        if state:
            result['state_subject_avgs'] = state_subject_avgs
        
        return result
        
    else:
        # Single-subject analysis
        return analyze_stay_switch_photometry_single(subject_id, state, behavior_df, threshold)


def analyze_stay_switch_photometry_single(subject_id, state=False, behavior_df=None, threshold=0.8):
    """
    Analyze photometry signals for trials where the current choice is the same as (stay)
    or different from (switch) the previous trial's choice for a single subject.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    state : bool, optional (default=False)
        If True, further subdivides the data by behavioral state (stochastic/biased)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state when state=True
        
    Returns:
    --------
    dict: Analysis results including photometry signals by stay/switch and reward outcome
    """
    print(f"Analyzing stay/switch photometry for {subject_id}...")
    
    subject_path = os.path.join(base_dir, subject_id)
    
    # Find all session directories for this subject
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                      if os.path.isdir(os.path.join(subject_path, d)) and
                      os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                      d in matching_pennies_sessions])
    
    # Store data by condition
    trial_data = {
        'switch_rewarded': [],
        'switch_unrewarded': [],
        'stay_rewarded': [],
        'stay_unrewarded': []
    }
    
    # If state=True, also store data by state
    state_data = {}
    if state:
        state_data = {
            'stochastic': {
                'switch_rewarded': [],
                'switch_unrewarded': [],
                'stay_rewarded': [],
                'stay_unrewarded': []
            },
            'biased': {
                'switch_rewarded': [],
                'switch_unrewarded': [],
                'stay_rewarded': [],
                'stay_unrewarded': []
            }
        }
    
    time_axis = None
    total_sessions = 0
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Get photometry data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue
            
        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
            
        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue
            
        # Get session-specific data frame if state=True
        session_df = None
        if state:
            session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]
            if session_df.empty:
                print(f"No behavioral state data found for {subject_id}/{session_date}, skipping state analysis")
                # Continue with non-state analysis
            elif not all(col in session_df.columns for col in ['p_stochastic', 'p_leftbias', 'p_rightbias']):
                print(f"Missing state probability columns for {subject_id}/{session_date}, skipping state analysis")
                # Continue with non-state analysis
        
        total_sessions += 1
        
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        choices = np.array(behavior_data['choice'])
        rewards = np.array(behavior_data['reward'])
        
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        non_miss_choices = choices[non_miss_mask]
        non_miss_rewards = rewards[non_miss_mask]
        
        # Get mapping from filtered indices to original indices
        non_miss_indices = np.where(non_miss_mask)[0]
        filtered_to_orig = {i: non_miss_indices[i] for i in range(len(non_miss_indices))}
        
        # Skip if not enough trials after filtering
        if len(non_miss_choices) < 2:
            print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials")
            continue
            
        # Identify stay/switch trials
        # We start from the second trial, as the first has no previous choice
        for i in range(1, len(non_miss_choices)):
            prev_choice = non_miss_choices[i-1]
            curr_choice = non_miss_choices[i]
            curr_reward = non_miss_rewards[i]
            
            # Determine if choice stayed or switched
            is_stay = prev_choice == curr_choice
            
            # Get the original index for this trial 
            orig_idx = filtered_to_orig[i]
            
            # Skip if no photometry data available for this trial
            if orig_idx not in session_result["valid_trials"]:
                continue
                
            # Get photometry data for this trial
            valid_idx = np.where(np.array(session_result["valid_trials"]) == orig_idx)[0]
            if len(valid_idx) == 0:
                continue
                
            photo_data = session_result['epoched_data'][valid_idx[0]]
            
            # Determine trial category based on stay/switch and reward
            if is_stay:
                if curr_reward == 1:
                    category = 'stay_rewarded'
                else:
                    category = 'stay_unrewarded'
            else:
                if curr_reward == 1:
                    category = 'switch_rewarded'
                else:
                    category = 'switch_unrewarded'
            
            # Add to the appropriate category
            trial_data[category].append(photo_data)
            
            # If state=True and we have state data, also add to state-specific categories
            if state and session_df is not None and not session_df.empty and i < len(session_df):
                p_stochastic = session_df.iloc[i]['p_stochastic']
                p_leftbias = session_df.iloc[i]['p_leftbias']
                p_rightbias = session_df.iloc[i]['p_rightbias']
                
                # Determine state based on threshold
                trial_state = 'uncertain'
                if p_stochastic >= threshold:
                    trial_state = 'stochastic'
                elif p_leftbias >= threshold or p_rightbias >= threshold:
                    trial_state = 'biased'
                    
                # Add to state-specific category if not uncertain
                if trial_state != 'uncertain':
                    state_data[trial_state][category].append(photo_data)
    
    if total_sessions == 0:
        print(f"No valid sessions found for {subject_id}")
        return None
        
    # Convert lists to numpy arrays for each condition
    for category in trial_data:
        if trial_data[category]:
            trial_data[category] = np.array(trial_data[category])
            
    # Convert state-specific data to numpy arrays if state=True
    if state:
        for state_type in state_data:
            for category in state_data[state_type]:
                if state_data[state_type][category]:
                    state_data[state_type][category] = np.array(state_data[state_type][category])
    
    # Calculate averages and SEMs for each condition
    condition_data = {}
    for category, data in trial_data.items():
        if isinstance(data, np.ndarray) and len(data) > 0:
            condition_data[category] = {
                'data': data,
                'avg': np.mean(data, axis=0),
                'sem': calculate_sem(data, axis=0),
                'count': len(data)
            }
        else:
            condition_data[category] = {
                'data': np.array([]) if not isinstance(data, np.ndarray) else data,
                'avg': None,
                'sem': None,
                'count': 0 if not isinstance(data, np.ndarray) else len(data)
            }
    
    # Calculate averages and SEMs for state-specific data if state=True
    state_condition_data = {}
    if state:
        for state_type in state_data:
            state_condition_data[state_type] = {}
            for category, data in state_data[state_type].items():
                if isinstance(data, np.ndarray) and len(data) > 0:
                    state_condition_data[state_type][category] = {
                        'data': data,
                        'avg': np.mean(data, axis=0),
                        'sem': calculate_sem(data, axis=0),
                        'count': len(data)
                    }
                else:
                    state_condition_data[state_type][category] = {
                        'data': np.array([]) if not isinstance(data, np.ndarray) else data,
                        'avg': None,
                        'sem': None,
                        'count': 0 if not isinstance(data, np.ndarray) else len(data)
                    }
    
    # Create main plot showing all conditions
    plt.figure(figsize=(12, 7))
    
    # Define colors and labels
    colors = {
        'switch_rewarded': 'blue',
        'switch_unrewarded': 'lightblue',
        'stay_rewarded': 'red',
        'stay_unrewarded': 'lightsalmon'
    }
    
    labels = {
        'switch_rewarded': 'Switch - Rewarded',
        'switch_unrewarded': 'Switch - Unrewarded',
        'stay_rewarded': 'Stay - Rewarded',
        'stay_unrewarded': 'Stay - Unrewarded'
    }
    
    # Plot each condition
    for category, color in colors.items():
        if condition_data[category]['avg'] is not None:
            plt.fill_between(time_axis,
                           condition_data[category]['avg'] - condition_data[category]['sem'],
                           condition_data[category]['avg'] + condition_data[category]['sem'],
                           color=color, alpha=0.3)
            plt.plot(time_axis, condition_data[category]['avg'],
                   color=color, linewidth=2,
                   label=f"{labels[category]} (n={condition_data[category]['count']})")
    
    # Add vertical line at cue onset
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(f'LC Signal by Stay/Switch and Outcome: {subject_id} ({total_sessions} sessions)', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", "stay_switch_all_conditions")
    plt.show()
    
    # If state=True, create state-specific plots
    if state:
        # 1. Create one plot per state showing all 4 conditions
        for state_type in ['stochastic', 'biased']:
            # Skip if no data for this state
            if all(state_condition_data[state_type][cat]['count'] == 0 for cat in state_condition_data[state_type]):
                print(f"No data for {state_type} state, skipping plot")
                continue
                
            plt.figure(figsize=(12, 7))
            
            # Plot each condition for this state
            for category, color in colors.items():
                if state_condition_data[state_type][category]['avg'] is not None:
                    plt.fill_between(time_axis,
                                   state_condition_data[state_type][category]['avg'] - state_condition_data[state_type][category]['sem'],
                                   state_condition_data[state_type][category]['avg'] + state_condition_data[state_type][category]['sem'],
                                   color=color, alpha=0.3)
                    plt.plot(time_axis, state_condition_data[state_type][category]['avg'],
                           color=color, linewidth=2,
                           label=f"{labels[category]} (n={state_condition_data[state_type][category]['count']})")
            
            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)
            plt.title(f'{state_type.capitalize()} State: LC Signal by Stay/Switch and Outcome: {subject_id}', fontsize=20)
            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            
            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"stay_switch_all_conditions_{state_type}")
            plt.show()
        
        # 2. Create one plot per condition showing both states
        for category in ['switch_rewarded', 'switch_unrewarded', 'stay_rewarded', 'stay_unrewarded']:
            # Skip if no data for this category
            if not any(state_condition_data[s][category]['count'] > 0 for s in ['stochastic', 'biased']):
                print(f"No state-specific data for {category}, skipping plot")
                continue
                
            plt.figure(figsize=(12, 7))
            
            # Plot each state for this condition
            for state_type, color in [('stochastic', 'green'), ('biased', 'purple')]:
                if state_condition_data[state_type][category]['avg'] is not None:
                    plt.fill_between(time_axis,
                                   state_condition_data[state_type][category]['avg'] - state_condition_data[state_type][category]['sem'],
                                   state_condition_data[state_type][category]['avg'] + state_condition_data[state_type][category]['sem'],
                                   color=color, alpha=0.3)
                    plt.plot(time_axis, state_condition_data[state_type][category]['avg'],
                           color=color, linewidth=2,
                           label=f"{state_type.capitalize()} (n={state_condition_data[state_type][category]['count']})")
            
            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)
            plt.title(f'{labels[category]}: LC Signal by State: {subject_id}', fontsize=20)
            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            
            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"stay_switch_{category}_by_state")
            plt.show()
    
    # Return analysis results
    result = {
        'subject_id': subject_id,
        'time_axis': time_axis,
        'total_sessions': total_sessions,
        'condition_data': condition_data
    }
    
    if state:
        result['state_data'] = state_condition_data
    
    return result


def analyze_reward_rate_state_win_loss(subject_id="All", window_size=20, threshold=0.8, behavior_df=None, specific_subjects=None, split_biased=True):
    """
    Analyze photometry signals based on smoothed reward rate and behavioral state, comparing
    win vs. loss signals and previous outcome effects.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject or "All" for cross-subject analysis
    window_size : int, optional (default=20)
        Size of moving window for calculating reward rates
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
    split_biased : bool, optional (default=True)
        If True, split biased trials into high/low reward rate groups
        If False, treat all biased trials as a single group
        
    Returns:
    --------
    dict: Analysis results including win-loss differences and previous outcome effects
         by state and reward rate category
    """
    if subject_id == "All":
        # Cross-subject analysis
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            
        # Default list of subjects
        if specific_subjects is None:
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Store subject-level results
        subject_results = []
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"Processing subject {subj} for reward rate and state analysis...")
            
            try:
                # Process individual subject
                subj_result = analyze_reward_rate_state_win_loss_single(subj, window_size, threshold, behavior_df, split_biased)
                
                if subj_result:
                    subject_results.append(subj_result)
            except Exception as e:
                print(f"Error processing subject {subj}: {e}")
                traceback.print_exc()
        
        if not subject_results:
            print("No valid subjects found for cross-subject analysis")
            return None
        
        # Store time axis from first subject
        time_axis = None
        for result in subject_results:
            if 'time_axis' in result and result['time_axis'] is not None:
                time_axis = result['time_axis']
                break
        
        if time_axis is None:
            print("No time axis found in subject results")
            return None
        
        # Define groups based on split_biased parameter
        if split_biased:
            groups = ['stochastic', 'biased_low', 'biased_high']
        else:
            groups = ['stochastic', 'biased']
        
        # Collect win-loss differences and previous outcome effects for each subject by group
        cross_subject_data = {
            'win_loss_diff': {group: [] for group in groups},
            'prev_outcome_diff': {group: [] for group in groups}
        }
        
        for result in subject_results:
            for diff_type in ['win_loss_diff', 'prev_outcome_diff']:
                if diff_type in result:
                    for group in groups:
                        if group in result[diff_type] and result[diff_type][group] is not None:
                            cross_subject_data[diff_type][group].append(result[diff_type][group])
        
        # Calculate cross-subject averages and SEMs
        cross_subject_averages = {
            'win_loss_diff': {},
            'prev_outcome_diff': {}
        }
        
        for diff_type in ['win_loss_diff', 'prev_outcome_diff']:
            for group in groups:
                if len(cross_subject_data[diff_type][group]) > 0:
                    group_data = np.array(cross_subject_data[diff_type][group])
                    cross_subject_averages[diff_type][group] = {
                        'avg': np.mean(group_data, axis=0),
                        'sem': np.std(group_data, axis=0) / np.sqrt(len(group_data)),
                        'count': len(group_data)
                    }
                else:
                    cross_subject_averages[diff_type][group] = {
                        'avg': None,
                        'sem': None,
                        'count': 0
                    }
        
        # Create plots for cross-subject analysis
        
        # 1. Win-Loss difference plot
        plt.figure(figsize=(12, 7))
        
        # Define colors and labels based on split_biased parameter
        if split_biased:
            colors = {
                'stochastic': 'green',
                'biased_low': 'red',
                'biased_high': 'blue'
            }
            
            labels = {
                'stochastic': 'Stochastic',
                'biased_low': 'Biased (Low Reward Rate)',
                'biased_high': 'Biased (High Reward Rate)'
            }
        else:
            colors = {
                'stochastic': 'green',
                'biased': 'purple'
            }
            
            labels = {
                'stochastic': 'Stochastic',
                'biased': 'Biased'
            }
        
        for group, color in colors.items():
            if cross_subject_averages['win_loss_diff'][group]['avg'] is not None:
                group_avg = cross_subject_averages['win_loss_diff'][group]['avg']
                group_sem = cross_subject_averages['win_loss_diff'][group]['sem']
                group_count = cross_subject_averages['win_loss_diff'][group]['count']
                
                plt.fill_between(time_axis,
                               group_avg - group_sem,
                               group_avg + group_sem,
                               color=color, alpha=0.3)
                plt.plot(time_axis, group_avg,
                       color=color, linewidth=2,
                       label=f"{labels[group]} (n={group_count} subjects)")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('Win - Loss z-ΔF/F Difference', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        split_status = "Split Biased" if split_biased else "Combined Biased"
        plt.title(f'Win-Loss Signal Difference by State', fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])
        #plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Save the figure with appropriate suffix
        split_suffix = "split" if split_biased else "combined"
        save_figure(plt.gcf(), "all_subjects", "pooled", f"win_loss_diff_by_state_{split_suffix}_w{window_size}")
        
        plt.show()
        
        # 2. Previous Outcome difference plot
        plt.figure(figsize=(12, 7))
        
        for group, color in colors.items():
            if cross_subject_averages['prev_outcome_diff'][group]['avg'] is not None:
                group_avg = cross_subject_averages['prev_outcome_diff'][group]['avg']
                group_sem = cross_subject_averages['prev_outcome_diff'][group]['sem']
                group_count = cross_subject_averages['prev_outcome_diff'][group]['count']
                
                plt.fill_between(time_axis,
                               group_avg - group_sem,
                               group_avg + group_sem,
                               color=color, alpha=0.3)
                plt.plot(time_axis, group_avg,
                       color=color, linewidth=2,
                       label=f"{labels[group]} (n={group_count} subjects)")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=24)
        plt.ylabel('Prev Win - Prev Loss ΔF/F Difference', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f'Previous Outcome Effect by State', fontsize=26)
        plt.xlim([-pre_cue_time, post_cue_time])
        #plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), "all_subjects", "pooled", f"prev_outcome_diff_by_state_{split_suffix}_w{window_size}")
        
        plt.show()
        
        return {
            'subject_id': 'All',
            'specific_subjects': [result['subject_id'] for result in subject_results],
            'window_size': window_size,
            'threshold': threshold,
            'split_biased': split_biased,
            'time_axis': time_axis,
            'cross_subject_data': cross_subject_data,
            'cross_subject_averages': cross_subject_averages
        }
    
    else:
        # Single-subject analysis
        return analyze_reward_rate_state_win_loss_single(subject_id, window_size, threshold, behavior_df, split_biased)

def analyze_reward_rate_state_win_loss_single(subject_id, window_size=20, threshold=0.8, behavior_df=None, split_biased=True):
    """
    Analyze photometry signals based on smoothed reward rate and behavioral state for a single subject,
    comparing win vs. loss signals and previous outcome effects.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    window_size : int, optional (default=20)
        Size of moving window for calculating reward rates
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    split_biased : bool, optional (default=True)
        If True, split biased trials into high/low reward rate groups
        If False, treat all biased trials as a single group
        
    Returns:
    --------
    dict: Analysis results including win-loss differences and previous outcome effects
         by state and reward rate category
    """
    print(f"Analyzing reward rate and state win-loss effects for {subject_id}...")
    
    subject_path = os.path.join(base_dir, subject_id)
    
    # Find all session directories for this subject
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                      if os.path.isdir(os.path.join(subject_path, d)) and
                      os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                      d in matching_pennies_sessions])
    
    # Define which states to track based on split_biased parameter
    if split_biased:
        states = ['stochastic', 'biased_high', 'biased_low']
    else:
        states = ['stochastic', 'biased']
    
    # Store all trials by state and outcome category
    all_trials = {}
    for state in states:
        all_trials[state] = {
            'win': [],           # Current win trials
            'loss': [],          # Current loss trials
            'prev_win': [],      # Trials where previous trial was a win
            'prev_loss': []      # Trials where previous trial was a loss
        }
    
    time_axis = None
    total_sessions = 0
    
    # Store all biased trials' reward rates to calculate median for splitting
    if split_biased:
        all_biased_reward_rates = []
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]
        
        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue
        
        # Get behavioral data
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue
        
        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
        
        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue
        
        total_sessions += 1
        
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_rewards = rewards[non_miss_mask]
        
        # Skip if not enough trials after filtering
        if len(filtered_rewards) < window_size:
            print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials after filtering")
            continue
        
        # Get mapping from filtered indices to original indices
        non_miss_indices = np.where(non_miss_mask)[0]
        filtered_to_orig = {i: non_miss_indices[i] for i in range(len(non_miss_indices))}
        
        # Calculate overall session reward rate (for weighting early trials)
        overall_reward_rate = np.mean(filtered_rewards)
        
        # Calculate moving average reward rates for each trial
        reward_rates = []
        
        for i in range(len(filtered_rewards)):
            if i < window_size:
                # For early trials, use available data plus weighted overall rate
                rewards_so_far = filtered_rewards[:i + 1]
                available_weight = len(rewards_so_far) / window_size
                rate = (np.mean(rewards_so_far) * available_weight +
                        overall_reward_rate * (1 - available_weight))
            else:
                # For later trials, use full window
                window_rewards = filtered_rewards[i - window_size + 1:i + 1]
                rate = np.mean(window_rewards)
            
            reward_rates.append(rate)
        
        # If splitting biased trials, collect all biased trials' reward rates
        if split_biased:
            for i in range(len(session_df)):
                if i < len(reward_rates):
                    p_stochastic = session_df.iloc[i]['p_stochastic']
                    p_leftbias = session_df.iloc[i]['p_leftbias']
                    p_rightbias = session_df.iloc[i]['p_rightbias']
                    
                    # Only include clearly biased trials for median calculation
                    if p_leftbias >= threshold or p_rightbias >= threshold:
                        all_biased_reward_rates.append(reward_rates[i])
        
        # Process trials to categorize by state, reward rate, and outcome
        for i in range(len(filtered_rewards)):
            # Skip if this filtered index doesn't have state data
            if i >= len(session_df):
                continue
            
            # Get state for this trial
            p_stochastic = session_df.iloc[i]['p_stochastic']
            p_leftbias = session_df.iloc[i]['p_leftbias']
            p_rightbias = session_df.iloc[i]['p_rightbias']
            
            # Skip uncertain trials
            if p_stochastic < threshold and p_leftbias < threshold and p_rightbias < threshold:
                continue
            
            # Get original index for this trial
            orig_idx = filtered_to_orig[i]
            
            # Skip if no photometry data available for this trial
            if orig_idx not in session_result["valid_trials"]:
                continue
            
            # Get photometry data for this trial
            valid_idx = np.where(np.array(session_result["valid_trials"]) == orig_idx)[0]
            if len(valid_idx) == 0:
                continue
            
            photo_data = session_result['epoched_data'][valid_idx[0]]
            
            # Determine state for this trial
            if p_stochastic >= threshold:
                state = 'stochastic'
            elif split_biased:
                # Store temporarily with reward rate - will categorize after computing median
                if 'biased_temp' not in all_trials:
                    all_trials['biased_temp'] = []
                all_trials['biased_temp'].append({
                    'photo_data': photo_data,
                    'reward': filtered_rewards[i],
                    'prev_reward': filtered_rewards[i-1] if i > 0 else None,
                    'reward_rate': reward_rates[i]
                })
                continue  # Skip further processing for now
            else:
                state = 'biased'
            
            # Add trial to appropriate categories in all_trials
            # By current outcome
            if filtered_rewards[i] == 1:
                all_trials[state]['win'].append(photo_data)
            else:
                all_trials[state]['loss'].append(photo_data)
            
            # By previous outcome (skip first trial)
            if i > 0:
                if filtered_rewards[i-1] == 1:
                    all_trials[state]['prev_win'].append(photo_data)
                else:
                    all_trials[state]['prev_loss'].append(photo_data)
    
    if total_sessions == 0:
        print(f"No valid sessions found for {subject_id}")
        return None
    
    # If splitting biased trials, now categorize based on median reward rate
    if split_biased and 'biased_temp' in all_trials and all_biased_reward_rates:
        biased_median = np.median(all_biased_reward_rates)
        print(f"Median reward rate for biased trials: {biased_median:.4f}")
        
        # Process biased trials based on reward rate
        for trial in all_trials['biased_temp']:
            photo_data = trial['photo_data']
            reward = trial['reward']
            prev_reward = trial['prev_reward']
            reward_rate = trial['reward_rate']
            
            # Determine state based on reward rate
            state = 'biased_high' if reward_rate >= biased_median else 'biased_low'
            
            # Add to appropriate categories
            if reward == 1:
                all_trials[state]['win'].append(photo_data)
            else:
                all_trials[state]['loss'].append(photo_data)
            
            if prev_reward is not None:
                if prev_reward == 1:
                    all_trials[state]['prev_win'].append(photo_data)
                else:
                    all_trials[state]['prev_loss'].append(photo_data)
        
        # Remove temporary storage
        del all_trials['biased_temp']
    
    # Convert all trial data to numpy arrays
    for state in all_trials:
        for category in all_trials[state]:
            if all_trials[state][category]:
                all_trials[state][category] = np.array(all_trials[state][category])
    
    # Calculate averages for each state and condition
    state_averages = {}
    for state in all_trials:
        state_averages[state] = {}
        
        for category in all_trials[state]:
            if isinstance(all_trials[state][category], np.ndarray) and len(all_trials[state][category]) > 0:
                # Calculate average across all trials in this category
                state_averages[state][category] = {
                    'avg': np.mean(all_trials[state][category], axis=0),
                    'count': len(all_trials[state][category])
                }
            else:
                state_averages[state][category] = {
                    'avg': None,
                    'count': 0
                }
    
    # Calculate differences for win-loss and previous outcome
    win_loss_diff = {}
    prev_outcome_diff = {}
    
    for state in states:
        # Calculate win-loss difference
        if (state_averages[state]['win']['avg'] is not None and 
            state_averages[state]['loss']['avg'] is not None):
            win_loss_diff[state] = state_averages[state]['win']['avg'] - state_averages[state]['loss']['avg']
        else:
            win_loss_diff[state] = None
        
        # Calculate previous outcome effect
        if (state_averages[state]['prev_win']['avg'] is not None and 
            state_averages[state]['prev_loss']['avg'] is not None):
            prev_outcome_diff[state] = state_averages[state]['prev_win']['avg'] - state_averages[state]['prev_loss']['avg']
        else:
            prev_outcome_diff[state] = None
    
    # Print counts for each state and category
    print("\nTrial counts for each state and category:")
    for state in sorted(all_trials.keys()):
        print(f"\n{state.upper()} state:")
        for category in sorted(all_trials[state].keys()):
            count = state_averages[state][category]['count']
            print(f"  {category}: {count} trials")
    
    # Define colors for each state
    if split_biased:
        colors = {
            'stochastic': 'green',
            'biased_low': 'red',
            'biased_high': 'blue'
        }
        
        labels = {
            'stochastic': 'Stochastic',
            'biased_low': 'Biased (Low Reward Rate)',
            'biased_high': 'Biased (High Reward Rate)'
        }
    else:
        colors = {
            'stochastic': 'green',
            'biased': 'purple'
        }
        
        labels = {
            'stochastic': 'Stochastic',
            'biased': 'Biased'
        }
    
    # Create plots
    
    # 1. Win-Loss difference plot
    plt.figure(figsize=(12, 7))
    
    for state, color in colors.items():
        if win_loss_diff[state] is not None:
            plt.plot(time_axis, win_loss_diff[state], color=color, linewidth=2,
                   label=f"{labels[state]} (win n={state_averages[state]['win']['count']}, loss n={state_averages[state]['loss']['count']})")
    
    # Add vertical line at cue onset
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Win - Loss ΔF/F Difference', fontsize=16)
    split_status = "Split Biased" if split_biased else "Combined Biased"
    plt.title(f'Win-Loss Signal Difference by State ({split_status}): {subject_id}', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    
    # Save the figure with appropriate suffix
    split_suffix = "split" if split_biased else "combined"
    save_figure(plt.gcf(), subject_id, "pooled", f"win_loss_diff_by_state_{split_suffix}_w{window_size}")
    
    plt.show()
    
    # 2. Previous Outcome difference plot
    plt.figure(figsize=(12, 7))
    
    for state, color in colors.items():
        if prev_outcome_diff[state] is not None:
            plt.plot(time_axis, prev_outcome_diff[state], color=color, linewidth=2,
                   label=f"{labels[state]} (prev win n={state_averages[state]['prev_win']['count']}, prev loss n={state_averages[state]['prev_loss']['count']})")
    
    # Add vertical line at cue onset
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Prev Win - Prev Loss ΔF/F Difference', fontsize=16)
    plt.title(f'Previous Outcome Effect by State ({split_status}): {subject_id}', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", f"prev_outcome_diff_by_state_{split_suffix}_w{window_size}")
    
    plt.show()
    
    return {
        'subject_id': subject_id,
        'window_size': window_size,
        'threshold': threshold,
        'split_biased': split_biased,
        'time_axis': time_axis,
        'total_sessions': total_sessions,
        'state_averages': state_averages,
        'win_loss_diff': win_loss_diff,
        'prev_outcome_diff': prev_outcome_diff
    }

def analyze_session_win_loss_difference_gap(subject_id, session_date=None, comp_conf=False, behavior_df=None, sem=True, state_sort=False):
    """
    Analyze win-loss difference across photometry sessions for a subject

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    comp_conf : bool, optional
        If True, sorts sessions by computer confidence rather than chronologically
    behavior_df : pandas.DataFrame, optional
        Pre-loaded dataframe to use for confidence calculations
    sem : bool, optional
        If True, display shaded standard error of the mean regions
    state_sort : bool, optional
        If True (and comp_conf is False), sort sessions by percentage of stochastic trials

    Returns:
    --------
    dict: Dictionary of session win-loss difference analyses
    """
    if session_date is None:
        if behavior_df is not None:
            # Get sessions from the provided dataframe
            sessions = sorted(behavior_df[behavior_df['subjid'] == subject_id]['date'].unique())
        else:
            # Get sessions from the filesystem
            subject_path = os.path.join(base_dir, subject_id)
            sessions = sorted([d for d in os.listdir(subject_path)
                               if os.path.isdir(os.path.join(subject_path, d)) and
                               os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]

    # Load dataframe if needed for confidence calculations and not provided
    if (comp_conf or state_sort) and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            print(f"Loaded parquet data for confidence/state calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
            state_sort = False
    else:
        df = behavior_df  

    # Store results for each session
    session_differences = {}
    session_confidences = {}
    session_stochastic_pcts = {}

    # Define threshold for stochastic state
    stochastic_threshold = 0.8

    # Process each session
    for idx, session_date in enumerate(sessions):
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df, z_score=False)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"]) 
                                  if idx in session_result["non_m_trials"]])

        # Get reward outcomes and photometry data
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        rewarded_trials = session_plots[reward_outcomes == 1]
        unrewarded_trials = session_plots[reward_outcomes == 0]

        # Compute average rewarded and unrewarded signals with SEM
        rewarded_avg = np.mean(rewarded_trials, axis=0)
        unrewarded_avg = np.mean(unrewarded_trials, axis=0)
        rewarded_sem = calculate_sem(rewarded_trials, axis=0)
        unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)

        # Compute win-loss difference
        rewarded_avg = rewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        unrewarded_avg = unrewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        win_loss_diff = rewarded_avg - unrewarded_avg
        win_loss_sem = np.sqrt(rewarded_sem**2 + unrewarded_sem**2)

        # Store the difference data
        session_differences[session_date] = {
            'diff': win_loss_diff,
            'sem': win_loss_sem,
            'time_axis': session_result['time_axis']
        }

        # Calculate stochastic state percentage for this session
        try:
            # Get data for this session
            session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
            
            if session_df.empty:
                print(f"No behavioral data found for {subject_id}/{session_date}")
                continue
            
            # Calculate stochastic state percentage
            if 'p_stochastic' in session_df.columns:
                total_trials = len(session_df)
                stochastic_trials = sum(session_df['p_stochastic'] >= stochastic_threshold)
                stochastic_pct = (stochastic_trials / total_trials) * 100
                
                session_stochastic_pcts[session_date] = stochastic_pct
                print(f"Session {session_date}: {stochastic_pct:.1f}% stochastic trials ({stochastic_trials}/{total_trials})")
            else:
                print(f"No p_stochastic column found in data for {subject_id}/{session_date}")
        except Exception as e:
            print(f"Error calculating stochastic percentage for {subject_id}/{session_date}: {e}")

        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Extract p-values and calculate confidence
                if 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) == 0:
                        print(f"No valid p-values for {subject_id}/{session_date}")
                        continue
                    
                    # Cap very small p-values at 10^-12 to avoid infinite confidence
                    min_p_value = 1e-12
                    p_values = np.maximum(p_values, min_p_value)
                    
                    # Calculate confidence as -log10(p_value)
                    confidence = -np.log10(p_values)
                    
                    # Calculate average confidence for the session
                    avg_confidence = np.mean(confidence)
                    session_confidences[session_date] = avg_confidence
                    print(f"Session {session_date} average confidence: {avg_confidence:.4f}")
                else:
                    print(f"No min_pvalue column found in data for {subject_id}/{session_date}")
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    # Check if we have any valid sessions
    if not session_differences:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Determine sort mode and filter sessions accordingly
    if comp_conf and session_confidences:
        # Sort by computer confidence
        # Filter out sessions with missing confidence values
        valid_sessions = [s for s in session_confidences.keys() if s in session_differences]
        
        if not valid_sessions:
            print("No sessions with valid confidence values found")
            sorted_sessions = sorted(session_differences.keys())
        else:
            # Sort by confidence (highest first)
            sorted_sessions = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_sessions):
                stoch_info = f", {session_stochastic_pcts.get(sess, 0):.1f}% stochastic" if sess in session_stochastic_pcts else ""
                print(f"{i+1}. Session {sess}: conf={session_confidences[sess]:.4f}{stoch_info}")
        
        sort_method = "by computer confidence"
    elif state_sort and session_stochastic_pcts:
        # Sort by stochastic percentage
        # Filter out sessions with missing stochastic percentage values
        valid_sessions = [s for s in session_stochastic_pcts.keys() if s in session_differences]
        
        if not valid_sessions:
            print("No sessions with valid stochastic percentages found")
            sorted_sessions = sorted(session_differences.keys())
        else:
            # Sort by stochastic percentage (lowest first)
            sorted_sessions = sorted(valid_sessions, key=lambda s: session_stochastic_pcts.get(s, 0))
            
            # Print the stochastic percentage ranking
            print("\nSessions ranked by stochastic state percentage (lowest to highest):")
            for i, sess in enumerate(sorted_sessions):
                print(f"{i+1}. Session {sess}: {session_stochastic_pcts[sess]:.1f}% stochastic")
        
        sort_method = "by stochastic percentage"
    else:
        # Use chronological sorting (default)
        sorted_sessions = sorted(session_differences.keys())
        sort_method = "chronologically"

    # Create colors for plotting
    num_sessions = len(sorted_sessions)
    
    # Set colormap based on sort method
    if subject_id == "JOA-M-0020":
        # Special coloring for JOA-M-0020: Yellow -> Green -> Blue gradient (viridis-like)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sessions))
    elif state_sort:
        # For stochastic percentage sorting: Blue (low %) to Yellow (high %)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sessions))
    else:
        # Regular Blues colormap for other subjects and sort methods
        if comp_conf and session_confidences:
            # When sorting by confidence, use reversed color gradient (highest confidence = lightest blue)
            colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))
        else:
            # Default: earliest session = lightest blue
            colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

    # Create plot
    plt.figure(figsize=(12, 7))
    chronological_sessions = sorted(session_differences.keys())

    # Create mapping from session date to chronological position (1-indexed)
    chronological_order = {sess_date: idx+1 for idx, sess_date in enumerate(chronological_sessions)}

    # Plot each session's win-loss difference
    for idx, sess_date in enumerate(sorted_sessions):
        time_axis = session_differences[sess_date]['time_axis']
        win_loss_diff = session_differences[sess_date]['diff']
        win_loss_sem = session_differences[sess_date]['sem']
        
        # Create label with relevant info (confidence or stochastic percentage)
        label_parts = [f'Session {chronological_order[sess_date]}']
        
        if comp_conf and sess_date in session_confidences:
            label_parts.append(f'conf: {session_confidences[sess_date]:.2f}')
            
        if sess_date in session_stochastic_pcts:
            label_parts.append(f'{session_stochastic_pcts[sess_date]:.1f}% stoch')
            
        label = f'{", ".join(label_parts)}'
        
        # Plot with shaded error region
        if sem:
            plt.fill_between(time_axis,
                            win_loss_diff - win_loss_sem,
                            win_loss_diff + win_loss_sem,
                            color=colors[idx], alpha=0.2)
        plt.plot(time_axis, win_loss_diff,
                color=colors[idx],
                label=label, linewidth=2)

    # Add reference lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s) after first lick', fontsize=24)
    plt.ylabel('Rewarded - Unrewarded ΔF/F', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    plt.title(f'Win-Loss Difference: {subject_id} (sorted {sort_method})', fontsize=26)
    
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the figure
    if comp_conf:
        sort_suffix = "by_comp_conf"
    elif state_sort:
        sort_suffix = "by_stochastic_pct"
    else:
        sort_suffix = "chronological"
        
    sem_suffix = "_sem" if sem else ""
    save_figure(plt.gcf(), subject_id, "win_loss_diff", f"win_loss_difference_{sort_suffix}{sem_suffix}")

    plt.show()

    # Return results
    return {
        'session_differences': session_differences,
        'session_confidences': session_confidences if comp_conf else None,
        'session_stochastic_pcts': session_stochastic_pcts,
        'sorted_sessions': sorted_sessions
    }

def analyze_confidence_stochastic_relationship(subject_id, session_date=None, behavior_df=None, window_size=20):
    """
    Analyze the relationship between computer confidence and percentage of stochastic trials,
    either across multiple sessions or within a single session.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze in detail. If None, analyze all sessions.
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
    window_size : int, optional (default=20)
        Size of moving window for calculating rolling averages within a session
        
    Returns:
    --------
    dict: Analysis results including confidence and stochastic percentages
    """
    print(f"Analyzing confidence and stochastic relationship for {subject_id}...")
    
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            behavior_df['date'] = behavior_df['date'].astype(str)
            print(f"Loaded behavior data from parquet file")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None
    
    # Filter data for this subject
    subject_data = behavior_df[behavior_df['subjid'] == subject_id]
    
    # Check if we have required columns
    required_columns = ['date', 'p_stochastic', 'min_pvalue']
    if not all(col in subject_data.columns for col in required_columns):
        print(f"Missing required columns in behavior data: need {required_columns}")
        return None
    
    # Define threshold for stochastic state
    stochastic_threshold = 0.8
    
    if session_date is None:
        # Cross-session analysis
        print("\nAnalyzing across multiple sessions...")
        
        # Get all sessions for this subject
        sessions = sorted(subject_data['date'].unique())
        
        # Store results for each session
        session_results = []
        
        # Process each session
        for idx, sess_date in enumerate(sessions):
            sess_data = subject_data[subject_data['date'] == sess_date]
            
            if len(sess_data) < 50:  # Skip sessions with too few trials
                print(f"Skipping session {sess_date}, too few trials: {len(sess_data)}")
                continue
            
            # Calculate stochastic percentage
            stochastic_trials = sum(sess_data['p_stochastic'] >= stochastic_threshold)
            total_trials = len(sess_data)
            stochastic_pct = (stochastic_trials / total_trials) * 100
            
            # Calculate computer confidence
            p_values = sess_data['min_pvalue'].values
            p_values = p_values[~np.isnan(p_values)]  # Remove NaN values
            
            if len(p_values) == 0:
                print(f"No valid p-values for {sess_date}, skipping")
                continue
            
            # Cap very small p-values to avoid infinite confidence
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)
            
            # Calculate confidence as -log10(p_value)
            confidence = -np.log10(p_values)
            avg_confidence = np.mean(confidence)
            
            # Store session results
            session_results.append({
                'session_date': sess_date,
                'session_idx': idx + 1,  # 1-indexed session number
                'stochastic_pct': stochastic_pct,
                'confidence': avg_confidence,
                'total_trials': total_trials
            })
            
            print(f"Session {idx+1} ({sess_date}): {stochastic_pct:.1f}% stochastic, confidence={avg_confidence:.4f}")
        
        if not session_results:
            print("No valid sessions found")
            return None
        
        # Create normalized versions for plotting
        stochastic_values = [res['stochastic_pct'] for res in session_results]
        confidence_values = [res['confidence'] for res in session_results]
        
        stochastic_min = min(stochastic_values)
        stochastic_max = max(stochastic_values)
        confidence_min = min(confidence_values)
        confidence_max = max(confidence_values)
        
        # Normalize to 0-1 range
        normalized_stochastic = [(x - stochastic_min) / (stochastic_max - stochastic_min) 
                                if stochastic_max > stochastic_min else 0.5 for x in stochastic_values]
        
        normalized_confidence = [(x - confidence_min) / (confidence_max - confidence_min) 
                               if confidence_max > confidence_min else 0.5 for x in confidence_values]
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # X-axis values: session numbers
        x_values = [res['session_idx'] for res in session_results]
        
        # Plot normalized values
        plt.plot(x_values, normalized_stochastic, 'go-', linewidth=2, markersize=8, 
                label=f'Stochastic % (range: {stochastic_min:.1f}% - {stochastic_max:.1f}%)')
        
        plt.plot(x_values, normalized_confidence, 'bo-', linewidth=2, markersize=8,
                label=f'Confidence (range: {confidence_min:.2f} - {confidence_max:.2f})')
        
        # Add reference line at 0.5
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Create second y-axis for original values
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot original values on second y-axis
        ax2.plot(x_values, stochastic_values, 'go-', alpha=0)  # Invisible, just to set scale
        ax2.plot(x_values, confidence_values, 'bo-', alpha=0)  # Invisible, just to set scale
        
        # Configure primary y-axis (normalized)
        ax1.set_ylabel('Normalized Value', fontsize=16)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Configure secondary y-axes
        ax2.set_ylabel('Original Values', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # X-axis formatting
        plt.xlabel('Session Number', fontsize=16)
        plt.xticks(x_values)
        
        plt.title(f'Stochastic Percentage and Computer Confidence by Session: {subject_id}', fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), subject_id, "analysis", "session_stochastic_confidence")
        
        plt.show()
        
        # Calculate correlation
        correlation = np.corrcoef(stochastic_values, confidence_values)[0, 1]
        print(f"\nCorrelation between stochastic percentage and confidence: {correlation:.4f}")
        
        return {
            'subject_id': subject_id,
            'session_results': session_results,
            'correlation': correlation
        }
    
    else:
        # Single session analysis
        print(f"\nAnalyzing single session: {session_date}")
        
        # Get data for the specific session
        sess_data = subject_data[subject_data['date'] == session_date]
        
        if len(sess_data) < 50:
            print(f"Not enough trials in session {session_date}: {len(sess_data)}")
            return None
            
        # Sort by trial index to ensure correct order
        if 'trial_idx' in sess_data.columns:
            sess_data = sess_data.sort_values('trial_idx')
        
        # Calculate across-session averages to use for early trials
        overall_stochastic_pct = (subject_data['p_stochastic'] >= stochastic_threshold).mean() * 100
        
        # For confidence, first handle NaNs and cap minimum p-values
        p_values = subject_data['min_pvalue'].values
        p_values = p_values[~np.isnan(p_values)]
        min_p_value = 1e-12
        p_values = np.maximum(p_values, min_p_value)
        overall_confidence = np.mean(-np.log10(p_values))
        
        # Extract data from session
        p_stochastic = sess_data['p_stochastic'].values
        p_values = sess_data['min_pvalue'].values
        
        # Prepare arrays to store moving averages
        stochastic_moving_avg = []
        confidence_moving_avg = []
        
        # Calculate moving averages for each trial
        for i in range(len(sess_data)):
            # For stochastic percentage
            if i < window_size:
                # For early trials, weight in the overall average
                stoch_so_far = (p_stochastic[:i+1] >= stochastic_threshold).mean() * 100
                available_weight = (i+1) / window_size
                stoch_avg = stoch_so_far * available_weight + overall_stochastic_pct * (1 - available_weight)
            else:
                # For later trials, use full window
                stoch_window = p_stochastic[i-window_size+1:i+1]
                stoch_avg = (stoch_window >= stochastic_threshold).mean() * 100
            
            stochastic_moving_avg.append(stoch_avg)
            
            # For computer confidence
            if i < window_size:
                # Extract p-values for trials so far
                p_window = p_values[:i+1]
                # Remove NaN values
                p_window = p_window[~np.isnan(p_window)]
                
                if len(p_window) > 0:
                    # Cap minimum p-values
                    p_window = np.maximum(p_window, min_p_value)
                    # Calculate confidence
                    conf_so_far = np.mean(-np.log10(p_window))
                    available_weight = (i+1) / window_size
                    conf_avg = conf_so_far * available_weight + overall_confidence * (1 - available_weight)
                else:
                    conf_avg = overall_confidence
            else:
                # Extract p-values for this window
                p_window = p_values[i-window_size+1:i+1]
                # Remove NaN values
                p_window = p_window[~np.isnan(p_window)]
                
                if len(p_window) > 0:
                    # Cap minimum p-values
                    p_window = np.maximum(p_window, min_p_value)
                    # Calculate confidence
                    conf_avg = np.mean(-np.log10(p_window))
                else:
                    conf_avg = None
            
            confidence_moving_avg.append(conf_avg)
        
        # Convert to numpy arrays for easier manipulation
        stochastic_moving_avg = np.array(stochastic_moving_avg)
        confidence_moving_avg = np.array(confidence_moving_avg, dtype=float)
        
        # Filter out any None values in confidence
        valid_indices = ~np.isnan(confidence_moving_avg)
        
        if sum(valid_indices) < 50:
            print(f"Not enough valid confidence data points in session: {sum(valid_indices)}")
            return None
        
        # Create normalized versions for plotting
        stochastic_min = np.min(stochastic_moving_avg)
        stochastic_max = np.max(stochastic_moving_avg)
        confidence_min = np.min(confidence_moving_avg[valid_indices])
        confidence_max = np.max(confidence_moving_avg[valid_indices])
        
        # Normalize to 0-1 range
        normalized_stochastic = (stochastic_moving_avg - stochastic_min) / (stochastic_max - stochastic_min)
        
        normalized_confidence = np.full_like(confidence_moving_avg, np.nan)
        normalized_confidence[valid_indices] = (confidence_moving_avg[valid_indices] - confidence_min) / (confidence_max - confidence_min)
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # X-axis values: trial indices
        x_values = np.arange(1, len(sess_data) + 1)
        
        # Plot normalized values
        plt.plot(x_values, normalized_stochastic, 'g-', linewidth=2,
                label=f'Stochastic % (range: {stochastic_min:.1f}% - {stochastic_max:.1f}%)')
        
        plt.plot(x_values, normalized_confidence, 'b-', linewidth=2,
                label=f'Confidence (range: {confidence_min:.2f} - {confidence_max:.2f})')
        
        # Add reference line at 0.5
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Create second y-axis for original values
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot original values on second y-axis
        ax2.plot(x_values, stochastic_moving_avg, 'g-', alpha=0)  # Invisible, just to set scale
        ax2.plot(x_values[valid_indices], confidence_moving_avg[valid_indices], 'b-', alpha=0)  # Invisible, just to set scale
        
        # Configure primary y-axis (normalized)
        ax1.set_ylabel('Normalized Value', fontsize=16)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelcolor='black')
        
        
        # X-axis formatting
        plt.xlabel('Trial Number', fontsize=16)
        
        plt.title(f'Moving Average ({window_size} trials) of\nStochastic Percentage and Computer Confidence: {subject_id}, {session_date}', fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), subject_id, session_date, "trial_stochastic_confidence")
        
        plt.show()
        
        # Calculate correlation
        valid_stoch = stochastic_moving_avg[valid_indices]
        valid_conf = confidence_moving_avg[valid_indices]
        correlation = np.corrcoef(valid_stoch, valid_conf)[0, 1]
        print(f"\nCorrelation between stochastic percentage and confidence: {correlation:.4f}")
        
        return {
            'subject_id': subject_id,
            'session_date': session_date,
            'stochastic_moving_avg': stochastic_moving_avg,
            'confidence_moving_avg': confidence_moving_avg,
            'correlation': correlation,
            'window_size': window_size
        }


def analyze_state_period_photometry(subject_id="All", threshold=0.8, behavior_df=None, specific_subjects=None):
    """
    Analyze photometry signals during different periods of behavioral states (early, middle, late)
    and during state transitions (early transition, late transition).
    
    States are separated into stochastic, leftbiased, and rightbiased. The function identifies
    state transitions between any two different states, and analyzes both within-state periods
    and transition periods.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject or "All" for cross-subject analysis
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    specific_subjects : list, optional
        List of subject IDs to include if subject_id="All"
    
    Returns:
    --------
    dict: Analysis results including photometry signals by state period
    """
    if subject_id == "All":
        # Cross-subject analysis
        if behavior_df is None:
            print("Loading behavior data for all subjects...")
            behavior_df = load_filtered_behavior_data("MatchingPennies")
        
        # Default list of subjects
        if specific_subjects is None:
            specific_subjects = ["JOA-M-0022", "JOA-M-0023", "JOA-M-0024", "JOA-M-0025", "JOA-M-0026"]
            print(f"Using default subject list: {specific_subjects}")
        
        # Container for subject-level results
        subject_results = []
        
        # Process each subject individually
        for subj in specific_subjects:
            print(f"\nProcessing subject {subj} for state period analysis...")
            try:
                # Process individual subject
                subj_result = analyze_state_period_photometry_single(subj, threshold, behavior_df)
                if subj_result:
                    subject_results.append(subj_result)
            except Exception as e:
                print(f"Error processing subject {subj}: {e}")
                traceback.print_exc()
        
        if not subject_results:
            print("No valid subjects found for cross-subject analysis")
            return None
        
        # Define state types
        state_types = ['stochastic', 'leftbiased', 'rightbiased']
        
        # Define the period categories
        within_periods = []
        for state in state_types:
            within_periods.extend([f"{state}_early", f"{state}_middle", f"{state}_late"])
        
        # Define transition periods
        transition_periods = []
        for target_state in state_types:
            for source_state in state_types:
                if source_state != target_state:
                    transition_periods.extend([
                        f"to_{target_state}_from_{source_state}_early",
                        f"to_{target_state}_from_{source_state}_late"
                    ])
        
        # Combined period list
        periods = within_periods + transition_periods
        
        # Store time axis from first subject
        time_axis = None
        for result in subject_results:
            if 'time_axis' in result and result['time_axis'] is not None:
                time_axis = result['time_axis']
                break
        
        if time_axis is None:
            print("No time axis found in subject results")
            return None
        
        # Collect data for each period across subjects
        cross_subject_data = {period: [] for period in periods}
        
        for result in subject_results:
            for period in periods:
                if period in result['period_data'] and result['period_data'][period]['avg'] is not None:
                    # Add this subject's average for this period
                    cross_subject_data[period].append(result['period_data'][period]['avg'])
        
        # Calculate cross-subject averages and SEMs
        cross_subject_averages = {}
        for period in periods:
            if len(cross_subject_data[period]) > 0:
                period_data = np.array(cross_subject_data[period])
                cross_subject_averages[period] = {
                    'avg': np.mean(period_data, axis=0),
                    'sem': np.std(period_data, axis=0) / np.sqrt(len(period_data)),
                    'count': len(period_data)
                }
            else:
                cross_subject_averages[period] = {
                    'avg': None,
                    'sem': None,
                    'count': 0
                }
        
        # Create plots for cross-subject analysis
        
        # 1. Plot all transition periods together
        plt.figure(figsize=(14, 8))
        
        # Define colors for transition periods
        transition_colors = {
            'stochastic': 'green',
            'leftbiased': 'blue',
            'rightbiased': 'red'
        }
        
        # Plot all transitions grouped by target state, separating early and late
        for target_state in state_types:
            for transition_phase in ['early', 'late']:
                # Combine data from transitions to this target state from all source states
                target_periods = [p for p in transition_periods if f"to_{target_state}" in p and p.endswith(f"_{transition_phase}")]
                
                # Skip if no data
                if not any(cross_subject_averages[p]['avg'] is not None for p in target_periods):
                    continue
                
                # Combine averages across all source states
                valid_avgs = [cross_subject_averages[p]['avg'] for p in target_periods 
                             if cross_subject_averages[p]['avg'] is not None]
                
                if not valid_avgs:
                    continue
                    
                combined_avg = np.mean(valid_avgs, axis=0)
                
                # Calculate combined SEM
                valid_sems = [cross_subject_averages[p]['sem'] for p in target_periods 
                             if cross_subject_averages[p]['sem'] is not None]
                combined_sem = np.sqrt(np.mean([sem**2 for sem in valid_sems], axis=0))
                
                # Count total subjects
                total_count = sum(cross_subject_averages[p]['count'] for p in target_periods)
                
                # Line style based on transition phase
                linestyle = '-' if transition_phase == 'early' else '--'
                
                # Plot combined transition data
                plt.fill_between(time_axis,
                              combined_avg - combined_sem,
                              combined_avg + combined_sem,
                              color=transition_colors[target_state], alpha=0.2)
                
                plt.plot(time_axis, combined_avg,
                       color=transition_colors[target_state], linestyle=linestyle, linewidth=2,
                       label=f"To {target_state.capitalize()} ({transition_phase}, n={total_count})")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=16)
        plt.ylabel('z-ΔF/F', fontsize=16)
        plt.title('Transition Period Photometry Signals (All Subjects)', fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        save_figure(plt.gcf(), "all_subjects", "pooled", "all_transitions_photometry")
        plt.show()
        
        # 2. Create individual plots for each target state's transitions
        for target_state in state_types:
            plt.figure(figsize=(12, 7))
            
            # Plot transitions to this target state from each source state
            for source_state in [s for s in state_types if s != target_state]:
                for phase in ['early', 'late']:
                    period = f"to_{target_state}_from_{source_state}_{phase}"
                    
                    if cross_subject_averages[period]['avg'] is not None:
                        avg = cross_subject_averages[period]['avg']
                        sem = cross_subject_averages[period]['sem']
                        count = cross_subject_averages[period]['count']
                        
                        # Determine color based on source state
                        color = transition_colors[source_state]
                        
                        # Line style based on transition phase
                        linestyle = '-' if phase == 'early' else '--'
                        
                        plt.fill_between(time_axis,
                                      avg - sem,
                                      avg + sem,
                                      color=color, alpha=0.2)
                        
                        plt.plot(time_axis, avg,
                               color=color, linestyle=linestyle, linewidth=2,
                               label=f"From {source_state.capitalize()} ({phase}, n={count})")
            
            # Add vertical line at cue onset
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            plt.xlabel('Time (s) after first lick', fontsize=16)
            plt.ylabel('z-ΔF/F', fontsize=16)
            plt.title(f'Transitions to {target_state.capitalize()} State (All Subjects)', fontsize=20)
            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            
            save_figure(plt.gcf(), "all_subjects", "pooled", f"transitions_to_{target_state}")
            plt.show()
        
        # 3. Plot within-state periods for each state
        for state in state_types:
            plt.figure(figsize=(12, 7))
            
            # Define colors for within-state periods
            within_colors = {
                'early': 'lightblue',
                'middle': 'blue',
                'late': 'darkblue'
            }
            
            # Plot each period within this state
            for period_type in ['early', 'middle', 'late']:
                period = f"{state}_{period_type}"
                
                if cross_subject_averages[period]['avg'] is not None:
                    avg = cross_subject_averages[period]['avg']
                    sem = cross_subject_averages[period]['sem']
                    count = cross_subject_averages[period]['count']
                    
                    plt.fill_between(time_axis,
                                  avg - sem,
                                  avg + sem,
                                  color=within_colors[period_type], alpha=0.3)
                    
                    plt.plot(time_axis, avg,
                           color=within_colors[period_type], linewidth=2,
                           label=f"{period_type.capitalize()} (n={count})")
            
            # Add vertical line at cue onset
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            plt.xlabel('Time (s) after first lick', fontsize=16)
            plt.ylabel('z-ΔF/F', fontsize=16)
            plt.title(f'{state.capitalize()} State Periods (All Subjects)', fontsize=20)
            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=12)
            plt.tight_layout()
            
            save_figure(plt.gcf(), "all_subjects", "pooled", f"{state}_periods_photometry")
            plt.show()
        
        # 4. Combined plot comparing all states (middle period only for clarity)
        plt.figure(figsize=(12, 7))
        
        # Define colors for each state
        state_colors = {
            'stochastic': 'green',
            'leftbiased': 'blue',
            'rightbiased': 'red'
        }
        
        # Plot middle period for each state
        for state in state_types:
            period = f"{state}_middle"
            
            if cross_subject_averages[period]['avg'] is not None:
                avg = cross_subject_averages[period]['avg']
                sem = cross_subject_averages[period]['sem']
                count = cross_subject_averages[period]['count']
                
                plt.fill_between(time_axis,
                              avg - sem,
                              avg + sem,
                              color=state_colors[state], alpha=0.3)
                
                plt.plot(time_axis, avg,
                       color=state_colors[state], linewidth=2,
                       label=f"{state.capitalize()} (n={count})")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=16)
        plt.ylabel('z-ΔF/F', fontsize=16)
        plt.title('Comparison Across States (Middle Period, All Subjects)', fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        save_figure(plt.gcf(), "all_subjects", "pooled", "all_states_middle_comparison")
        plt.show()
        
        return {
            'subject_id': 'All',
            'specific_subjects': [result['subject_id'] for result in subject_results],
            'threshold': threshold,
            'time_axis': time_axis,
            'cross_subject_data': cross_subject_data,
            'cross_subject_averages': cross_subject_averages
        }
    
    else:
        # Single-subject analysis
        return analyze_state_period_photometry_single(subject_id, threshold, behavior_df)


def analyze_state_period_photometry_single(subject_id, threshold=0.8, behavior_df=None):
    """
    Analyze photometry signals for a single subject during different periods of 
    behavioral states (early, middle, late) and during state transitions (early transition, late transition).
    
    States are separated into stochastic, leftbiased, and rightbiased. The function identifies
    state transitions between any two different states, and analyzes both within-state periods
    and transition periods.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    threshold : float, optional (default=0.8)
        Probability threshold for assigning trials to a state
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from file
    
    Returns:
    --------
    dict: Analysis results including photometry signals by state period
    """
    print(f"Analyzing state periods for {subject_id}...")
    
    subject_path = os.path.join(base_dir, subject_id)
    # Load the parquet file to get state probability data
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            subject_data = df[(df['subjid'] == subject_id) & (df['protocol'].str.contains('MatchingPennies', na=False))]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} from parquet file")
    except Exception as e:
        print(f"Warning: Could not load session info: {e}")

    # Sort sessions chronologically, filtering to only include MatchingPennies sessions
    sessions = sorted([d for d in os.listdir(subject_path)
                      if os.path.isdir(os.path.join(subject_path, d)) and
                      os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                      d in matching_pennies_sessions])
    
    # Define the states
    state_types = ['stochastic', 'leftbiased', 'rightbiased']
    
    # Initialize empty lists to store photometry data for within-state periods
    within_periods = {}
    for state in state_types:
        within_periods[f"{state}_early"] = []
        within_periods[f"{state}_middle"] = []
        within_periods[f"{state}_late"] = []
    
    # Initialize empty lists for transition periods
    transition_periods = {}
    for target_state in state_types:
        for source_state in state_types:
            if source_state != target_state:
                transition_periods[f"to_{target_state}_from_{source_state}_early"] = []
                transition_periods[f"to_{target_state}_from_{source_state}_late"] = []
    
    # Combine all period types
    period_trials = {**within_periods, **transition_periods}
    
    # Store photometry time axis
    time_axis = None
    session_count = 0
    state_run_counts = {state: 0 for state in state_types}
    state_trial_counts = {state: 0 for state in state_types}
    transition_counts = {f"{source}_to_{target}": 0 
                         for source in state_types for target in state_types if source != target}
    
    # Parameters for state runs
    min_state_duration = 15  # Minimum trials for a state to be considered a valid state period
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Get session data from parquet file
        session_df = behavior_df[(behavior_df['subjid'] == subject_id) & (behavior_df['date'] == session_date)]
        
        if session_df.empty:
            print(f"No behavioral data found for {subject_id}/{session_date}")
            continue
        
        # Check if required state probability columns exist
        if not all(col in session_df.columns for col in ['p_stochastic', 'p_leftbias', 'p_rightbias']):
            print(f"Missing state probability columns for {subject_id}/{session_date}")
            continue
        
        # Get photometry data for the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result or 'epoched_data' not in session_result:
            print(f"No photometry data found for {subject_id}/{session_date}")
            continue
        
        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
        
        session_count += 1
        
        # Get state probability data for the session
        p_stochastic = session_df['p_stochastic'].values
        p_leftbias = session_df['p_leftbias'].values  
        p_rightbias = session_df['p_rightbias'].values
        
        # Initialize array to track state for each trial
        trial_states = np.full(len(session_df), 'uncertain', dtype=object)
        
        # Determine state for each trial
        for i in range(len(session_df)):
            if p_stochastic[i] >= threshold:
                trial_states[i] = 'stochastic'
            elif p_leftbias[i] >= threshold:
                trial_states[i] = 'leftbiased'
            elif p_rightbias[i] >= threshold:
                trial_states[i] = 'rightbiased'
        
        # Find contiguous state runs
        state_runs = []
        current_state = trial_states[0]
        current_run_start = 0
        
        # Loop through trials to find state runs
        for i in range(1, len(trial_states)):
            if trial_states[i] != current_state:
                # End of a run
                if i - current_run_start >= min_state_duration and current_state != 'uncertain':
                    state_runs.append({
                        'state': current_state, 
                        'start': current_run_start, 
                        'end': i - 1,
                        'length': i - current_run_start
                    })
                # Start new run
                current_state = trial_states[i]
                current_run_start = i
        
        # Handle the last run
        if len(trial_states) - current_run_start >= min_state_duration and current_state != 'uncertain':
            state_runs.append({
                'state': current_state, 
                'start': current_run_start, 
                'end': len(trial_states) - 1,
                'length': len(trial_states) - current_run_start
            })
        
        # Find transitions between states
        transitions = []
        for i in range(len(state_runs) - 1):
            current_run = state_runs[i]
            next_run = state_runs[i+1]
            
            if current_run['state'] != next_run['state']:
                # There's a transition between the end of current run and start of next run
                transition_start = current_run['end'] + 1
                transition_end = next_run['start'] - 1
                
                if transition_end >= transition_start:  # Only if there are actual transition trials
                    transitions.append({
                        'from_state': current_run['state'],
                        'to_state': next_run['state'],
                        'start': transition_start,
                        'end': transition_end,
                        'length': transition_end - transition_start + 1
                    })
        
        # Get valid trial indices
        valid_trials = np.array(session_result['valid_trials'])
        
        # Process state runs to get within-state photometry data
        for run in state_runs:
            state = run['state']
            run_length = run['length']
            
            # Skip uncertain state or short runs
            if state == 'uncertain' or run_length < min_state_duration:
                continue
            
            # Update state run statistics
            state_run_counts[state] += 1
            state_trial_counts[state] += run_length
            
            # Divide the run into three equal parts (early, middle, late)
            part_length = run_length // 3
            
            # Define ranges for each part, ensuring to include all trials
            early_range = range(run['start'], run['start'] + part_length)
            middle_range = range(run['start'] + part_length, run['end'] - part_length + 1)
            late_range = range(run['end'] - part_length + 1, run['end'] + 1)
            
            # Map to valid trial indices and get photometry data
            for trial_idx in early_range:
                if trial_idx < len(session_df):  # Ensure trial index is valid
                    # Find corresponding index in valid_trials
                    photo_idx = np.where(valid_trials == trial_idx)[0]
                    if len(photo_idx) > 0:
                        period_trials[f"{state}_early"].append(session_result['epoched_data'][photo_idx[0]])
            
            for trial_idx in middle_range:
                if trial_idx < len(session_df):
                    photo_idx = np.where(valid_trials == trial_idx)[0]
                    if len(photo_idx) > 0:
                        period_trials[f"{state}_middle"].append(session_result['epoched_data'][photo_idx[0]])
            
            for trial_idx in late_range:
                if trial_idx < len(session_df):
                    photo_idx = np.where(valid_trials == trial_idx)[0]
                    if len(photo_idx) > 0:
                        period_trials[f"{state}_late"].append(session_result['epoched_data'][photo_idx[0]])
        
        # Process transitions
        for transition in transitions:
            from_state = transition['from_state']
            to_state = transition['to_state']
            transition_length = transition['length']
            
            # Skip transitions involving uncertain state
            if from_state == 'uncertain' or to_state == 'uncertain':
                continue
                
            # Update transition statistics
            transition_counts[f"{from_state}_to_{to_state}"] += 1
            
            # Divide transition into early and late parts
            midpoint = transition['start'] + transition_length // 2
            early_trans_range = range(transition['start'], midpoint)
            late_trans_range = range(midpoint, transition['end'] + 1)
            
            # Get photometry data for early transition trials
            for trial_idx in early_trans_range:
                if trial_idx < len(session_df):
                    photo_idx = np.where(valid_trials == trial_idx)[0]
                    if len(photo_idx) > 0:
                        period_name = f"to_{to_state}_from_{from_state}_early"
                        period_trials[period_name].append(session_result['epoched_data'][photo_idx[0]])
            
            # Get photometry data for late transition trials
            for trial_idx in late_trans_range:
                if trial_idx < len(session_df):
                    photo_idx = np.where(valid_trials == trial_idx)[0]
                    if len(photo_idx) > 0:
                        period_name = f"to_{to_state}_from_{from_state}_late"
                        period_trials[period_name].append(session_result['epoched_data'][photo_idx[0]])
    
    if session_count == 0:
        print(f"No valid sessions found for {subject_id}")
        return None
    
    # Print statistics
    print(f"\nProcessed {session_count} sessions for {subject_id}")
    for state in state_types:
        print(f"{state.capitalize()} state: {state_run_counts[state]} runs, {state_trial_counts[state]} total trials")
    
    print("\nTransition counts:")
    for trans, count in transition_counts.items():
        if count > 0:
            print(f"{trans}: {count} transitions")
    
    # Calculate statistics for each period
    period_data = {}
    for period, trials in period_trials.items():
        if len(trials) > 0:
            # Convert to numpy array first before calculating stats
            trials_array = np.array(trials)
            period_data[period] = {
                'trials': trials_array,
                'avg': np.mean(trials_array, axis=0),
                'sem': calculate_sem(trials_array, axis=0),
                'count': len(trials)
            }
        else:
            period_data[period] = {
                'trials': np.array([]),
                'avg': None,
                'sem': None,
                'count': 0
            }
    
    # Print trial counts for each period
    print("\nTrial counts for each period:")
    for period, data in period_data.items():
        print(f"{period}: {data['count']} trials")
    
    # Create plots
    
    # 1. Plot transition periods
    plt.figure(figsize=(14, 8))
    
    # Define colors for transition periods by target state
    transition_colors = {
        'stochastic': 'green',
        'leftbiased': 'blue',
        'rightbiased': 'red'
    }
    
    # Plot all transitions, grouped by target state
    for target_state in state_types:
        for transition_phase in ['early', 'late']:
            # Combine data from transitions to this target state from all source states
            target_periods = [p for p in transition_periods.keys() 
                             if f"to_{target_state}" in p and p.endswith(f"_{transition_phase}")]
            
            # Skip if no data
            valid_periods = [p for p in target_periods if period_data[p]['avg'] is not None]
            if not valid_periods:
                continue
            
            # Get average across all source states, weighted by trial count
            total_count = sum(period_data[p]['count'] for p in valid_periods)
            if total_count == 0:
                continue
                
            weighted_avg = np.zeros_like(period_data[valid_periods[0]]['avg'])
            for p in valid_periods:
                if period_data[p]['count'] > 0:
                    weight = period_data[p]['count'] / total_count
                    weighted_avg += period_data[p]['avg'] * weight
            
            # Line style based on transition phase
            linestyle = '-' if transition_phase == 'early' else '--'
            
            plt.plot(time_axis, weighted_avg,
                   color=transition_colors[target_state], linestyle=linestyle, linewidth=2,
                   label=f"To {target_state.capitalize()} ({transition_phase}, n={total_count})")
    
    # Add vertical line at cue onset
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.xlabel('Time (s) after first lick', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(f'Transition Period Photometry Signals: {subject_id}', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    
    save_figure(plt.gcf(), subject_id, "pooled", "all_transitions_photometry")
    plt.show()
    
    # 2. Plot transitions to each target state
    for target_state in state_types:
        # Skip if no transitions to this state
        target_periods = [p for p in transition_periods.keys() if f"to_{target_state}" in p]
        if not any(period_data[p]['avg'] is not None for p in target_periods):
            continue
            
        plt.figure(figsize=(12, 7))
        
        # Plot transitions to this target state from each source state
        for source_state in [s for s in state_types if s != target_state]:
            for phase in ['early', 'late']:
                period = f"to_{target_state}_from_{source_state}_{phase}"
                
                if period_data[period]['avg'] is not None and period_data[period]['count'] > 0:
                    # Determine color based on source state
                    color = transition_colors[source_state]
                    
                    # Line style based on transition phase
                    linestyle = '-' if phase == 'early' else '--'
                    
                    plt.plot(time_axis, period_data[period]['avg'],
                           color=color, linestyle=linestyle, linewidth=2,
                           label=f"From {source_state.capitalize()} ({phase}, n={period_data[period]['count']})")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        plt.title(f'Transitions to {target_state.capitalize()} State: {subject_id}', fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        save_figure(plt.gcf(), subject_id, "pooled", f"transitions_to_{target_state}")
        plt.show()
    
    # 3. Plot within-state periods
    for state in state_types:
        # Skip if no data for this state
        state_periods = [p for p in within_periods.keys() if p.startswith(f"{state}_")]
        if not any(period_data[p]['avg'] is not None for p in state_periods):
            continue
            
        plt.figure(figsize=(12, 7))
        
        # Define colors for within-state periods
        phase_colors = {
            'early': 'lightblue',
            'middle': 'blue',
            'late': 'darkblue'
        }
        
        # Plot each phase within this state
        for phase in ['early', 'middle', 'late']:
            period = f"{state}_{phase}"
            
            if period_data[period]['avg'] is not None and period_data[period]['count'] > 0:
                plt.plot(time_axis, period_data[period]['avg'],
                       color=phase_colors[phase], linewidth=2,
                       label=f"{phase.capitalize()} (n={period_data[period]['count']})")
        
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        plt.title(f'{state.capitalize()} State Periods: {subject_id}', fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        save_figure(plt.gcf(), subject_id, "pooled", f"{state}_periods_photometry")
        plt.show()
    
    # 4. Compare all states (middle period)
    plt.figure(figsize=(12, 7))
    
    # Define colors for each state
    state_colors = {
        'stochastic': 'green',
        'leftbiased': 'blue',
        'rightbiased': 'red'
    }
    
    # Plot middle period for each state
    valid_states = 0
    for state in state_types:
        period = f"{state}_middle"
        
        if period_data[period]['avg'] is not None and period_data[period]['count'] > 0:
            valid_states += 1
            plt.plot(time_axis, period_data[period]['avg'],
                   color=state_colors[state], linewidth=2,
                   label=f"{state.capitalize()} (n={period_data[period]['count']})")
    
    # Only show plot if at least two states have data
    if valid_states >= 1:
        # Add vertical line at cue onset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s) after first lick', fontsize=16)
        plt.ylabel('ΔF/F', fontsize=16)
        plt.title(f'Comparison Across States (Middle Period): {subject_id}', fontsize=20)
        plt.xlim([-pre_cue_time, post_cue_time])
        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        
        save_figure(plt.gcf(), subject_id, "pooled", "all_states_middle_comparison")
        plt.show()
    
    # Return analysis results
    return {
        'subject_id': subject_id,
        'threshold': threshold,
        'time_axis': time_axis,
        'session_count': session_count,
        'state_run_counts': state_run_counts,
        'state_trial_counts': state_trial_counts,
        'transition_counts': transition_counts,
        'period_data': period_data
    }