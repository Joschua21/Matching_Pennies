import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact
from photometry_analysis import calculate_sem, save_figure, process_session, plot_session_results, check_saved_pooled_results, save_pooled_results, analyze_pooled_data

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global parameters
base_dir = r"Z:\delab\lab-members\joanna\photometry\preprocess"
output_dir = r"Z:\delab\lab-members\joschua\photometry_analysis"
sampling_rate = 120
pre_cue_time = 3
post_cue_time = 5
pre_cue_samples = int(pre_cue_time * sampling_rate)
post_cue_samples = int(post_cue_time * sampling_rate)
total_window_samples = pre_cue_samples + post_cue_samples

PARQUET_PATH = r"Z:\delab\matchingpennies\matchingpennies_datatable.parquet"
CODE_VERSION = "1.0.4"  # Increment this when making analysis changes --> will force recomputation of all data
_SESSION_CACHE = {}

def analyze_behavioral_states(subject_id, win_loss=False, threshold=0.8):
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
        
    Returns:
    --------
    dict: Analysis results including state-specific photometry signals
    """
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None
    
    # Load the parquet file to get state probability data
    try:
        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
        df['date'] = df['date'].astype(str)  # Ensure date is a string
        print(f"Loaded parquet data with columns: {df.columns.tolist()}")
        
        # Check if the required columns exist
        required_columns = ['p_stochastic', 'p_leftbias', 'p_rightbias']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing required columns in parquet data: {missing_cols}")
            return None
    except Exception as e:
        print(f"Error loading parquet data: {e}")
        return None
        
    # Store results
    state_data = {
        'stochastic': {
            'data': [],
            'win_data': [],
            'loss_data': []
        },
        'biased': {
            'data': [],
            'win_data': [],
            'loss_data': []
        }
    }
    
    time_axis = None
    
    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d)) and
                       os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        
        # Get session data from parquet file
        session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
        
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
            
        # Skip sessions with too few trials
        if len(session_result['non_m_trials']) < 10:
            print(f"Skipping {subject_id}/{session_date}, insufficient valid trials")
            continue
            
        # Get reward outcomes for non-missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                 if idx in session_result["non_m_trials"]])
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        
        # Get state probabilities for each trial
        # We need to map the photometry trials to the behavioral data
        behavior_data = session_result['behavioral_data']
        choices = np.array(behavior_data['choice'])
        
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        
        # Get original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]
        
        # Check if lengths match - if not, we have a mapping issue
        if len(orig_non_miss_indices) != len(session_df):
            print(f"Warning: Mismatch between non-missed trials ({len(orig_non_miss_indices)}) and session data ({len(session_df)})")
            print("This may cause incorrect state assignments. Skipping session.")
            continue
            
        # Create a mapping from filtered trial indices to state probabilities
        trial_states = []
        
        # For each valid photometry trial, determine its state
        for orig_idx in session_result["valid_trials"]:
            if orig_idx not in session_result["non_m_trials"]:
                continue
                
            # Find the corresponding row in the behavioral data
            if orig_idx < len(non_miss_mask) and non_miss_mask[orig_idx]:
                # Find the position of this trial in the filtered space
                filtered_idx = np.where(orig_non_miss_indices == orig_idx)[0]
                
                if len(filtered_idx) > 0 and filtered_idx[0] < len(session_df):
                    # Get the corresponding row from session_df
                    row_idx = filtered_idx[0]
                    
                    # Extract state probabilities
                    p_stochastic = session_df.iloc[row_idx]['p_stochastic']
                    p_leftbias = session_df.iloc[row_idx]['p_leftbias']
                    p_rightbias = session_df.iloc[row_idx]['p_rightbias']
                    
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
                
        # For each valid trial, add its photometry data to the appropriate group
        for i, (state, reward) in enumerate(zip(trial_states, reward_outcomes)):
            if state == 'uncertain':
                continue
                
            # Get photometry data for this trial
            photometry_data = session_result['plotting_data'][i]
            
            # Add to overall state data
            state_data[state]['data'].append(photometry_data)
            
            # Add to win/loss subgroups if requested
            if reward == 1:  # Win
                state_data[state]['win_data'].append(photometry_data)
            else:  # Loss
                state_data[state]['loss_data'].append(photometry_data)
    
    # Check if we found any valid trials
    stochastic_count = len(state_data['stochastic']['data'])
    biased_count = len(state_data['biased']['data'])
    
    if stochastic_count + biased_count == 0:
        print(f"No trials meeting state criteria found for {subject_id}")
        return None
        
    print(f"\nFound {stochastic_count} stochastic trials and {biased_count} biased trials")
    
    # Convert lists to numpy arrays
    for state in ['stochastic', 'biased']:
        if state_data[state]['data']:
            state_data[state]['data'] = np.array(state_data[state]['data'])
        if state_data[state]['win_data']:
            state_data[state]['win_data'] = np.array(state_data[state]['win_data'])
        if state_data[state]['loss_data']:
            state_data[state]['loss_data'] = np.array(state_data[state]['loss_data'])
    
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
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    if win_loss:
        # Plot win/loss separated for each state
        colors = {
            'stochastic_win': 'mediumblue',
            'stochastic_loss': 'royalblue',
            'biased_win': 'darkred',
            'biased_loss': 'indianred'
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
    else:
        # Plot overall state averages without separating by outcome
        colors = {
            'stochastic': 'blue',
            'biased': 'red'
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
    
    # Add reference lines
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    
    win_loss_suffix = "Win-Loss" if win_loss else "Combined"
    plt.title(f'Photometry by Behavioral State ({win_loss_suffix}): {subject_id}', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    
    # Add threshold info
    plt.figtext(0.01, 0.01, f"State threshold: {threshold}", fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", 
               f"behavioral_states_{'winloss' if win_loss else 'combined'}")
    
    plt.show()
    
    # Return results
    return {
        'subject_id': subject_id,
        'time_axis': time_axis,
        'stochastic_trials': stochastic_count,
        'biased_trials': biased_count,
        'state_data': state_data,
        'threshold': threshold
    }

