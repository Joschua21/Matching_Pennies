import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact

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
CODE_VERSION = "1.0.9"  # Increment this when making analysis changes --> will force recomputation of all data
_SESSION_CACHE = {}


def clear_memory():
    #used to clear the session cache to reduce working memory utilization
    global _SESSION_CACHE
    _SESSION_CACHE = {}
    import gc
    gc.collect()

def calculate_sem(data, axis=0):
    """Calculate Standard Error of Mean (SEM)"""
    std = np.std(data, axis=axis)
    n = data.shape[axis]
    return std / np.sqrt(n)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory {directory}")


def get_output_path(subject_id, session_date):
    subject_dir = os.path.join(output_dir, subject_id)
    session_dir = os.path.join(subject_dir, session_date)

    ensure_directory_exists(subject_dir)
    ensure_directory_exists(session_dir)

    return session_dir


def load_filtered_behavior_data(protocol_filter="MatchingPennies", subject_id=None, ignore=0, behavior_df=None):
    """
    Load behavioral data, filtering for a specific protocol, subject, and ignore status.
    Returns the filtered DataFrame for reuse across functions.
    
    Parameters:
    -----------
    protocol_filter : str, optional
        String that must be contained in protocol name
    subject_id : str, optional
        If provided, filter for this specific subject
    ignore : int or None, optional
        If provided, filter for this ignore value (default=0)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded dataframe to filter instead of loading from parquet
        
    Returns:
    --------
    pandas.DataFrame: Filtered behavioral data
    """
    try:
        if behavior_df is None:
            # Load from parquet if no dataframe provided
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
        else:
            # Use the provided dataframe
            df = behavior_df.copy()
            if 'date' in df.columns and df['date'].dtype != str:
                df['date'] = df['date'].astype(str)
        
        # Create a mask for protocol filtering
        mask = df['protocol'].str.contains(protocol_filter, na=False) 
        
        # Add ignore filter if provided
        if ignore is not None:
            mask &= (df["ignore"] == ignore)
            
        # Add subject filter if provided
        if subject_id is not None:
            mask &= (df["subjid"] == subject_id)
            
        # Apply all filters at once
        filtered_df = df[mask]
        
        # Print summary info
        source = 'provided dataframe' if behavior_df is not None else 'parquet file'
        filter_info = f"protocol='{protocol_filter}', ignore={ignore}"
        if subject_id:
            filter_info += f", subject='{subject_id}'"
            
        print(f"Using {source}: {len(filtered_df)} trials matching {filter_info}")
        return filtered_df
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None

def load_behavior_data(subject_id, session_date, behavior_df=None):
    """Load behavioral data for a specific subject and session date"""
    try:
        if behavior_df is None:
            # Load the parquet file with pyarrow
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
        else:
            # Use the provided dataframe
            df = behavior_df.copy()
            if df['date'].dtype != str:
                df['date'] = df['date'].astype(str)
                
        session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
        
        if session_data.empty:
            print(f"No behavioral data found for {subject_id} on {session_date}")

        lick_timestamps = session_data["LickTimestamp"].values
        sound_timestamps = session_data["SoundTimestamp"].values
        corrections = (lick_timestamps - sound_timestamps) * sampling_rate

        return {
            "choice": session_data["choice"].tolist(),  # List of 'R' or 'L'
            "lick_timestamps": lick_timestamps,  # List of lick times
            "soundtimestamps": sound_timestamps,
            "corrections": corrections,
            "reward": session_data["reward"].tolist(),
        }

    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None
    

def find_pkl_file(directory):
    """Find the first .pkl file in the given directory"""
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))
    return pkl_files[0] if pkl_files else None


def check_saved_results(subject_id, session_date):
    """Checks if calculation has already be done and results are stored in a .pkl file. If Code Version changed, recomputes and overwrites results file"""
    session_dir = get_output_path(subject_id, session_date)
    results_file = os.path.join(session_dir, "analysis_results.pkl")

    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                result = pickle.load(f)
            # Check if saved result has version info and matches current version
            if 'code_version' not in result or result['code_version'] != CODE_VERSION:
                print(f"Saved results for {subject_id}/{session_date} are from different code version. Recomputing...")
                return None
            print(f"Loaded saved results for {subject_id}/{session_date}")
            return result
        except Exception as e:
            print(f"Error loading saved results: {e}")

    return None


def save_results(result, subject_id, session_date):
    if not result:
        return

    session_dir = get_output_path(subject_id, session_date)
    results_file = os.path.join(session_dir, "analysis_results.pkl")

    try:
        with open(results_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved saved results for {subject_id}/{session_date} to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def save_figure(fig, subject_id, session_date, fig_name="figure"):
    """Save figure to file with error handling"""
    if fig is None:
        print("Warning: No figure to save")
        return

    session_dir = get_output_path(subject_id, session_date)
    fig_file = os.path.join(session_dir, f"{fig_name}.png")

    try:
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure for {subject_id}/{session_date} to {fig_file}")
    except Exception as e:
        print(f"Error saving figure: {e}")


def process_session(subject_id, session_date, force_recompute=False, use_global_cache=True, behavior_df=None):
    """Process a single session for a given subject"""


    cache_key = f"{subject_id}/{session_date}"
    if use_global_cache and cache_key in _SESSION_CACHE and not force_recompute:
        return _SESSION_CACHE[cache_key]
    
    # Then check saved results
    if not force_recompute:
        saved_result = check_saved_results(subject_id, session_date)  
        if saved_result is not None:
            return saved_result

    full_dir = os.path.join(base_dir, subject_id, session_date)
    deltaff_file = os.path.join(full_dir, "deltaff.npy")
    pkl_file = find_pkl_file(full_dir)

    if not pkl_file or not os.path.exists(deltaff_file):
        print(f"Missing files for {subject_id}/{session_date}")
        return None

    try:
        deltaff_data = np.load(deltaff_file)
        print(f"{subject_id}/{session_date}: Loaded deltaff data with shape: {deltaff_data.shape}")

        with open(pkl_file, "rb") as f:
            pkl_data = pickle.load(f)
        print(f"{subject_id}/{session_date}: Loaded pkl data: {os.path.basename(pkl_file)}")

        # Extract pulse indices
        pulse_indices = pkl_data["pulse_inds_1"]
        num_trials = len(pulse_indices)
        print(f"{subject_id}/{session_date}: Found {num_trials} pulse indices")

        behavior_data = load_behavior_data(subject_id, session_date, behavior_df)
        if behavior_data is None:
            print(f"Warning: No behavioral data found for {subject_id} on {session_date}")
            return None

        corrections = np.array(behavior_data['corrections'])

        if len(pulse_indices) != len(corrections):
            print(
                f"Mismatch between pulse indices and corrections for {subject_id}/{session_date}. Skipping correction")
            corrected_indices = pulse_indices
        else:
            corrected_indices = pulse_indices + corrections.astype(int)

        # Epoch the data
        epoched_data = np.zeros((num_trials, total_window_samples))
        valid_trials = []
        choice_values = np.array(behavior_data["choice"])

        for i, index in enumerate(corrected_indices):
            if index >= pre_cue_samples and index + post_cue_samples < len(deltaff_data):
                start_idx = index - pre_cue_samples
                end_idx = index + post_cue_samples
                epoched_data[i] = deltaff_data[start_idx:end_idx]
                valid_trials.append(i)
            else:
                epoched_data[i] = np.nan

        # Count actual M trials instead of using difference
        non_m_trials = []
        m_trial_count = 0
        for i in valid_trials:
            if i < len(choice_values):
                if choice_values[i] == 'M':
                    m_trial_count += 1
                else:
                    non_m_trials.append(i)

        plotting_data = epoched_data[non_m_trials]
        all_epoched_data = epoched_data[valid_trials]

        print(f"{subject_id}/{session_date}: Successfully epoched {len(valid_trials)} valid trials")
        print(f"{subject_id}/{session_date}: {m_trial_count} missed trials will be excluded from statistics and plots")

        # Calculate statistics
        time_axis = np.linspace(-pre_cue_time, post_cue_time, total_window_samples, endpoint=False)
        trial_average = np.mean(plotting_data, axis=0)
        trial_sem = calculate_sem(plotting_data, axis=0)

        reward_outcomes = np.array(behavior_data["reward"])[valid_trials]
        choices = np.array(behavior_data["choice"])[valid_trials]

        result = {
            'code_version': CODE_VERSION,  # Add version to saved results
            'subject_id': subject_id,
            'session_date': session_date,
            'epoched_data': all_epoched_data,
            'time_axis': time_axis,
            'trial_average': trial_average,
            'trial_sem': trial_sem,
            'valid_trials': valid_trials,
            'non_m_trials': non_m_trials,
            'num_trials': num_trials,
            'behavioral_data': behavior_data,
            'reward_outcomes': reward_outcomes,
            'choices': choices,
            'plotting_data': plotting_data
        }

        save_results(result, subject_id, session_date)
        if use_global_cache:
            _SESSION_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"Error processing session {subject_id}/{session_date}: {e}")
        return None


def plot_session_results(analysis_result, show_heatmap=False, win_loss=False, save_fig=True):
    """Plot the results for a single session with an optional heatmap of individual trials."""
    if not analysis_result:
        print("No results to plot")
        return

    # Set up subplots: 1x2 if heatmap is enabled, else just 1 plot
    fig, axes = plt.subplots(1, 2 if show_heatmap else 1, figsize=(16 if show_heatmap else 12, 7))

    # Ensure `axes` is always iterable
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First subplot: Trial Average ± Standard Deviation
    ax1 = axes[0]
    if win_loss and isinstance(analysis_result, dict) and "reward_outcomes" in analysis_result:
        non_m_indices = np.array([i for i, idx in enumerate(analysis_result["valid_trials"])
                                  if idx in analysis_result["non_m_trials"]])
        filtered_reward_outcomes = analysis_result["reward_outcomes"][non_m_indices]

        # Split trials by reward outcome
        rewarded_trials = analysis_result["plotting_data"][filtered_reward_outcomes == 1]
        unrewarded_trials = analysis_result["plotting_data"][filtered_reward_outcomes == 0]

        # Plot rewarded trials
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            ax1.fill_between(analysis_result['time_axis'], rewarded_avg - rewarded_sem,
                             rewarded_avg + rewarded_sem, color='lightgreen', alpha=0.4, label='Rewarded ± SEM')
            ax1.plot(analysis_result['time_axis'], rewarded_avg, color='green', linewidth=2.5, label='Rewarded Avg')

        # Plot unrewarded trials
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            ax1.fill_between(analysis_result['time_axis'], unrewarded_avg - unrewarded_sem,
                             unrewarded_avg + unrewarded_sem, color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')
            ax1.plot(analysis_result['time_axis'], unrewarded_avg, color='darkorange', linewidth=2.5,
                     label='Unrewarded Avg')

    else:  # Default behavior
        trial_sem = calculate_sem(analysis_result['plotting_data'], axis=0)
        ax1.fill_between(analysis_result['time_axis'],
                         analysis_result['trial_average'] - trial_sem,
                         analysis_result['trial_average'] + trial_sem,
                         color='lightgreen', alpha=0.4, label='Mean ± SEM')

        ax1.plot(analysis_result['time_axis'], analysis_result['trial_average'],
                 color='darkgreen', linewidth=2.5, label='Trial Average')

    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Time (s)', fontsize=16)
    ax1.set_ylabel('ΔF/F', fontsize=16)
    ax1.set_title(f'Photometry Response: {analysis_result["subject_id"]} - {analysis_result["session_date"]}',
                  fontsize=20)
    ax1.set_xlim([-pre_cue_time, post_cue_time])
    ax1.legend(loc='upper right', fontsize=16)

    # Add statistics text box
    stats_text = (f"Trials: {len(analysis_result['non_m_trials'])} (excluding missed trials)\n"
                  f"Peak: {np.max(analysis_result['trial_average']):.4f}\n"
                  f"Baseline: {np.mean(analysis_result['trial_average'][:pre_cue_samples]):.4f}")
    ax1.text(-pre_cue_time + 0.2, np.max(analysis_result['trial_average']) * 0.9, stats_text,
             bbox=dict(facecolor='white', alpha=0.7), fontsize=16)

    # Second subplot: Heatmap (if enabled)
    if show_heatmap:
        ax2 = axes[1]
        trial_data = analysis_result['plotting_data']
        im = ax2.imshow(trial_data,
                        aspect='auto',
                        extent=[-pre_cue_time, post_cue_time, 0, len(trial_data)],
                        cmap='viridis',
                        interpolation='nearest')

        ax2.set_xlabel('Time (s)', fontsize=16)
        ax2.set_ylabel('Trial Number', fontsize=16)
        ax2.set_title('Trial-by-Trial Heatmap', fontsize=20)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

        # Add colorbar
        plt.colorbar(im, ax=ax2, label='ΔF/F')

    plt.tight_layout()

    if save_fig:
        save_figure(fig, analysis_result["subject_id"], analysis_result["session_date"],
                    f"session_results{'_heatmap' if show_heatmap else ''}{'_winloss' if win_loss else ''}")

    plt.show()
    return analysis_result


def check_saved_pooled_results(subject_id, win_loss=False):
    subject_dir = os.path.join(output_dir, subject_id)
    filename = f"pooled_results{'_winloss' if win_loss else ''}.pkl"
    pooled_file = os.path.join(subject_dir, filename)

    if os.path.exists(pooled_file):
        try:
            with open(pooled_file, "rb") as f:
                results = pickle.load(f)
            # Check version
            if 'code_version' not in results or results['code_version'] != CODE_VERSION:
                print(f"Saved pooled results for {subject_id} are from different code version. Recomputing...")
                return None
            print(f"Loaded saved pooled results for {subject_id}")
            return results
        except Exception as e:
            print(f"Error loading saved pooled results: {e}")
    return None


def save_pooled_results(result, subject_id, win_loss=False):
    """Save pooled analysis results to file"""
    if not result:
        return

    subject_dir = os.path.join(output_dir, subject_id)
    ensure_directory_exists(subject_dir)

    filename = f"pooled_results{'_winloss' if win_loss else ''}.pkl"
    pooled_file = os.path.join(subject_dir, filename)

    try:
        with open(pooled_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved pooled results to {pooled_file}")
    except Exception as e:
        print(f"Error saving pooled results: {e}")

def analyze_pooled_data(subject_id, win_loss=False, force_recompute=False, fig=None, show_session_traces=False, behavior_df=None):
    """Analyze and visualize pooled data for a subject"""
    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(12, 7))

    # Check for saved pooled results
    if not force_recompute:
        saved_results = check_saved_pooled_results(subject_id, win_loss)
        if saved_results is not None:
            print(f"Using saved results for {subject_id}")
            
            # Create a visualization with saved data
            # This way, we can control show_session_traces without recomputing
            time_axis = saved_results['time_axis']
            
            # Create a new figure for the visualization
            plt.figure(figsize=(12, 7))
            
            # Optionally plot session traces
            if show_session_traces and 'session_averages' in saved_results:
                blue_colors = plt.cm.Blues(np.linspace(0.3, 1, len(saved_results['session_dates'])))
                for idx, (session_date, session_avg) in enumerate(zip(saved_results['session_dates'], saved_results['session_averages'])):
                    plt.plot(time_axis, session_avg,
                            alpha=0.6, linewidth=1, linestyle='-',
                            color=blue_colors[idx],
                            label=f"Session {session_date}")
            
            # Plot the main data (win/loss or average)
            if win_loss:
                if saved_results['rewarded_avg'] is not None:
                    plt.fill_between(time_axis, 
                                   saved_results['rewarded_avg'] - saved_results['rewarded_sem'],  
                                   saved_results['rewarded_avg'] + saved_results['rewarded_sem'],  
                                   color='lightgreen', alpha=0.4, label='Rewarded ± SEM')  
                    plt.plot(time_axis, saved_results['rewarded_avg'], 
                           color='green', linewidth=2.5, label='Rewarded Avg')

                if saved_results['unrewarded_avg'] is not None:
                    plt.fill_between(time_axis, 
                                   saved_results['unrewarded_avg'] - saved_results['unrewarded_sem'],  
                                   saved_results['unrewarded_avg'] + saved_results['unrewarded_sem'],  
                                   color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')  
                    plt.plot(time_axis, saved_results['unrewarded_avg'], 
                           color='darkorange', linewidth=2.5, label='Unrewarded Avg')
            else:
                plt.fill_between(time_axis,
                               saved_results['pooled_average'] - saved_results['pooled_sem'],  
                               saved_results['pooled_average'] + saved_results['pooled_sem'],  
                               color='lightgreen', alpha=0.4,
                               label='Mean ± SEM')  
                plt.plot(time_axis, saved_results['pooled_average'], 
                       color='green', linewidth=2.5, label='Overall Avg')
            
            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)
            plt.title(f'Pooled Photometry Response: {subject_id} ({len(saved_results["session_dates"])} sessions)', fontsize=14)
            plt.xlim([-pre_cue_time, post_cue_time])
            
            # Fix the legend - limit session traces if too many
            if show_session_traces and 'session_dates' in saved_results and len(saved_results['session_dates']) > 5:
                handles, labels = plt.gca().get_legend_handles_labels()
                # Find how many session labels we have
                session_handles = [h for h, l in zip(handles, labels) if l.startswith("Session")]
                non_session_handles = [h for h, l in zip(handles, labels) if not l.startswith("Session")]
                session_labels = [l for l in labels if l.startswith("Session")]
                non_session_labels = [l for l in labels if not l.startswith("Session")]
                
                # Only keep the first 5 session labels
                limited_session_handles = session_handles[:5]
                limited_session_labels = session_labels[:5]

                # Show limited legend
                plt.legend(limited_session_handles + non_session_handles, 
                          limited_session_labels + non_session_labels, 
                          loc='upper right', fontsize=10)
            else:
                plt.legend(loc='upper right', fontsize=10)
                
            plt.tight_layout()
            
            # Add statistics - count only non-'M' trials
            total_trials = saved_results['total_trials']
            stats_text = (f"Total Sessions: {len(saved_results['session_dates'])}\n"
                        f"Total Trials: {total_trials} (excluding 'M' choices)\n"
                        f"Peak: {np.max(saved_results['pooled_average']):.4f}\n"
                        f"Baseline: {np.mean(saved_results['pooled_average'][:pre_cue_samples]):.4f}")
            plt.text(-pre_cue_time + 0.2, saved_results['pooled_average'].max() * 1.2, stats_text,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Save the figure with appropriate suffix
            trace_suffix = "_with_sessions" if show_session_traces else ""
            save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{trace_suffix}{'_winloss' if win_loss else ''}")
            
            plt.show()
            return saved_results

    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    session_dates = []
    session_averages = []

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
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

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy")) and
                      (d in matching_pennies_sessions or not matching_pennies_sessions)])
    
    num_sessions = len(sessions)
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))  # Create blue gradient

    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            # Pass behavior_df to process_session to reuse data
            result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if result:
                if len(result['non_m_trials']) < 100:
                    print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(result['non_m_trials'])}).")
                    continue

                all_sessions.append(result)
                all_plotting_data.append(result['plotting_data'])
                session_dates.append(session_date)
                session_averages.append(result['trial_average'])

    if not all_sessions:
        print(f"No processed sessions found for subject {subject_id}")
        return None

    # Concatenate all trials from all sessions
    pooled_data = np.vstack(all_plotting_data)
    pooled_average = np.mean(pooled_data, axis=0)
    time_axis = all_sessions[0]['time_axis']  

    # Create the pooled plot
    plt.figure(figsize=(12, 7))

    rewarded_avg = None
    unrewarded_avg = None
    rewarded_data = []
    unrewarded_data = []

    if win_loss:
        for session in all_sessions:
            non_m_indices = np.array(
                [i for i, idx in enumerate(session["valid_trials"]) if idx in session["non_m_trials"]])
            filtered_reward_outcomes = session["reward_outcomes"][non_m_indices]

            session_plot_data = session["plotting_data"]

            if len(session_plot_data) > 0:
                session_rewarded = session_plot_data[filtered_reward_outcomes == 1]
                session_unrewarded = session_plot_data[filtered_reward_outcomes == 0]

                if len(session_rewarded) > 0:
                    rewarded_data.append(session_rewarded)
                if len(session_unrewarded) > 0:
                    unrewarded_data.append(session_unrewarded)

        # Filter out empty arrays before stacking
        rewarded_data = np.vstack(rewarded_data) if rewarded_data else np.array([])
        unrewarded_data = np.vstack(unrewarded_data) if unrewarded_data else np.array([])

        # Compute averages and std deviations
        if rewarded_data.size > 0:
            rewarded_avg = np.mean(rewarded_data, axis=0)
            rewarded_sem = calculate_sem(rewarded_data, axis=0)  
            plt.fill_between(time_axis, 
                             rewarded_avg - rewarded_sem,  
                             rewarded_avg + rewarded_sem,  
                             color='lightgreen', alpha=0.4, label='Rewarded ± SEM')  
            plt.plot(time_axis, rewarded_avg, color='green', linewidth=2.5, label='Rewarded Avg')

        if unrewarded_data.size > 0:
            unrewarded_avg = np.mean(unrewarded_data, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_data, axis=0) 
            plt.fill_between(time_axis, 
                             unrewarded_avg - unrewarded_sem,  
                             unrewarded_avg + unrewarded_sem,  
                             color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')  
            plt.plot(time_axis, unrewarded_avg, color='darkorange', linewidth=2.5, label='Unrewarded Avg')

    else:
        pooled_sem = calculate_sem(pooled_data, axis=0)  
        plt.fill_between(time_axis,
                         pooled_average - pooled_sem,  
                         pooled_average + pooled_sem,  
                         color='lightgreen', alpha=0.4,
                         label='Mean ± SEM')  
        plt.plot(time_axis, pooled_average, color='green', linewidth=2.5, label='Overall Avg')

    if show_session_traces:
        # Plot individual session averages with blue gradient
        for idx, (session_date, session_avg) in enumerate(zip(session_dates, session_averages)):
            plt.plot(time_axis, session_avg,
                     alpha=0.6, linewidth=1, linestyle='-',
                     color=blue_colors[idx],  # Use blue gradient
                     label=f"Session {session_date}")
            
    # Add a vertical line at the cue onset (time=0)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(f'Pooled Photometry Response: {subject_id} ({len(all_sessions)} sessions)', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])

    # Limit legend items if too many sessions
    if len(all_sessions) > 5:
        handles, labels = plt.gca().get_legend_handles_labels()
        limited_handles = handles[:8]
        limited_labels = labels[:8]
        plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=16)
    else:
        plt.legend(loc='upper right', fontsize=16)

    plt.tight_layout()

    # Add statistics - count only non-'M' trials
    total_trials = sum(len(session['non_m_trials']) for session in all_sessions)
    stats_text = (f"Total Sessions: {len(all_sessions)}\n"
                  f"Total Trials: {total_trials} (excluding 'M' choices)\n"
                  f"Peak: {np.max(pooled_average):.4f}\n"
                  f"Baseline: {np.mean(pooled_average[:pre_cue_samples]):.4f}")
    plt.text(-pre_cue_time + 0.2, pooled_average.max() * 1.2, stats_text,
             bbox=dict(facecolor='white', alpha=0.7))

    # Prepare pooled result
    pooled_result = {
        'code_version': CODE_VERSION,
        'subject_id': subject_id,
        'session_dates': session_dates,
        'pooled_data': pooled_data,
        'pooled_average': pooled_average,
        'pooled_sem': calculate_sem(pooled_data, axis=0),  
        'time_axis': time_axis,
        'total_trials': total_trials,
        'session_averages': session_averages,
        'rewarded_avg': rewarded_avg,
        'rewarded_sem': calculate_sem(rewarded_data, axis=0) if rewarded_data.size > 0 else None, 
        'unrewarded_avg': unrewarded_avg,
        'unrewarded_sem': calculate_sem(unrewarded_data, axis=0) if unrewarded_data.size > 0 else None 
    }

    # Save pooled results
    save_pooled_results(pooled_result, subject_id, win_loss)

    # Save the figure
    trace_suffix = "_with_sessions" if show_session_traces else ""
    save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{trace_suffix}{'_winloss' if win_loss else ''}")

    plt.show()
    return pooled_result


def analyze_all_subjects(win_loss=False, force_recompute=False):
    """Process all subjects and return results without saving"""
    # Find all subject directories
    all_results = {}
    all_subjects = []

    for subject_id in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        print(f"\nProcessing subject: {subject_id}")

        # Check if the subject has any valid sessions
        has_valid_sessions = False
        for session_date in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_date)
            if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
                has_valid_sessions = True
                break

        if has_valid_sessions:
            all_subjects.append(subject_id)
            # Pass win_loss parameter to analyze_pooled_data
            pooled_result = analyze_pooled_data(subject_id, win_loss=win_loss)
            if pooled_result:
                all_results[subject_id] = pooled_result

    print(f"\nProcessed {len(all_results)} subjects with data")
    return all_results, all_subjects


def select_and_visualize(show_heatmap=False, win_loss=False, force_recompute=False):
    """Interactive function to select and visualize data"""
    # Get list of all subjects
    subjects = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))]

    if not subjects:
        print(f"No subject directories found in {base_dir}")
        return

    print("Available subjects:")
    for i, subj in enumerate(subjects):
        print(f"{i + 1}. {subj}")

    # Select a subject
    try:
        subj_idx = int(input("Enter subject number: ")) - 1
        if subj_idx < 0 or subj_idx >= len(subjects):
            print("Invalid subject number")
            return

        selected_subject = subjects[subj_idx]
    except ValueError:
        print("Please enter a valid number")
        return

    # Ask for win/loss analysis preference
    win_loss_input = input("Show win/loss analysis? (y/n): ").lower()
    win_loss = win_loss_input.startswith('y')

    # Get sessions for this subject
    subject_dir = os.path.join(base_dir, selected_subject)
    sessions = [d for d in os.listdir(subject_dir)
                if os.path.isdir(os.path.join(subject_dir, d)) and
                os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))]

    if not sessions:
        print(f"No valid sessions found for subject {selected_subject}")
        return

    print(f"\nAvailable sessions for {selected_subject}:")
    for i, sess in enumerate(sessions):
        print(f"{i + 1}. {sess}")
    print(f"{len(sessions) + 1}. [POOLED] All sessions")

    # Select a session or pooled analysis
    try:
        sess_idx = int(input("Enter session number or select pooled analysis: ")) - 1
        if sess_idx < 0 or sess_idx > len(sessions):
            print("Invalid selection")
            return

        if sess_idx == len(sessions):  # Pooled analysis selected
            print(f"\nGenerating pooled analysis for {selected_subject}...")
            analyze_pooled_data(selected_subject, win_loss=win_loss)
        else:  # Single session selected
            selected_session = sessions[sess_idx]
            print(f"\nAnalyzing session {selected_subject}/{selected_session}...")
            result = process_session(selected_subject, selected_session)
            if result:
                plot_session_results(result, show_heatmap=show_heatmap, win_loss=win_loss)
            else:
                print("Failed to process session")

    except ValueError:
        print("Please enter a valid number")
        return


def analyze_specific_session(subject_id, session_date, show_heatmap=False, win_loss=False):
    """Analyze and visualize a specific session"""
    print(f"Analyzing session {subject_id}/{session_date}...")
    analysis_result = process_session(subject_id, session_date)
    if analysis_result:
        return plot_session_results(analysis_result, show_heatmap=show_heatmap, win_loss=win_loss)
    return None


def pooled_results(subject_id, win_loss=False, force_recompute=False, show_session_traces=False, behavior_df=None):
    """Analyze and visualize pooled results for a subject"""
    print(f"Analyzing pooled results for subject {subject_id}...")
    return analyze_pooled_data(subject_id, win_loss=win_loss, force_recompute=force_recompute,
                               show_session_traces=show_session_traces, behavior_df=behavior_df)


def all_results(win_loss=False, force_recompute=False):
    """Analyze and visualize results for all subjects"""
    print("Analyzing all subjects...")
    return analyze_all_subjects(win_loss=win_loss, force_recompute=force_recompute)


def analyze_group_reward_rates(behavior_df=None, min_trials=100, save_fig=True, subjids=None):
    """
    Calculate and plot average reward rates across sessions for all subjects chronologically.

    Parameters:
    -----------
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavioral dataframe to analyze. If None, will load from parquet.
    min_trials : int, optional
        Minimum number of trials required to include a session (default=100)
    save_fig : bool, optional
        Whether to save the generated figure (default=True)
    subjids : list, optional
        List of subject IDs to include in the analysis. If None, use all subjects.

    Returns:
    --------
    dict: Dictionary with reward rate data by subject and session
    """
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            print(f"Loaded behavior data from parquet file: {len(behavior_df)} rows")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None

    # Prepare storage for results
    results = {
        'subjects': [],
        'session_reward_rates': {},
        'max_sessions': 0,
        'subject_avg_rates': {},  # Store average reward rate per subject
        'subject_session_counts': {}  # Store session count per subject
    }

    # Get unique subjects, filtered by provided list if specified
    all_subjects = behavior_df['subjid'].unique()
    if subjids is not None:
        subjects = [s for s in all_subjects if s in subjids]
        print(f"Filtering {len(all_subjects)} subjects to {len(subjects)} specified subjects")
    else:
        subjects = all_subjects

    results['subjects'] = list(subjects)
    print(f"Analyzing {len(subjects)} subjects")

    # Process each subject
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_data = behavior_df[behavior_df['subjid'] == subject_id]

        # Sort sessions chronologically
        sessions = sorted(subject_data['date'].unique())
        session_reward_rates = []

        for session_date in sessions:
            session_df = subject_data[subject_data['date'] == session_date]

            # Skip sessions with too few trials
            if len(session_df) < min_trials:
                print(f"  Skipping {subject_id}/{session_date}, fewer than {min_trials} trials ({len(session_df)})")
                continue

            # Get rewards for this session
            rewards = session_df['reward'].values
            window_size = 20
            reward_rates = []
            overall_rate = np.mean(rewards)

            # Calculate moving average reward rate
            for i in range(len(rewards)):
                if i < window_size:
                    available_data = rewards[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
                else:
                    rate = np.mean(rewards[i - window_size + 1:i + 1])
                reward_rates.append(rate)

            # Calculate average reward rate for the session
            session_avg_rate = np.mean(reward_rates)
            session_reward_rates.append(session_avg_rate)
            print(
                f"  {subject_id}/{session_date}: Average reward rate = {session_avg_rate:.3f} ({len(session_df)} trials)")

        # Store session data for this subject
        results['session_reward_rates'][subject_id] = session_reward_rates
        results['max_sessions'] = max(results['max_sessions'], len(session_reward_rates))

        # Calculate and store the average reward rate across all sessions for this subject
        if session_reward_rates:
            avg_reward_rate = np.mean(session_reward_rates)
            results['subject_avg_rates'][subject_id] = avg_reward_rate
            results['subject_session_counts'][subject_id] = len(session_reward_rates)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Create colors for subjects
    num_subjects = len(subjects)
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_subjects))

    # Plot each subject's reward rate progression
    for i, subject_id in enumerate(subjects):
        rates = results['session_reward_rates'][subject_id]
        if rates:
            # Create enhanced legend label with session count and average reward rate
            session_count = results['subject_session_counts'][subject_id]
            avg_rate = results['subject_avg_rates'][subject_id]
            label = f"{subject_id} (Sessions: {session_count}, RR: {avg_rate:.2f})"

            plt.plot(range(1, len(rates) + 1), rates, 'o-',
                     color=colors[i], linewidth=2, label=label)

    # Add a reference line at 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)

    # Set axis labels and title
    plt.xlabel('Session Number', fontsize=14)
    plt.ylabel('Average Reward Rate', fontsize=14)
    plt.title('Reward Rate Progression Across Sessions by Subject', fontsize=16)

    # Set axis limits
    plt.xlim(0.5, results['max_sessions'] + 0.5)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add legend (outside plot if many subjects)
    if num_subjects > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(loc='upper right')

    plt.tight_layout()

    # Save the figure
    if save_fig:
        figure_path = os.path.join(output_dir, "group_analysis")
        ensure_directory_exists(figure_path)
        fig_file = os.path.join(figure_path, f"group_reward_rates.png")

        try:
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_file}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    plt.show()


def analyze_group_computer_confidence(behavior_df=None, min_trials=100, save_fig=True, subjids=None):
    """
    Calculate and plot average computer confidence across sessions for all subjects chronologically.

    Parameters:
    -----------
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavioral dataframe to analyze. If None, will load from parquet.
    min_trials : int, optional
        Minimum number of trials required to include a session (default=100)
    save_fig : bool, optional
        Whether to save the generated figure (default=True)
    subjids : list, optional
        List of subject IDs to include in the analysis. If None, use all subjects.

    Returns:
    --------
    dict: Dictionary with computer confidence data by subject and session
    """
    # Load behavior data if not provided
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            print(f"Loaded behavior data from parquet file: {len(behavior_df)} rows")
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None

    # Check if min_pvalue is in the dataframe
    if 'min_pvalue' not in behavior_df.columns:
        print("Error: 'min_pvalue' column not found in behavior data")
        return None

    # Prepare storage for results
    results = {
        'subjects': [],
        'session_confidence': {},
        'max_sessions': 0,
        'subject_avg_confidence': {},  # Store average confidence per subject
        'subject_session_counts': {}  # Store session count per subject
    }

    # Get unique subjects, filtered by provided list if specified
    all_subjects = behavior_df['subjid'].unique()
    if subjids is not None:
        subjects = [s for s in all_subjects if s in subjids]
        print(f"Filtering {len(all_subjects)} subjects to {len(subjects)} specified subjects")
    else:
        subjects = all_subjects

    results['subjects'] = list(subjects)
    print(f"Analyzing {len(subjects)} subjects")

    # Process each subject
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_data = behavior_df[behavior_df['subjid'] == subject_id]

        # Sort sessions chronologically
        sessions = sorted(subject_data['date'].unique())
        session_confidence = []

        for session_date in sessions:
            session_df = subject_data[subject_data['date'] == session_date]

            # Skip sessions with too few trials
            if len(session_df) < min_trials:
                print(f"  Skipping {subject_id}/{session_date}, fewer than {min_trials} trials ({len(session_df)})")
                continue

            # Get p-values for this session
            p_values = session_df['min_pvalue'].values

            # Replace NaN values with 1.0 (no confidence)
            p_values = np.nan_to_num(p_values, nan=1.0)

            # Apply cutoff for very small p-values (p < 10^-12)
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)

            # Calculate confidence as -log10(p)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 20
            window_size = 20
            confidence_values = []
            overall_confidence = np.mean(confidence)

            # Calculate moving average
            for i in range(len(confidence)):
                if i < window_size:
                    available_data = confidence[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    conf = (np.sum(
                        available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                else:
                    conf = np.mean(confidence[i - window_size + 1:i + 1])
                confidence_values.append(conf)

            # Calculate average confidence for the session
            session_avg_conf = np.mean(confidence_values)
            session_confidence.append(session_avg_conf)
            print(
                f"  {subject_id}/{session_date}: Average confidence = {session_avg_conf:.3f} ({len(session_df)} trials)")

        # Store session data for this subject
        results['session_confidence'][subject_id] = session_confidence
        results['max_sessions'] = max(results['max_sessions'], len(session_confidence))

        # Calculate and store the average confidence across all sessions for this subject
        if session_confidence:
            avg_confidence = np.mean(session_confidence)
            results['subject_avg_confidence'][subject_id] = avg_confidence
            results['subject_session_counts'][subject_id] = len(session_confidence)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Create colors for subjects
    num_subjects = len(subjects)
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_subjects))

    # Plot each subject's confidence progression
    for i, subject_id in enumerate(subjects):
        confidence = results['session_confidence'][subject_id]
        if confidence:
            # Create enhanced legend label with session count and average confidence
            session_count = results['subject_session_counts'][subject_id]
            avg_conf = results['subject_avg_confidence'][subject_id]
            label = f"{subject_id} (Sessions: {session_count}, CC: {avg_conf:.2f})"

            plt.plot(range(1, len(confidence) + 1), confidence, 'o-',
                     color=colors[i], linewidth=2, label=label)

    # Add a reference line at p=0.05 (-log10(0.05) ≈ 1.3)
    p_05_line = -np.log10(0.05)
    plt.axhline(y=p_05_line, color='red', linestyle='--', linewidth=1.5,
                label='p = 0.05')

    # Set axis labels and title
    plt.xlabel('Session Number', fontsize=14)
    plt.ylabel('Computer Confidence (-log10(p))', fontsize=14)
    plt.title('Computer Confidence Progression Across Sessions by Subject', fontsize=16)

    # Set axis limits
    plt.xlim(0.5, results['max_sessions'] + 0.5)
    plt.grid(True, alpha=0.3)

    # Add legend (outside plot if many subjects)
    if num_subjects > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(loc='upper right')

    plt.tight_layout()

    # Save the figure
    if save_fig:
        figure_path = os.path.join(output_dir, "group_analysis")
        ensure_directory_exists(figure_path)
        fig_file = os.path.join(figure_path, f"group_computer_confidence.png")

        try:
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_file}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    plt.show()


def analyze_reward_rate_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by reward rate quartiles for a single session or pooled across sessions
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
        
    Returns:
    --------
    dict: Analysis results including quartile bins and signal data
    """
    all_plotting_data = []
    all_reward_rates = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
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

        sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
        
        # Process each session separately to maintain session-specific reward rate context
        for session_date in sessions:
            # Pass behavior_df to process_session to reuse data
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate reward rates for this session
            behavior_data = session_result['behavioral_data']
            rewards = np.array(behavior_data['reward'])
            window_size = 20
            reward_rates = []
            overall_rate = np.mean(rewards)

            for i in range(len(rewards)):
                if i < window_size:
                    available_data = rewards[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
                else:
                    rate = np.mean(rewards[i - window_size + 1:i + 1])
                reward_rates.append(rate)

            # Get valid trials
            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                    if idx in session_result["non_m_trials"]])
            
            # Store data and corresponding reward rates
            all_plotting_data.append(session_result['plotting_data'])
            all_reward_rates.extend(np.array(reward_rates)[non_m_indices])
            all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
            time_axis = session_result['time_axis']
        
        plot_title = f'Pooled Photometry by Reward Rate Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        reward_rates = np.array(all_reward_rates)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        window_size = 20
        reward_rates = []
        overall_rate = np.mean(rewards)

        for i in range(len(rewards)):
            if i < window_size:
                available_data = rewards[:i + 1]
                missing_data_weight = (window_size - len(available_data)) / window_size
                rate = (np.sum(available_data) + missing_data_weight * window_size * overall_rate) / window_size
            else:
                rate = np.mean(rewards[i - window_size + 1:i + 1])
            reward_rates.append(rate)

        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                if idx in session_result["non_m_trials"]])
        
        plotting_data = session_result['plotting_data']
        reward_rates = np.array(reward_rates)[non_m_indices]
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        time_axis = session_result['time_axis']
        plot_title = f'LC Signal by Reward Rate Quartiles: {subject_id} - {session_date}'

    # Create quartile bins based on all reward rates
    quartile_bins = pd.qcut(reward_rates, q=4, labels=False)
    
    # Calculate average reward rate for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(reward_rates[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)
    
    # Print average reward rates for each quartile
    print(f"\nAverage reward rates by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                               rewarded_avg - rewarded_sem,
                               rewarded_avg + rewarded_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                        color=colors[quartile], linewidth=2,
                        label=f'Quartile {quartile + 1} Rewarded (n={np.sum(quartile_rewarded)})')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                        color=colors[quartile], linewidth=2, linestyle='--',
                        label=f'Quartile {quartile + 1} Unrewarded (n={np.sum(quartile_unrewarded)})')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            if np.sum(quartile_trials) > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                               quartile_avg - quartile_sem,
                               quartile_avg + quartile_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                        color=colors[quartile], linewidth=2,
                        label=f'Quartile {quartile + 1} (n={np.sum(quartile_trials)})')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(plot_title, fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=16)
    
    # Add text with quartile averages at the bottom of the plot
    quartile_text = "Average reward rates: " + ", ".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"reward_rate_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled", 
               f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()

def analyze_comp_confidence_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by computer confidence quartiles for a single session or pooled across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet

    Returns:
    --------
    dict: Analysis results including quartile bins and confidence values
    """
    all_plotting_data = []
    all_confidences = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
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

        # Process each session separately to maintain session-specific confidence context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Get behavioral data for this session
            behavior_data = session_result['behavioral_data']

            # Get p-value data for this session
            try:
                if behavior_df is not None:
                    # Simply filter from the already filtered dataframe (no need to check protocol again)
                    session_data = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                            (behavior_df['date'] == session_date)]
                else:
                    # Load from parquet file - this should rarely happen with your new approach
                    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                    df['date'] = df['date'].astype(str)
                    session_data = df[(df['subjid'] == subject_id) & 
                                    (df['date'] == session_date) & 
                                    (df["ignore"] == 0) & 
                                    (df['protocol'].str.contains('MatchingPennies', na=False))]

                if session_data.empty:
                    print(f"No p-value data found for {subject_id} on {session_date}")
                    continue
                # Extract p-values and calculate confidence
                p_values = session_data['min_pvalue'].values
                min_p_value = 1e-12
                p_values = np.maximum(p_values, min_p_value)
                confidence = -np.log10(p_values)

                # Calculate moving average confidence with window size 15
                window_size = 20
                confidence_rates = []
                overall_confidence = np.mean(confidence)

                for i in range(len(confidence)):
                    if i < window_size:
                        available_data = confidence[:i + 1]
                        missing_data_weight = (window_size - len(available_data)) / window_size
                        rate = (np.sum(
                            available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                    else:
                        rate = np.mean(confidence[i - window_size + 1:i + 1])
                    confidence_rates.append(rate)

                # Get valid trials
                non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                          if idx in session_result["non_m_trials"]])

                # Store data and corresponding confidence rates
                all_plotting_data.append(session_result['plotting_data'])
                all_confidences.extend(np.array(confidence_rates)[non_m_indices])
                all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
                time_axis = session_result['time_axis']

            except Exception as e:
                print(f"Error processing p-values for {subject_id}/{session_date}: {e}")
                continue

        plot_title = f'Pooled Photometry by Computer Confidence Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        confidence_rates = np.array(all_confidences)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Get p-value data for this session
        try:
            if behavior_df is not None:
                # Filter from provided dataframe
                session_data = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                          (behavior_df['date'] == session_date) & 
                                          (behavior_df["ignore"] == 0) & 
                                          (behavior_df['protocol'].str.contains('MatchingPennies', na=False))]
            else:
                # Load from parquet file
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)  # Ensure date is a string
                session_data = df[(df['subjid'] == subject_id) & 
                                 (df['date'] == session_date) & 
                                 (df["ignore"] == 0) & 
                                 (df['protocol'].str.contains('MatchingPennies', na=False))]

            if session_data.empty:
                print(f"No p-value data found for {subject_id} on {session_date}")
                return None

            # Extract p-values and calculate confidence
            p_values = session_data['min_pvalue'].values
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 15
            window_size = 20
            confidence_rates = []
            overall_confidence = np.mean(confidence)

            for i in range(len(confidence)):
                if i < window_size:
                    available_data = confidence[:i + 1]
                    missing_data_weight = (window_size - len(available_data)) / window_size
                    rate = (np.sum(
                        available_data) + missing_data_weight * window_size * overall_confidence) / window_size
                else:
                    rate = np.mean(confidence[i - window_size + 1:i + 1])
                confidence_rates.append(rate)

            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                      if idx in session_result["non_m_trials"]])

            plotting_data = session_result['plotting_data']
            confidence_rates = np.array(confidence_rates)[non_m_indices]
            reward_outcomes = session_result["reward_outcomes"][non_m_indices]
            time_axis = session_result['time_axis']
            plot_title = f'LC Signal by Computer Confidence Quartiles: {subject_id} - {session_date}'

        except Exception as e:
            print(f"Error processing p-values for {subject_id}/{session_date}: {e}")
            return None

    # Create quartile bins based on all confidence values
    quartile_bins = pd.qcut(confidence_rates, q=4, labels=False)

    # Calculate average confidence for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(confidence_rates[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)
    
    # Print average confidence for each quartile
    print(f"\nAverage computer confidence by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']  # From lowest to highest confidence

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                                 rewarded_avg - rewarded_sem,
                                 rewarded_avg + rewarded_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} Rewarded (n={np.sum(quartile_rewarded)})')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                         color=colors[quartile], linewidth=2, linestyle='--',
                         label=f'Quartile {quartile + 1} Unrewarded (n={np.sum(quartile_unrewarded)})')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            if np.sum(quartile_trials) > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                                 quartile_avg - quartile_sem,
                                 quartile_avg + quartile_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} (n={np.sum(quartile_trials)})')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    
    # Add text with quartile averages at the bottom of the plot
    quartile_text = "Average confidence values: " + ", ".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"computer_confidence_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled",
                f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()


def plot_choice_history(behavior_data, subject_id, session_date):
    # Extract choices and rewards
    choices = behavior_data['choice']
    rewards = behavior_data['reward']
    trials = np.arange(1, len(choices) + 1)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 3))
    
    # Plot the choices
    for i, choice in enumerate(choices):
        if choice == 'L':
            plt.plot([i + 1, i + 1], [0, 1], 'r-', linewidth=1.5)
            if rewards[i] == 1:
                plt.plot(i + 1, 1, 'ro', markersize=8, fillstyle='none')
        elif choice == 'R':
            plt.plot([i + 1, i + 1], [0, -1], 'b-', linewidth=1.5)
            if rewards[i] == 1:
                plt.plot(i + 1, -1, 'bo', markersize=8, fillstyle='none')
        # 'M' choices result in a gap (no line)
    
    # Add the middle line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Set the y-axis limits and labels
    plt.ylim(-1.5, 1.5)
    plt.yticks([-1, 0, 1], ['Right', '', 'Left'])
    
    # Set the x-axis and title
    plt.xlabel('Trial Number')
    plt.title(f'Choice History: {subject_id} - {session_date}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_per_session_win_loss(subject_id, behavior_df=None):
    """
    Plot win/loss traces for each session of a subject with choice history
    and state probabilities
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
        
    Returns:
    --------
    dict: Dictionary of session win-loss analyses
    """
    # Find all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    
    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
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

    # Find max peak for consistent y-axis scaling
    max_peak = float('-inf')
    min_peak = float('inf')
    session_analyses = {}
    valid_sessions = []

    # First pass: find valid sessions and determine y-axis scaling
    for session_date in sessions:
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue  # Skip this session
            
        # Add to valid sessions list since it has enough trials
        valid_sessions.append(session_date)

        # Filter out missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                if idx in session_result["non_m_trials"]])
        
        # Get reward outcomes and photometry data for valid non-missed trials
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        win_plots = session_plots[reward_outcomes == 1]
        loss_plots = session_plots[reward_outcomes == 0]
        
        # Skip this session if not enough win/loss trials
        if len(win_plots) < 5 or len(loss_plots) < 5:
            print(f"Skipping {subject_id}/{session_date}, not enough win/loss trials.")
            valid_sessions.pop()  # Remove from valid sessions
            continue
            
        # Calculate averages
        win_avg = np.mean(win_plots, axis=0)
        loss_avg = np.mean(loss_plots, axis=0)
        
        # Calculate SEMs
        win_sem = calculate_sem(win_plots, axis=0)
        loss_sem = calculate_sem(loss_plots, axis=0)
        
        # Find max/min for y-axis scaling
        max_peak = max(max_peak, np.max(win_avg + win_sem), np.max(loss_avg + loss_sem))
        min_peak = min(min_peak, np.min(win_avg - win_sem), np.min(loss_avg - loss_sem))
        
        # Store session data
        session_analyses[session_date] = {
            'win_avg': win_avg,
            'loss_avg': loss_avg,
            'win_sem': win_sem,
            'loss_sem': loss_sem,
            'win_count': len(win_plots),
            'loss_count': len(loss_plots),
            'session_result': session_result  # Keep the full result for later
        }
    
    if not valid_sessions:
        print(f"No valid sessions found for {subject_id}")
        return {}
        
    # Add some buffer to y-axis limits
    y_range = max_peak - min_peak
    y_max = max_peak + 0.05 * y_range
    y_min = min_peak - 0.05 * y_range
    
    # Calculate layout
    n_sessions = len(valid_sessions)
    n_cols = 3  # Show 5 sessions per row
    n_rows = (n_sessions + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with proper size
    fig = plt.figure(figsize=(18, 8*n_rows))  # Each row is 7 inches tall
    
    # Add a big general title above all plots
    fig.suptitle(f"Session History for {subject_id}", fontsize=24, y=0.98)
    
    # Create GridSpec to control the layout - 3 rows per session (photometry + choice + state)
    gs = plt.GridSpec(n_rows*3, n_cols, height_ratios=[2, 1, 1] * n_rows)  # Repeat [2,1,1] pattern for each row
    
    # Second pass: create all the plots
    time_axis = None
    
    for i, session_date in enumerate(valid_sessions):
        row = i // n_cols
        col = i % n_cols
        
        session_data = session_analyses[session_date]
        session_result = session_data['session_result']
        behavior_data = session_result['behavioral_data']
        
        # Set time axis from the first valid session if not set yet
        if time_axis is None:
            time_axis = session_result['time_axis']
            
        # Photometry plot (1st row of 3 for this session)
        ax1 = fig.add_subplot(gs[row*3, col])
        
        # Plot reward outcomes - using pre-calculated values from first pass
        ax1.fill_between(time_axis, 
                         session_data['win_avg'] - session_data['win_sem'],
                         session_data['win_avg'] + session_data['win_sem'],
                         color='lightgreen', alpha=0.3)
                         
        ax1.plot(time_axis, session_data['win_avg'], color='green', linewidth=2, 
                 label=f'Win (n={session_data["win_count"]})')
                 
        ax1.fill_between(time_axis, 
                         session_data['loss_avg'] - session_data['loss_sem'],
                         session_data['loss_avg'] + session_data['loss_sem'],
                         color='lightsalmon', alpha=0.3)
                         
        ax1.plot(time_axis, session_data['loss_avg'], color='darkorange', linewidth=2,
                 label=f'Loss (n={session_data["loss_count"]})')
        
        # Add vertical line at cue onset
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Lick Timing')
        
        # Use consistent y-axis scaling across all sessions
        ax1.set_ylim(y_min, y_max)
        
        # Add horizontal line at y=0
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Only add x labels to bottom row
        ax1.set_xlabel('Time (s)')
        
        # Only add y labels to leftmost column
        if col == 0:
            ax1.set_ylabel('ΔF/F')
            
        ax1.set_title(f"{session_date}", fontsize=12)
        
        # Add legend to first plot only
        if i == 0:
            ax1.legend(loc='upper right')
        
        # Choice history plot (2nd row of 3 for this session)
        ax2 = fig.add_subplot(gs[row*3 + 1, col])

        # Extract choices and rewards
        choices = np.array(behavior_data['choice'])
        rewards = np.array(behavior_data['reward'])

        # Plot choice history
        for j, choice in enumerate(choices):
            if choice == 'L':
                ax2.plot([j + 1, j + 1], [0, 1], 'r-', linewidth=1.5)
                if rewards[j] == 1:
                    ax2.plot(j + 1, 1, 'ro', markersize=8, fillstyle='none')
            elif choice == 'R':
                ax2.plot([j + 1, j + 1], [0, -1], 'b-', linewidth=1.5)
                if rewards[j] == 1:
                    ax2.plot(j + 1, -1, 'bo', markersize=8, fillstyle='none')
            # 'M' choices result in a gap (no line)

        # Add the middle line
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)

        # Set the y-axis limits and labels
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Right', '', 'Left'])

        # Set the x-axis and title
        ax2.set_xlabel('Trial Number')
        ax2.set_title('Choice History')
        ax2.grid(True, alpha=0.3)


        # State probabilities plot (3rd row of 3 for this session)
        ax3 = fig.add_subplot(gs[row*3 + 2, col])
        
        # Get state probability data for this session
        if behavior_df is not None:
            session_df = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                    (behavior_df['date'] == session_date)]
        else:
            # Load from parquet file - this should rarely happen with your new approach
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            session_df = df[(df['subjid'] == subject_id) & 
                            (df['date'] == session_date) & 
                            (df["ignore"] == 0) & 
                            (df['protocol'].str.contains('MatchingPennies', na=False))]
        
        if not session_df.empty and 'p_stochastic' in session_df.columns:
            # Plot state probabilities if available
            x_values = np.arange(len(session_df))
            
            # Plot each state probability
            ax3.plot(x_values, session_df['p_stochastic'], color='green', linewidth=1.5, label='Stochastic')
            ax3.plot(x_values, session_df['p_leftbias'], color='red', linewidth=1.5, label='Left Bias')
            ax3.plot(x_values, session_df['p_rightbias'], color='blue', linewidth=1.5, label='Right Bias')
            
            # Add threshold line
            ax3.axhline(y=0.8, color='black', linestyle='--', linewidth=0.5, alpha=0.7, label='Threshold')
            
            # Add text marker for highest probability at final trial
            final_idx = len(session_df) - 1
            if final_idx >= 0:
                probs = {
                    'Stochastic': session_df.iloc[final_idx]['p_stochastic'],
                    'Left Bias': session_df.iloc[final_idx]['p_leftbias'],
                    'Right Bias': session_df.iloc[final_idx]['p_rightbias']
                }
                max_state = max(probs, key=probs.get)
                max_prob = probs[max_state]
                
                if max_prob >= 0.8:  # Only add marker if probability is high enough
                    ax3.text(final_idx, max_prob, max_state[0], 
                            fontsize=8, ha='center', va='bottom')
                    
            # Set y-axis limits
            ax3.set_ylim(-0.05, 1.05)
            ax3.set_ylabel('State Prob.')
            
            # Only show legend for first plot to avoid clutter
            if i == 0:
                ax3.legend(loc='upper right', fontsize=7)
                
            # Set x-axis limits
            ax3.set_xlim(0, len(session_df))
        else:
            # No state data available
            ax3.text(0.5, 0.5, "No state probability data", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_ylabel('State Prob.')
            ax3.set_ylim(0, 1)
        
        ax3.set_xlabel('Trial Number')
    
    # Adjust layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
    
    # Save the figure
    save_figure(fig, subject_id, "all_sessions", "per_session_win_loss_with_choices_and_states")
    
    plt.show()
    

def analyze_session_win_loss_difference_gap(subject_id, session_date=None, comp_conf=False, behavior_df=None, sem=True):
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
    df : pandas.DataFrame, optional
        Pre-loaded dataframe to use for confidence calculations

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
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Store results for each session
    session_differences = {}
    session_confidences = {}

    # Process each session
    for idx, session_date in enumerate(sessions):
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
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

        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session from provided or loaded dataframe
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if session_df.empty:
                    print(f"No behavioral data found for {subject_id}/{session_date}")
                    continue
                
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

    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
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
                print(f"{i+1}. Session {sess}: {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_sessions = sorted(session_differences.keys())

    # Create colors for plotting
    num_sessions = len(sorted_sessions)
    if comp_conf and session_confidences:
        # When sorting by confidence, use reversed color gradient (highest confidence = lightest blue)
        blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))
    else:
        # Default: earliest session = lightest blue
        blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

    # Create plot
    plt.figure(figsize=(12, 7))

    # Plot each session's win-loss difference
    for idx, sess_date in enumerate(sorted_sessions):
        time_axis = session_differences[sess_date]['time_axis']
        win_loss_diff = session_differences[sess_date]['diff']
        win_loss_sem = session_differences[sess_date]['sem']
        
        # Create label with confidence value if available
        if comp_conf and sess_date in session_confidences:
            label = f'Session {sess_date} (conf: {session_confidences[sess_date]:.2f})'
        else:
            label = f'Session {sess_date}'
        
        # Plot with shaded error region
        if sem:
            plt.fill_between(time_axis,
                            win_loss_diff - win_loss_sem,
                            win_loss_diff + win_loss_sem,
                            color=blue_colors[idx], alpha=0.2)
        plt.plot(time_axis, win_loss_diff,
                 color=blue_colors[idx],
                 label=label, linewidth=2)

    # Add reference lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Rewarded - Unrewarded ΔF/F', fontsize=16)
    
    sort_type = "Computer Confidence" if comp_conf else "Chronological"
    plt.title(f'Win-Loss Difference: {subject_id} (sorted by {sort_type})', fontsize=14)
    
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save the figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    sem_suffix = "_sem" if sem else ""
    save_figure(plt.gcf(), subject_id, "win_loss_diff", f"win_loss_difference_{sort_suffix}{sem_suffix}")

    plt.show()

    # Return results
    return {
        'session_differences': session_differences,
        'session_confidences': session_confidences if comp_conf else None,
        'sorted_sessions': sorted_sessions
    }


def analyze_previous_outcome_effect(subject_id, time_split=False, behavior_df=None):
    """
    Analyze photometry signals based on previous and current trial outcomes.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    time_split : bool, optional (default=False)
        If True, additionally split data by early/middle/late sessions and
        create separate plots showing temporal evolution
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet

    Returns:
    --------
    dict: Analysis results
    """

    subject_path = os.path.join(base_dir, subject_id)

    # Find all session directories for this subject
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, filter it for this subject
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(
                f"Found {len(matching_pennies_sessions)} MatchingPennies sessions for {subject_id} in provided dataframe")
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

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    all_prev_rewards = []
    all_curr_rewards = []
    session_dates = []

    # For time split analysis, we'll store data separately for each time period
    if time_split:
        # First pass: identify valid sessions with enough trials
        valid_sessions = []
        
        # Check each session for having enough trials
        for session_date in sessions:
            session_path = os.path.join(subject_path, session_date)
            if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
                # Process session to check trial count
                result = process_session(subject_id, session_date, behavior_df=behavior_df)
                if result and len(result['non_m_trials']) >= 100:
                    valid_sessions.append(session_date)
                else:
                    print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")

        # Now split the valid sessions evenly
        num_valid = len(valid_sessions)
        if num_valid < 3:
            print(f"Warning: Only {num_valid} valid sessions available. Need at least 3 for time split analysis.")
            time_split = False  # Fall back to non-split analysis
        else:
            # Calculate split points for even division
            early_end = num_valid // 3
            middle_end = 2 * (num_valid // 3)
            
            # Adjust for remainder (ensure most even split possible)
            if num_valid % 3 == 1:
                # Add the extra session to the last group
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end]
                late_sessions = valid_sessions[middle_end:]
            elif num_valid % 3 == 2:
                # Add one extra session to middle and late groups
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end+1]
                late_sessions = valid_sessions[middle_end+1:]
            else:
                # Perfect division
                early_sessions = valid_sessions[:early_end]
                middle_sessions = valid_sessions[early_end:middle_end]
                late_sessions = valid_sessions[middle_end:]

            # Initialize data containers for each time period
            time_periods = {
                'early': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []},
                'middle': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []},
                'late': {'sessions': [], 'plotting_data': [], 'prev_rewards': [], 'curr_rewards': []}
            }
            
            print(f"Time split analysis: Early ({len(early_sessions)} sessions), " +
                  f"Middle ({len(middle_sessions)} sessions), Late ({len(late_sessions)} sessions)")

    # Process sessions in chronological order
    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_path, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            # Pass behavior_df to process_session to reuse data
            result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if result:
                if len(result['non_m_trials']) < 100:
                    print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                    continue

                all_sessions.append(result)
                session_dates.append(session_date)

                # Get behavioral data
                behavior_data = result['behavioral_data']
                rewards = behavior_data['reward']

                # Filter out missed trials
                non_m_indices = np.array([i for i, idx in enumerate(result["valid_trials"])
                                          if idx in result["non_m_trials"]])

                # Extract data for valid trials
                session_plots = result['plotting_data']
                curr_rewards = result["reward_outcomes"][non_m_indices]

                # Create previous reward array (shifted)
                # First trial has no previous trial, assign -1 (will be filtered out)
                prev_rewards = np.zeros_like(curr_rewards)
                prev_rewards[0] = -1  # No previous trial for first trial
                prev_rewards[1:] = curr_rewards[:-1]

                # Store data for overall analysis
                all_plotting_data.append(session_plots)
                all_curr_rewards.append(curr_rewards)
                all_prev_rewards.append(prev_rewards)

                # If doing time split analysis, store in the appropriate time period
                if time_split:
                    # Determine which period this session belongs to
                    if session_date in early_sessions:
                        period = 'early'
                    elif session_date in middle_sessions:
                        period = 'middle'
                    elif session_date in late_sessions:
                        period = 'late'
                    else:
                        # Skip sessions that were excluded during validation
                        continue
                        
                    time_periods[period]['sessions'].append(result)
                    time_periods[period]['plotting_data'].append(session_plots)
                    time_periods[period]['curr_rewards'].append(curr_rewards)
                    time_periods[period]['prev_rewards'].append(prev_rewards)
    

    if not all_sessions:
        print(f"No processed sessions found for subject {subject_id}")
        return None

    # Combine all trials from all sessions
    plotting_data = np.vstack(all_plotting_data)
    curr_rewards = np.concatenate(all_curr_rewards)
    prev_rewards = np.concatenate(all_prev_rewards)

    # Filter out first trials (no previous outcome)
    valid_trials = prev_rewards != -1
    plotting_data = plotting_data[valid_trials]
    curr_rewards = curr_rewards[valid_trials]
    prev_rewards = prev_rewards[valid_trials]

    # Get time axis
    time_axis = all_sessions[0]['time_axis']

    # Create condition masks
    prev_win_curr_win = (prev_rewards == 1) & (curr_rewards == 1)
    prev_win_curr_loss = (prev_rewards == 1) & (curr_rewards == 0)
    prev_loss_curr_win = (prev_rewards == 0) & (curr_rewards == 1)
    prev_loss_curr_loss = (prev_rewards == 0) & (curr_rewards == 0)

    # Calculate averages and SEM for each condition
    condition_data = {
        'prev_win_curr_win': {
            'data': plotting_data[prev_win_curr_win],
            'avg': np.mean(plotting_data[prev_win_curr_win], axis=0) if np.any(prev_win_curr_win) else None,
            'sem': calculate_sem(plotting_data[prev_win_curr_win], axis=0) if np.any(prev_win_curr_win) else None,
            'count': np.sum(prev_win_curr_win)
        },
        'prev_win_curr_loss': {
            'data': plotting_data[prev_win_curr_loss],
            'avg': np.mean(plotting_data[prev_win_curr_loss], axis=0) if np.any(prev_win_curr_loss) else None,
            'sem': calculate_sem(plotting_data[prev_win_curr_loss], axis=0) if np.any(prev_win_curr_loss) else None,
            'count': np.sum(prev_win_curr_loss)
        },
        'prev_loss_curr_win': {
            'data': plotting_data[prev_loss_curr_win],
            'avg': np.mean(plotting_data[prev_loss_curr_win], axis=0) if np.any(prev_loss_curr_win) else None,
            'sem': calculate_sem(plotting_data[prev_loss_curr_win], axis=0) if np.any(prev_loss_curr_win) else None,
            'count': np.sum(prev_loss_curr_win)
        },
        'prev_loss_curr_loss': {
            'data': plotting_data[prev_loss_curr_loss],
            'avg': np.mean(plotting_data[prev_loss_curr_loss], axis=0) if np.any(prev_loss_curr_loss) else None,
            'sem': calculate_sem(plotting_data[prev_loss_curr_loss], axis=0) if np.any(prev_loss_curr_loss) else None,
            'count': np.sum(prev_loss_curr_loss)
        }
    }

    # If doing time split analysis, compute condition data for each time period
    time_period_data = {}
    if time_split:
        for period in ['early', 'middle', 'late']:
            if not time_periods[period]['sessions']:
                print(f"No sessions in {period} period. Skipping.")
                continue

            # Combine data for this time period
            period_plotting_data = np.vstack(time_periods[period]['plotting_data'])
            period_curr_rewards = np.concatenate(time_periods[period]['curr_rewards'])
            period_prev_rewards = np.concatenate(time_periods[period]['prev_rewards'])

            # Filter out first trials
            period_valid = period_prev_rewards != -1
            period_plotting_data = period_plotting_data[period_valid]
            period_curr_rewards = period_curr_rewards[period_valid]
            period_prev_rewards = period_prev_rewards[period_valid]

            # Create condition masks
            period_prev_win_curr_win = (period_prev_rewards == 1) & (period_curr_rewards == 1)
            period_prev_win_curr_loss = (period_prev_rewards == 1) & (period_curr_rewards == 0)
            period_prev_loss_curr_win = (period_prev_rewards == 0) & (period_curr_rewards == 1)
            period_prev_loss_curr_loss = (period_prev_rewards == 0) & (period_curr_rewards == 0)

            # Calculate averages and SEM for each condition
            time_period_data[period] = {
                'prev_win_curr_win': {
                    'data': period_plotting_data[period_prev_win_curr_win],
                    'avg': np.mean(period_plotting_data[period_prev_win_curr_win], axis=0)
                    if np.any(period_prev_win_curr_win) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_win_curr_win], axis=0)
                    if np.any(period_prev_win_curr_win) else None,
                    'count': np.sum(period_prev_win_curr_win)
                },
                'prev_win_curr_loss': {
                    'data': period_plotting_data[period_prev_win_curr_loss],
                    'avg': np.mean(period_plotting_data[period_prev_win_curr_loss], axis=0)
                    if np.any(period_prev_win_curr_loss) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_win_curr_loss], axis=0)
                    if np.any(period_prev_win_curr_loss) else None,
                    'count': np.sum(period_prev_win_curr_loss)
                },
                'prev_loss_curr_win': {
                    'data': period_plotting_data[period_prev_loss_curr_win],
                    'avg': np.mean(period_plotting_data[period_prev_loss_curr_win], axis=0)
                    if np.any(period_prev_loss_curr_win) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_loss_curr_win], axis=0)
                    if np.any(period_prev_loss_curr_win) else None,
                    'count': np.sum(period_prev_loss_curr_win)
                },
                'prev_loss_curr_loss': {
                    'data': period_plotting_data[period_prev_loss_curr_loss],
                    'avg': np.mean(period_plotting_data[period_prev_loss_curr_loss], axis=0)
                    if np.any(period_prev_loss_curr_loss) else None,
                    'sem': calculate_sem(period_plotting_data[period_prev_loss_curr_loss], axis=0)
                    if np.any(period_prev_loss_curr_loss) else None,
                    'count': np.sum(period_prev_loss_curr_loss)
                }
            }

    # Create the regular (non-time-split) plot
    plt.figure(figsize=(12, 7))

    # Define colors and labels
    colors = {
        'prev_win_curr_win': 'darkgreen',
        'prev_win_curr_loss': 'firebrick',
        'prev_loss_curr_win': 'mediumseagreen',
        'prev_loss_curr_loss': 'indianred'
    }

    labels = {
        'prev_win_curr_win': f"Win-Win (n={condition_data['prev_win_curr_win']['count']})",
        'prev_win_curr_loss': f"Win-Loss (n={condition_data['prev_win_curr_loss']['count']})",
        'prev_loss_curr_win': f"Loss-Win (n={condition_data['prev_loss_curr_win']['count']})",
        'prev_loss_curr_loss': f"Loss-Loss (n={condition_data['prev_loss_curr_loss']['count']})"
    }

    # Plot each condition
    for condition, color in colors.items():
        if condition_data[condition]['avg'] is not None:
            plt.fill_between(time_axis,
                             condition_data[condition]['avg'] - condition_data[condition]['sem'],
                             condition_data[condition]['avg'] + condition_data[condition]['sem'],
                             color=color, alpha=0.3)
            plt.plot(time_axis, condition_data[condition]['avg'],
                     color=color, linewidth=2, label=labels[condition])

    # Add vertical line at cue onset
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)
    plt.title(f'LC Signal by Previous Trial Outcome: {subject_id} ({len(all_sessions)} sessions)', fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()

    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", "previous_outcome_effect")
    plt.show()

    # If time_split is enabled, create additional plots
    if time_split:
        # Define style parameters for time-split plots
        period_colors = {'early': 'lightskyblue', 'middle': 'royalblue', 'late': 'darkblue'}
        condition_styles = {
            'prev_win_curr_win': {'color': 'darkgreen', 'linestyle': '-', 'marker': 'o'},
            'prev_win_curr_loss': {'color': 'firebrick', 'linestyle': '-', 'marker': 's'},
            'prev_loss_curr_win': {'color': 'mediumseagreen', 'linestyle': '-', 'marker': '^'},
            'prev_loss_curr_loss': {'color': 'indianred', 'linestyle': '-', 'marker': 'D'}
        }

        # 1. Create plots for each time period (early, middle, late) - all conditions
        for period in ['early', 'middle', 'late']:
            if period not in time_period_data:
                continue

            plt.figure(figsize=(12, 7))

            # Plot each condition for this time period
            for condition, style in condition_styles.items():
                if time_period_data[period][condition]['avg'] is not None:
                    count = time_period_data[period][condition]['count']
                    plt.fill_between(time_axis,
                                     time_period_data[period][condition]['avg'] - time_period_data[period][condition][
                                         'sem'],
                                     time_period_data[period][condition]['avg'] + time_period_data[period][condition][
                                         'sem'],
                                     color=style['color'], alpha=0.3)
                    plt.plot(time_axis, time_period_data[period][condition]['avg'],
                            color=style['color'], linewidth=2,
                            label=f"{condition.replace('prev_', '').replace('curr_', '-').replace('_', '').title()} (n={count})")

            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)

            # Count sessions in this period
            num_sessions = len(time_periods[period]['sessions'])
            plt.title(
                f'{period.capitalize()} Sessions - Previous Outcome Effect: {subject_id} ({num_sessions} sessions)',
                fontsize=20)

            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=16)
            plt.tight_layout()

            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"previous_outcome_effect_{period}")
            plt.show()

        # 2. Create plots for each condition (across time periods)
        for condition, style in condition_styles.items():
            plt.figure(figsize=(12, 7))

            # Plot this condition for each time period
            for period, color in period_colors.items():
                if period in time_period_data and time_period_data[period][condition]['avg'] is not None:
                    count = time_period_data[period][condition]['count']
                    plt.fill_between(time_axis,
                                     time_period_data[period][condition]['avg'] - time_period_data[period][condition][
                                         'sem'],
                                     time_period_data[period][condition]['avg'] + time_period_data[period][condition][
                                         'sem'],
                                     color=color, alpha=0.3)
                    plt.plot(time_axis, time_period_data[period][condition]['avg'],
                             color=color, linewidth=2,
                             label=f"{period.capitalize()} Sessions (n={count})")

            # Add vertical line at cue onset
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Labels and formatting
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('ΔF/F', fontsize=16)

            # Format condition name for title
            condition_title = condition.replace('prev_', '').replace('curr_', '-').replace('_', '').title()
            plt.title(f'Temporal Evolution of {condition_title}: {subject_id}', fontsize=20)

            plt.xlim([-pre_cue_time, post_cue_time])
            plt.legend(loc='upper right', fontsize=16)
            plt.tight_layout()

            # Save the figure
            save_figure(plt.gcf(), subject_id, "pooled", f"previous_outcome_effect_evolution_{condition}")
            plt.show()

    # Return analysis results
    result = {
        'subject_id': subject_id,
        'condition_data': condition_data,
        'time_axis': time_axis,
        'num_sessions': len(all_sessions),
        'session_dates': session_dates
    }


def analyze_win_stay_lose_switch(subject_id, session_date=None, behavior_df=None):
    """
    Calculate Win-Stay, Lose-Switch statistics for a subject using all behavioral trials

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.

    Returns:
    --------
    dict: Analysis results including WSLS counts and percentages
    """

    if behavior_df is not None:
        df = behavior_df
        if 'date' in df.columns and df['date'].dtype != str:
            df['date'] = df['date'].astype(str)
    else:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
        except Exception as e:
            print(f"Error loading parquet data: {e}")
        df = None  # We'll proceed without state analysis if file can't be loaded

    # Store results
    wsls_results = {
        'win_stay_count': 0,
        'lose_switch_count': 0,
        'total_valid_pairs': 0,
        'total_win_trials': 0,
        'total_lose_trials': 0,
        'sessions_analyzed': 0
    }
    
    # Store per-session results
    session_data = []

    # Get sessions to analyze
    if session_date is None:
        # Get all sessions from provided dataframe for this subject
        sessions = sorted(df[df['subjid'] == subject_id]['date'].unique())
    else:
        sessions = [session_date]


    # Process each session
    for sess in sessions:
        print(f"Analyzing WSLS for {subject_id}/{sess}...")
        
        # Get behavior data directly from parquet file
        try:
            session_df = df[(df['subjid'] == subject_id) & (df['date'] == sess)]
            
            if session_df.empty:
                print(f"No behavioral data found for {subject_id}/{sess}")
                continue
                
            choices = session_df['choice'].tolist()
            rewards = session_df['reward'].tolist()
        except Exception as e:
            print(f"Error loading behavioral data for {subject_id}/{sess}: {e}")
            continue

        # Filter out missed ('M') trials
        valid_indices = [i for i, choice in enumerate(choices) if choice in ['L', 'R']]
        valid_choices = [choices[i] for i in valid_indices]
        valid_rewards = [rewards[i] for i in valid_indices]

        # Skip session if not enough valid trials
        if len(valid_choices) < 2:
            print(f"Session {sess} has fewer than 2 valid trials. Skipping...")
            continue

        # Count WSLS occurrences
        win_stay_count = 0
        lose_switch_count = 0
        win_trials = 0
        lose_trials = 0

        # We start from the second trial since we need to know the previous trial's outcome
        for i in range(1, len(valid_choices)):
            prev_choice = valid_choices[i - 1]
            curr_choice = valid_choices[i]
            prev_reward = valid_rewards[i - 1]

            # Count win and lose trials from previous trial
            if prev_reward == 1:
                win_trials += 1
                # Win-Stay: previous trial was rewarded and animal made same choice
                if prev_choice == curr_choice:
                    win_stay_count += 1
            else:
                lose_trials += 1
                # Lose-Switch: previous trial was not rewarded and animal switched choice
                if prev_choice != curr_choice:
                    lose_switch_count += 1

        # Add to total counts
        wsls_results['win_stay_count'] += win_stay_count
        wsls_results['lose_switch_count'] += lose_switch_count
        wsls_results['total_valid_pairs'] += (len(valid_choices) - 1)  # pairs of trials
        wsls_results['total_win_trials'] += win_trials
        wsls_results['total_lose_trials'] += lose_trials
        wsls_results['sessions_analyzed'] += 1

        # Calculate session-specific percentages
        win_stay_pct = (win_stay_count / win_trials * 100) if win_trials > 0 else 0
        lose_switch_pct = (lose_switch_count / lose_trials * 100) if lose_trials > 0 else 0
        wsls_pct = ((win_stay_count + lose_switch_count) / (len(valid_choices) - 1) * 100) if len(valid_choices) > 1 else 0
        
        # Store session data
        session_data.append({
            'session_date': sess,
            'win_stay_count': win_stay_count,
            'win_trials': win_trials,
            'win_stay_pct': win_stay_pct,
            'lose_switch_count': lose_switch_count,
            'lose_trials': lose_trials,
            'lose_switch_pct': lose_switch_pct,
            'total_wsls_count': win_stay_count + lose_switch_count,
            'total_trials': len(valid_choices) - 1,
            'wsls_pct': wsls_pct
        })

        # Print session-specific results
        print(f"  Session {sess} WSLS stats:")
        print(f"    Win-Stay: {win_stay_count}/{win_trials} trials ({win_stay_pct:.1f}%)")
        print(f"    Lose-Switch: {lose_switch_count}/{lose_trials} trials ({lose_switch_pct:.1f}%)")
        print(f"    Total WSLS: {win_stay_count + lose_switch_count}/{len(valid_choices) - 1} trials ({wsls_pct:.1f}%)")
        print()

    # Calculate overall percentages
    if wsls_results['total_win_trials'] > 0:
        wsls_results['win_stay_percentage'] = (wsls_results['win_stay_count'] / wsls_results['total_win_trials']) * 100
    else:
        wsls_results['win_stay_percentage'] = 0

    if wsls_results['total_lose_trials'] > 0:
        wsls_results['lose_switch_percentage'] = (wsls_results['lose_switch_count'] / wsls_results['total_lose_trials']) * 100
    else:
        wsls_results['lose_switch_percentage'] = 0

    if wsls_results['total_valid_pairs'] > 0:
        wsls_results['total_wsls_percentage'] = ((wsls_results['win_stay_count'] + wsls_results['lose_switch_count']) /
                                                 wsls_results['total_valid_pairs']) * 100
    else:
        wsls_results['total_wsls_percentage'] = 0

    # Print overall results
    print("\n=== Win-Stay, Lose-Switch Analysis ===")
    print(f"Subject: {subject_id}")
    print(f"Sessions analyzed: {wsls_results['sessions_analyzed']}")
    print(f"Total valid trial pairs: {wsls_results['total_valid_pairs']}")
    print(f"Win-Stay: {wsls_results['win_stay_count']}/{wsls_results['total_win_trials']} trials ({wsls_results['win_stay_percentage']:.1f}%)")
    print(f"Lose-Switch: {wsls_results['lose_switch_count']}/{wsls_results['total_lose_trials']} trials ({wsls_results['lose_switch_percentage']:.1f}%)")
    print(f"Overall WSLS: {wsls_results['win_stay_count'] + wsls_results['lose_switch_count']}/{wsls_results['total_valid_pairs']} trials ({wsls_results['total_wsls_percentage']:.1f}%)")

    # Visualization based on whether we're analyzing one or multiple sessions
    if session_date is None and len(session_data) > 1:
        # Multiple sessions: show WSLS, win-stay, and lose-switch percentages across sessions
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        session_numbers = list(range(1, len(session_data) + 1))
        win_stay_pcts = [s['win_stay_pct'] for s in session_data]
        lose_switch_pcts = [s['lose_switch_pct'] for s in session_data]
        wsls_pcts = [s['wsls_pct'] for s in session_data]
        
        # Plot win-stay, lose-switch, and overall WSLS percentages
        plt.plot(session_numbers, win_stay_pcts, 'o-', color='green', label='Win-Stay %')
        plt.plot(session_numbers, lose_switch_pcts, 'o-', color='orange', label='Lose-Switch %')
        plt.plot(session_numbers, wsls_pcts, 'o-', color='blue', linewidth=2, label='Overall WSLS %')
        
        # Add reference line at 50%
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        
        # Formatting
        plt.xlabel('Session Number')
        plt.ylabel('Percentage (%)')
        plt.title(f'WSLS Analysis Across Sessions: {subject_id}')
        plt.xticks(session_numbers)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save figure
        save_figure(plt.gcf(), subject_id, "all_sessions", "wsls_across_sessions")
        plt.show()
    
    # Store session data in results
    wsls_results['session_data'] = session_data
    return wsls_results


def analyze_loss_streaks_before_win(subject_id, skipped_missed=True, only_1_5=False, behavior_df=None):
    """
    Analyze photometry signals for loss streaks of different lengths that end with a win.
    This function identifies trials that were not rewarded but where the next trial was rewarded,
    and categorizes them based on the number of consecutive losses before that trial.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    skipped_missed : bool, optional (default=True)
        If True, filter out missed trials ('M') from streak calculation
        If False, include missed trials as losses as long as reward=0
    only_1_5 : bool, optional (default=False)
        If True, only plot categories 1 and 5+ (shortest and longest streaks)
    behavior_df : pandas.DataFrame, optional
        Pre-loaded behavior dataframe to use instead of loading from parquet
        
    Returns:
    --------
    dict: Analysis results for different loss streak lengths
    """
    # Find all session directories for this subject
    subject_path = os.path.join(base_dir, subject_id)
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

    # Store data for each loss streak category
    streak_data = {
        '1_loss': [],  # T0 loss, T-1 no loss
        '2_loss': [],  # T0 & T-1 loss, T-2 no loss
        '3_loss': [],  # T0, T-1, T-2 loss, T-3 no loss
        '4_loss': [],  # T0, T-1, T-2, T-3 loss, T-4 no loss
        '5plus_loss': []  # T0 loss preceded by 4+ losses
    }

    time_axis = None  # Will be set from the first valid session

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Get the full trial data - including photometry for all valid trials
        # (both missed and non-missed that have good photometry recordings)
        all_valid_trials = session_result['valid_trials']
        all_valid_epoched_data = session_result['epoched_data'][all_valid_trials]

        # Filter behavioral data based on skipped_missed parameter
        if skipped_missed:
            # Create a mask where choice is not 'M'
            valid_mask = choices != 'M'
            # Apply filter to raw behavior data
            filtered_rewards = rewards[valid_mask]
            filtered_indices = np.where(valid_mask)[0]
            # Create a mapping from filtered indices to original indices
            filtered_to_orig = {i: filtered_indices[i] for i in range(len(filtered_indices))}
            # Create reverse mapping from original to filtered indices
            orig_to_filtered = {filtered_indices[i]: i for i in range(len(filtered_indices))}
        else:
            # Use all trials without filtering
            filtered_rewards = rewards
            filtered_to_orig = {i: i for i in range(len(rewards))}
            orig_to_filtered = {i: i for i in range(len(rewards))}

        # Skip if session has too few trials
        if len(filtered_rewards) < 6:  # Need at least 6 trials to determine 5+ loss streak
            print(f"Skipping {subject_id}/{session_date}, insufficient trials after filtering")
            continue

        # Find trials that were losses followed by a win in the filtered behavioral data
        for i in range(len(filtered_rewards) - 1):
            # Check if current trial is a loss and next is a win
            if filtered_rewards[i] == 0 and filtered_rewards[i + 1] == 1:
                # This is a loss trial followed by a win
                orig_trial_idx = filtered_to_orig[i]

                # Skip if we don't have photometry data for this trial
                if orig_trial_idx not in all_valid_trials:
                    continue

                # Get the photometry data for this trial
                # Find the index in the valid_trials array
                valid_trial_idx = np.where(np.array(all_valid_trials) == orig_trial_idx)[0]
                if len(valid_trial_idx) == 0:
                    # No photometry data for this trial
                    continue

                photometry_data = all_valid_epoched_data[valid_trial_idx[0]]

                # Now count consecutive losses going backward from current trial
                loss_streak = 1  # Start with 1 (current trial is a loss)

                if skipped_missed:
                    # Looking back in filtered space (no missed trials)
                    for j in range(i - 1, -1, -1):
                        if filtered_rewards[j] == 0:
                            loss_streak += 1
                        else:
                            # Found a win, streak ends
                            break
                else:
                    # Looking back in original space (can include missed trials)
                    for j in range(orig_trial_idx - 1, -1, -1):
                        if j < len(rewards) and rewards[j] == 0:
                            loss_streak += 1
                        else:
                            # Found a win, streak ends
                            break

                # Categorize the trial based on streak length
                if loss_streak == 1:
                    streak_data['1_loss'].append(photometry_data)
                elif loss_streak == 2:
                    streak_data['2_loss'].append(photometry_data)
                elif loss_streak == 3:
                    streak_data['3_loss'].append(photometry_data)
                elif loss_streak == 4:
                    streak_data['4_loss'].append(photometry_data)
                else:  # 5 or more consecutive losses
                    streak_data['5plus_loss'].append(photometry_data)

    # Check if we found any valid streaks
    total_trials = sum(len(data) for data in streak_data.values())
    if total_trials == 0:
        print(f"No valid loss streaks found for {subject_id}")
        return None

    # Convert lists to numpy arrays for each category (if not empty)
    for category in streak_data:
        if streak_data[category]:
            streak_data[category] = np.array(streak_data[category])

    # Calculate averages and SEM for each streak length
    streak_averages = {}
    streak_sems = {}
    for category, data in streak_data.items():
        if len(data) > 0:
            streak_averages[category] = np.mean(data, axis=0)
            streak_sems[category] = calculate_sem(data, axis=0)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Define colors and labels with trial counts
    colors = {
        '1_loss': 'blue',
        '2_loss': 'green',
        '3_loss': 'orange',
        '4_loss': 'red',
        '5plus_loss': 'purple'
    }

    labels = {
        '1_loss': f"1 Loss (n={len(streak_data['1_loss'])})",
        '2_loss': f"2 Consecutive Losses (n={len(streak_data['2_loss'])})",
        '3_loss': f"3 Consecutive Losses (n={len(streak_data['3_loss'])})",
        '4_loss': f"4 Consecutive Losses (n={len(streak_data['4_loss'])})",
        '5plus_loss': f"5+ Consecutive Losses (n={len(streak_data['5plus_loss'])})"
    }

    # Determine which categories to plot based on only_1_5 parameter
    categories_to_plot = ['1_loss', '5plus_loss'] if only_1_5 else ['1_loss', '2_loss', '3_loss', '4_loss', '5plus_loss']

    # Plot selected streak categories
    for category in categories_to_plot:
        if category in streak_averages and len(streak_data[category]) > 0:
            plt.fill_between(time_axis,
                             streak_averages[category] - streak_sems[category],
                             streak_averages[category] + streak_sems[category],
                             color=colors[category], alpha=0.3)
            plt.plot(time_axis, streak_averages[category],
                     color=colors[category], linewidth=2, label=labels[category])

    # Add vertical line at cue onset
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Labels and formatting
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('ΔF/F', fontsize=16)

    missed_text = "excluding" if skipped_missed else "including"
    plot_cat_text = "1_and_5" if only_1_5 else "all_cats"
    plt.title(f'LC Signal for Cumulative Loss: {subject_id}',
              fontsize=20)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Add total trials information
    displayed_trials = sum(len(streak_data[cat]) for cat in categories_to_plot)
    plt.figtext(0.02, 0.02, f"Total trials analyzed: {displayed_trials} (of {total_trials})", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", f"loss_streaks_before_win_{missed_text}_missed_{plot_cat_text}")

    plt.show()

    # Return analysis results
    return {
        'subject_id': subject_id,
        'streak_data': streak_data,
        'streak_averages': streak_averages,
        'streak_sems': streak_sems,
        'time_axis': time_axis,
        'total_trials': total_trials,
        'displayed_trials': displayed_trials,
        'skipped_missed': skipped_missed,
        'only_1_5': only_1_5
    }


def analyze_session_win_loss_difference_heatmap(subject_id, comp_conf=False, behavior_df=None):
    """
    Create a heatmap visualization of win-loss signal differences across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    comp_conf : bool, optional (default=False)
        If True, sorts sessions by computer confidence rather than chronologically

    Returns:
    --------
    dict: Analysis results including win-loss differences across all sessions
    """
    # Get all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    matching_pennies_sessions = set()
    try:
        if behavior_df is not None:
            # If behavior_df is provided, we assume it's already filtered for MatchingPennies protocol
            subject_data = behavior_df[behavior_df['subjid'] == subject_id]
            matching_pennies_sessions = set(subject_data['date'].unique())
            print(f"Found {len(matching_pennies_sessions)} sessions for {subject_id} in provided dataframe")
        else:
            # Otherwise load from parquet file with protocol filtering
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

    # Store results for each session
    session_differences = []
    session_dates = []
    time_axis = None
    peak_differences = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(
                f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
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

        if len(rewarded_trials) == 0 or len(unrewarded_trials) == 0:
            print(f"Skipping {subject_id}/{session_date}, missing reward outcomes.")
            continue

        # Compute average rewarded and unrewarded signals
        rewarded_avg = np.mean(rewarded_trials, axis=0)
        unrewarded_avg = np.mean(unrewarded_trials, axis=0)

        rewarded_avg = rewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))
        unrewarded_avg = unrewarded_avg + np.abs(np.min([rewarded_avg, unrewarded_avg]))

        # Compute win-loss difference
        win_loss_diff = rewarded_avg - unrewarded_avg

        # Store session information
        session_differences.append(win_loss_diff)
        session_dates.append(session_date)

        # Store time axis (same for all sessions)
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Calculate peak difference (maximum absolute difference in post-cue window)
        post_cue_indices = time_axis > 0
        peak_diff = np.max(np.abs(win_loss_diff[post_cue_indices]))
        peak_differences.append(peak_diff)
        
        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if not session_df.empty and 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) > 0:
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
                    print(f"No min_pvalue data found for {subject_id}/{session_date}")
                    
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    if not session_differences:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
        # Filter out sessions without confidence values
        valid_sessions = [s for s in session_dates if s in session_confidences]
        
        if len(valid_sessions) < len(session_dates):
            print(f"Warning: Only {len(valid_sessions)} of {len(session_dates)} sessions have confidence values")
        
        if not valid_sessions:
            print("No sessions with valid confidence values found, using chronological sorting")
            sorted_indices = list(range(len(session_dates)))
        else:
            # Create mapping from session date to index
            date_to_idx = {date: idx for idx, date in enumerate(session_dates)}
            
            # Sort sessions by confidence (highest first)
            sorted_session_dates = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Get indices in the original data
            sorted_indices = [date_to_idx[date] for date in sorted_session_dates]
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_session_dates):
                print(f"{i+1}. Session {sess}: {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_indices = list(range(len(session_dates)))

    # Reorder arrays based on sorted indices
    session_differences = [session_differences[i] for i in sorted_indices]
    session_dates = [session_dates[i] for i in sorted_indices]
    peak_differences = [peak_differences[i] for i in sorted_indices]

    # Convert to array for heatmap
    win_loss_array = np.array(session_differences)

    # Create figure for heatmap and peak differences plot
    fig = plt.figure(figsize=(18, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # Plot heatmap of win-loss differences (top)
    ax_heatmap = fig.add_subplot(gs[0])

    # Flip the array vertically for display (earliest/lowest confidence at bottom)
    win_loss_array = np.flipud(win_loss_array)
    flipped_session_dates = session_dates[::-1]  # Reverse the session dates for y-axis labels

    # Create the heatmap
    im = ax_heatmap.imshow(win_loss_array,
                           aspect='auto',
                           extent=[time_axis[0], time_axis[-1], 0, len(session_differences)],
                           origin='lower',
                           cmap='RdBu_r',
                           interpolation='nearest')

    # Add vertical line at cue onset
    ax_heatmap.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Labels and formatting
    sort_type = "Computer Confidence" if comp_conf else "Chronological (oldest first)"
    ax_heatmap.set_xlabel('Time (s)', fontsize=16)
    ax_heatmap.set_ylabel('Session', fontsize=16)
    ax_heatmap.set_title(f'Win-Loss Signal Difference Across Sessions: {subject_id} (sorted by {sort_type})', fontsize=20)

    # Add specific y-tick labels at regular intervals
    tick_step = max(1, len(session_dates) // 10)  # Show at most 10 session labels
    y_ticks = np.arange(0.5, len(session_dates), tick_step)

    # Create labels for every tick_step interval
    y_label_with_num = []
    for i in range(0, len(flipped_session_dates), tick_step):
        if comp_conf and flipped_session_dates[i] in session_confidences:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]} (conf: {session_confidences[flipped_session_dates[i]]:.2f})")
        else:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]}")

    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_label_with_num)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Win-Loss ΔF/F Difference', fontsize=10)

    # Save figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    save_figure(fig, subject_id, "all_sessions", f"win_loss_difference_heatmap_{sort_suffix}")

    plt.show()

    # Return analysis results
    return {
        'subject_id': subject_id,
        'session_dates': session_dates,
        'time_axis': time_axis,
        'win_loss_differences': session_differences,
        'peak_differences': peak_differences,
        'session_confidences': session_confidences if comp_conf else None
    }


def analyze_session_average_heatmap(subject_id, comp_conf=False, behavior_df=None):
    """
    Create a heatmap visualization of average photometry signals across sessions

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    comp_conf : bool, optional (default=False)
        If True, sorts sessions by computer confidence rather than chronologically

    Returns:
    --------
    dict: Analysis results including average signals across all sessions
    """
    # Get all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
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

    # Store results for each session
    session_averages = []
    session_dates = []
    time_axis = None
    peak_averages = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf and behavior_df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            df = df[df['protocol'].str.contains('MatchingPennies', na=False)]
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting
    else:
        df = behavior_df  

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            continue

        if len(session_result['non_m_trials']) < 100:
            print(
                f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
            continue

        # Get photometry data from all non-missed trials
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])
        session_plots = session_result['plotting_data']

        if len(session_plots) == 0:
            print(f"Skipping {subject_id}/{session_date}, no photometry data.")
            continue

        # Compute average signal across all trials
        session_avg = np.mean(session_plots, axis=0)

        # Store session information
        session_averages.append(session_avg)
        session_dates.append(session_date)

        # Store time axis (same for all sessions)
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Calculate peak average (maximum signal in post-cue window)
        post_cue_indices = time_axis > 0
        peak_avg = np.max(session_avg[post_cue_indices])
        peak_averages.append(peak_avg)
        
        # Calculate computer confidence if requested
        if comp_conf:
            try:
                # Get data for this session
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date)]
                
                if not session_df.empty and 'min_pvalue' in session_df.columns:
                    # Extract p-values
                    p_values = session_df['min_pvalue'].values
                    
                    # Remove NaN values
                    p_values = p_values[~np.isnan(p_values)]
                    
                    if len(p_values) > 0:
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
                    print(f"No min_pvalue data found for {subject_id}/{session_date}")
                    
            except Exception as e:
                print(f"Error calculating confidence for {subject_id}/{session_date}: {e}")

    if not session_averages:
        print(f"No valid sessions found for {subject_id}")
        return None

    # Sort sessions based on confidence or chronologically
    if comp_conf and session_confidences:
        # Filter out sessions without confidence values
        valid_sessions = [s for s in session_dates if s in session_confidences]
        
        if len(valid_sessions) < len(session_dates):
            print(f"Warning: Only {len(valid_sessions)} of {len(session_dates)} sessions have confidence values")
        
        if not valid_sessions:
            print("No sessions with valid confidence values found, using chronological sorting")
            sorted_indices = list(range(len(session_dates)))
        else:
            # Create mapping from session date to index
            date_to_idx = {date: idx for idx, date in enumerate(session_dates)}
            
            # Sort sessions by confidence (highest first)
            sorted_session_dates = sorted(valid_sessions, key=lambda s: session_confidences.get(s, 0), reverse=True)
            
            # Get indices in the original data
            sorted_indices = [date_to_idx[date] for date in sorted_session_dates]
            
            # Print the confidence ranking
            print("\nSessions ranked by computer confidence (highest to lowest):")
            for i, sess in enumerate(sorted_session_dates):
                print(f"{i+1}. Session {sess}: {session_confidences[sess]:.4f}")
    else:
        # Use chronological sorting (default)
        sorted_indices = list(range(len(session_dates)))

    # Reorder arrays based on sorted indices
    session_averages = [session_averages[i] for i in sorted_indices]
    session_dates = [session_dates[i] for i in sorted_indices]
    peak_averages = [peak_averages[i] for i in sorted_indices]

    # Convert to array for heatmap
    avg_signal_array = np.array(session_averages)

    # Create figure for heatmap and peak averages plot
    fig = plt.figure(figsize=(18, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # Plot heatmap of average signals (top)
    ax_heatmap = fig.add_subplot(gs[0])

    # Flip the array vertically for display (earliest/lowest confidence at bottom)
    avg_signal_array = np.flipud(avg_signal_array)
    flipped_session_dates = session_dates[::-1]  # Reverse the session dates for y-axis labels

    # Create the heatmap with blue-to-yellow-to-white colormap
    im = ax_heatmap.imshow(avg_signal_array,
                           aspect='auto',
                           extent=[time_axis[0], time_axis[-1], 0, len(session_averages)],
                           origin='lower',
                           cmap='viridis',  # blue-low, yellow/white-high
                           interpolation='nearest')

    # Add vertical line at cue onset
    ax_heatmap.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    # Labels and formatting
    sort_type = "Computer Confidence" if comp_conf else "Chronological (oldest first)"
    ax_heatmap.set_xlabel('Time (s)', fontsize=12)
    ax_heatmap.set_ylabel('Session', fontsize=12)
    ax_heatmap.set_title(f'Average LC Signal Across Sessions: {subject_id} (sorted by {sort_type})', fontsize=20)

    # Add specific y-tick labels at regular intervals
    tick_step = max(1, len(session_dates) // 10)  # Show at most 10 session labels
    y_ticks = np.arange(0.5, len(session_dates), tick_step)

    # Create labels for every tick_step interval
    y_label_with_num = []
    for i in range(0, len(flipped_session_dates), tick_step):
        if comp_conf and flipped_session_dates[i] in session_confidences:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]} (conf: {session_confidences[flipped_session_dates[i]]:.2f})")
        else:
            y_label_with_num.append(f"{i+1}: {flipped_session_dates[i]}")

    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_label_with_num)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('ΔF/F', fontsize=10)


    # Save figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    save_figure(fig, subject_id, "all_sessions", f"average_signal_heatmap_{sort_suffix}")

    plt.show()

def add_statistical_test(subject_id, quartile_labels, all_choice_switches, quartile_trial_counts,
                         quartile_switch_rates):
    """
    Add statistical analysis to the loss trials signal quartiles analysis.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    quartile_labels : numpy.ndarray
        Array containing quartile labels (0-3) for each trial
    all_choice_switches : numpy.ndarray
        Boolean array indicating whether a switch occurred for each trial
    quartile_trial_counts : list
        List of trial counts for each quartile
    quartile_switch_rates : list
        List of switch percentages for each quartile
    """

    print("\n=== Statistical Analysis of Switch Rates Across Quartiles ===")

    contingency_table = []

    for quartile in range(4):
        quartile_mask = quartile_labels == quartile
        trials_in_quartile = np.sum(quartile_mask)

        if trials_in_quartile > 0:
            switch_count = np.sum(all_choice_switches[quartile_mask])
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

def analyze_signal_quartiles_by_outcome(subject_id, signal_window='pre_cue', condition='loss', plot_verification=True, behavior_df=None):
    """
    Analyze trials based on photometry signal in specified time window and determine choice switching behavior.
    Works with both win and loss trials as the condition.
    
    For post-cue windows ('early_post' and 'late_post'):
    - Selects current win/loss trials (T0) and sorts them by signal in the time window
    - Calculates % of trials where the next choice (T+1) is different from current choice (T0)
    
    For pre-cue window ('pre_cue'):
    - Selects trials that follow a win/loss (T0 follows T-1 win/loss) and sorts by pre-cue signal
    - Calculates % of trials where the current choice (T0) is different from previous choice (T-1)
    
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
        
    Returns:
    --------
    dict: Analysis results including quartile switch rates and plotted data
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
    
    # Determine analysis mode based on time window
    is_pre_cue_analysis = signal_window == 'pre_cue'
    
    # Store trial data and corresponding behavior
    all_trials_data = []         # Photometry data for all selected trials
    all_trial_signals = []       # Average signal in window for selected trials
    all_choice_switches = []     # Boolean: True if switch in choice
    all_next_reward = []         # Boolean: True if next trial rewarded 
    all_state_probs = []         # Store state probabilities for each trial
    time_axis = None             # Will be set from the first valid session
    
    # Set the reward value we're looking for based on condition
    target_reward = 1 if condition == 'win' else 0
    
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Get matching pennies sessions from behavior dataframe if provided, otherwise load from parquet
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
    sessions = sorted([d for d in os.listdir(subject_dir)
                if os.path.isdir(os.path.join(subject_dir, d)) and
                os.path.exists(os.path.join(subject_dir, d, "deltaff.npy")) and
                d in matching_pennies_sessions])
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue
            
        # Store time axis from the first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']
            
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Skip sessions with too few trials
        min_required = 3  # Need at least 3 trials for a meaningful analysis
        if len(rewards) < min_required:
            print(f"Skipping {subject_id}/{session_date}, insufficient trials")
            continue
            
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
        
        # Get state probabilities for this session if parquet data is available
        if behavior_df is not None:
            # Filter from the provided dataframe (which is already filtered by protocol)
            session_df = behavior_df[(behavior_df['subjid'] == subject_id) & 
                                    (behavior_df['date'] == session_date)]
        else:
            # This is a fallback if no behavior_df is provided
            try:
                # Load from parquet file and filter
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)
                session_df = df[(df['subjid'] == subject_id) & 
                            (df['date'] == session_date) & 
                            (df["ignore"] == 0) & 
                            (df['protocol'].str.contains('MatchingPennies', na=False))]
            except Exception as e:
                print(f"Error loading parquet data: {e}")
                session_df = None
        
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
                
                # Get state probabilities if available
                state_info = {'p_stochastic': None, 'p_leftbias': None, 'p_rightbias': None}
                if session_df is not None and i < len(session_df):
                    # Get state probabilities for this trial
                    state_info['p_stochastic'] = session_df.iloc[i].get('p_stochastic', None)
                    state_info['p_leftbias'] = session_df.iloc[i].get('p_leftbias', None) 
                    state_info['p_rightbias'] = session_df.iloc[i].get('p_rightbias', None)
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[curr_trial_idx])  # Current trial's reward outcome
                all_state_probs.append(state_info)
                
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
                
                # Get state probabilities if available
                state_info = {'p_stochastic': None, 'p_leftbias': None, 'p_rightbias': None}
                if session_df is not None and i < len(session_df):
                    # Get state probabilities for this trial
                    state_info['p_stochastic'] = session_df.iloc[i].get('p_stochastic', None)
                    state_info['p_leftbias'] = session_df.iloc[i].get('p_leftbias', None) 
                    state_info['p_rightbias'] = session_df.iloc[i].get('p_rightbias', None)
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[next_trial_idx])  # Next trial's reward outcome
                all_state_probs.append(state_info)
    
    # Check if we found any valid trials
    if len(all_trials_data) == 0:
        print(f"No valid {condition} trials found for analysis ({signal_window}) for {subject_id}")
        return None
        
    # Convert lists to numpy arrays
    all_trials_data = np.array(all_trials_data)
    all_trial_signals = np.array(all_trial_signals)
    all_choice_switches = np.array(all_choice_switches)
    all_next_reward = np.array(all_next_reward)
    
    # Sort trials into quartiles based on signal
    quartile_labels = pd.qcut(all_trial_signals, 4, labels=False)
    
    # Calculate switch rate for each quartile
    quartile_switch_rates = []
    quartile_trial_counts = []
    quartile_reward_rates = []
    
    # State threshold analysis per quartile
    quartile_state_counts = []
    state_threshold = 0.8  # Threshold for state assignment
    
    # Process each quartile
    for quartile in range(4):
        quartile_mask = quartile_labels == quartile
        trials_in_quartile = np.sum(quartile_mask)
        
        if trials_in_quartile > 0:
            # Calculate switch rate (% of trials where choice switched)
            switch_count = np.sum(all_choice_switches[quartile_mask])
            switch_rate = (switch_count / trials_in_quartile) * 100
            
            # Calculate reward rate for the relevant trial
            reward_rate = (np.sum(all_next_reward[quartile_mask]) / trials_in_quartile) * 100
            
            quartile_switch_rates.append(switch_rate)
            quartile_trial_counts.append(trials_in_quartile)
            quartile_reward_rates.append(reward_rate)
            
            # Calculate state proportions for this quartile
            quartile_states = {'stochastic': 0, 'biased': 0, 'uncertain': 0}
            quartile_indices = np.where(quartile_mask)[0]
            
            for idx in quartile_indices:
                state_info = all_state_probs[idx]
                p_stoch = state_info['p_stochastic']
                p_left = state_info['p_leftbias']
                p_right = state_info['p_rightbias']
                
                # Skip trials with missing state information
                if p_stoch is None or p_left is None or p_right is None:
                    quartile_states['uncertain'] += 1
                    continue
                
                # Assign state based on threshold
                if p_stoch >= state_threshold:
                    quartile_states['stochastic'] += 1
                elif p_left >= state_threshold or p_right >= state_threshold:
                    quartile_states['biased'] += 1
                else:
                    quartile_states['uncertain'] += 1
            
            quartile_state_counts.append(quartile_states)
        else:
            quartile_switch_rates.append(0)
            quartile_trial_counts.append(0)
            quartile_reward_rates.append(0)
            quartile_state_counts.append({'stochastic': 0, 'biased': 0, 'uncertain': 0})
            
    # Print results
    outcome_label = "Win" if condition == 'win' else "Loss"
    print(f"\n=== Analysis of {signal_window} Signal Quartiles ({outcome_label} trials): {subject_id} ===")
    print(f"Time window: ({window_start}s to {window_end}s)")
    print(f"Total trials analyzed: {len(all_trial_signals)}")
    
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
    
    # Print state distribution by quartile
    print("\nState Distribution by Signal Quartile (threshold = 0.8):")
    for quartile in range(4):
        states = quartile_state_counts[quartile]
        total = quartile_trial_counts[quartile]
        
        if total > 0:
            # Calculate percentages
            stoch_pct = (states['stochastic'] / total) * 100
            biased_pct = (states['biased'] / total) * 100
            uncertain_pct = (states['uncertain'] / total) * 100
            
            print(f"Quartile {quartile + 1}: "
                  f"Stochastic: {states['stochastic']} ({stoch_pct:.1f}%), "
                  f"Biased: {states['biased']} ({biased_pct:.1f}%), "
                  f"Uncertain: {states['uncertain']} ({uncertain_pct:.1f}%)")
        else:
            print(f"Quartile {quartile + 1}: No trials")
    
    # Create verification plot if requested
    if plot_verification:
        # Create a single figure for just the photometry plot (no subplots needed)
        plt.figure(figsize=(12, 7))
        
        # Define colors for quartiles
        colors = ['blue', 'green', 'orange', 'red']  # Colors for quartiles
        
        # Plot each quartile's average trace
        for quartile in range(4):
            quartile_mask = quartile_labels == quartile
            if np.sum(quartile_mask) > 0:
                quartile_data = all_trials_data[quartile_mask]
                quartile_avg = np.mean(quartile_data, axis=0)
                quartile_sem = calculate_sem(quartile_data, axis=0)
                
                plt.fill_between(time_axis,
                               quartile_avg - quartile_sem,
                               quartile_avg + quartile_sem,
                               color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                       color=colors[quartile], linewidth=2,
                       label=f'Quartile {quartile+1} (n={quartile_trial_counts[quartile]})')
                       
        # Highlight the time window used for sorting
        plt.axvspan(window_start, window_end, color='gray', alpha=0.3, label='Sorting Window')
        
        # Add reference lines
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('ΔF/F', fontsize=12)
        
        if is_pre_cue_analysis:
            plt.title(f'Trials After {outcome_label} Sorted by Pre-Cue Signal: {subject_id}', fontsize=14)
        else:
            plt.title(f'{outcome_label} Trials Sorted by {signal_window} Signal: {subject_id}', fontsize=14)
            
        plt.legend(loc='upper right')
        plt.xlim([-pre_cue_time, post_cue_time])
        
        # Add total trials information as text at the bottom of the plot
        plt.figtext(0.5, 0.01, f"Total trials analyzed: {len(all_trial_signals)}", 
                   ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Save the figure
        condition_str = condition  # 'win' or 'loss'
        mode_suffix = f"after_{condition_str}" if is_pre_cue_analysis else f"{condition_str}_trials"
        save_figure(plt.gcf(), subject_id, "pooled", f"{mode_suffix}_{signal_window}_quartiles")
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom
        plt.show()

    # Add statistical analysis if we have enough data
    if len(all_trial_signals) >= 100:  # Only perform stats with sufficient data
        add_statistical_test(subject_id, quartile_labels, all_choice_switches,
                             quartile_trial_counts, quartile_switch_rates)
        
    # Return analysis results
    return {
        'subject_id': subject_id,
        'time_axis': time_axis,
        'signal_window': signal_window,
        'condition': condition,
        'window_bounds': (window_start, window_end),
        'all_trials_data': all_trials_data,
        'all_trial_signals': all_trial_signals,
        'quartile_labels': quartile_labels,
        'quartile_switch_rates': quartile_switch_rates,
        'quartile_trial_counts': quartile_trial_counts,
        'quartile_reward_rates': quartile_reward_rates,
        'is_pre_cue_analysis': is_pre_cue_analysis
    }

def analyze_switch_probabilities(subject_id, session_date=None, behavior_df=None):
    """
    Analyze probability of switching choices following win vs loss trials 
    for a subject, and test if there is a significant difference.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
        
    Returns:
    --------
    dict: Analysis results including switch probabilities and statistical tests
    """
    # Store results
    results = {
        'subject_id': subject_id,
        'after_win': {'switches': 0, 'total': 0, 'rate': 0},
        'after_loss': {'switches': 0, 'total': 0, 'rate': 0}, 
        'sessions_analyzed': 0,
        'session_results': []
    }
    
    # Determine which sessions to analyze
    if session_date is None:
        if behavior_df is not None:
            # Extract sessions for this subject from behavior_df
            sessions = sorted(behavior_df[behavior_df['subjid'] == subject_id]['date'].unique())
        else:
            # Get all sessions from the file system
            subject_path = os.path.join(base_dir, subject_id)
            sessions = sorted([d for d in os.listdir(subject_path)
                            if os.path.isdir(os.path.join(subject_path, d)) and
                            os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]
    
    print(f"Analyzing switch probabilities for {subject_id} across {len(sessions)} sessions...")
    
    # Process each session
    for sess in sessions:
        print(f"Processing {subject_id}/{sess}...")
        session_result = process_session(subject_id, sess, behavior_df=behavior_df)
        if not session_result:
            continue
            
        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])
        
        # Skip sessions with too few trials
        if len(choices) < 10:  # Need reasonable number of trials
            print(f"Skipping {subject_id}/{sess} - too few trials")
            continue
            
        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_choices = choices[non_miss_mask]
        filtered_rewards = rewards[non_miss_mask]
        
        # Skip if not enough trials after filtering
        if len(filtered_choices) < 10:
            print(f"Skipping {subject_id}/{sess} - too few non-missed trials")
            continue
            
        # Initialize counters for this session
        session_stats = {
            'session_date': sess,
            'after_win': {'switches': 0, 'total': 0, 'rate': 0},
            'after_loss': {'switches': 0, 'total': 0, 'rate': 0}
        }
        
        # Count switches after wins and losses
        for i in range(1, len(filtered_choices)):
            prev_outcome = filtered_rewards[i-1]
            curr_choice = filtered_choices[i]
            prev_choice = filtered_choices[i-1]
            
            # Check if choice switched
            switched = curr_choice != prev_choice
            
            # Categorize by previous trial outcome
            if prev_outcome == 1:  # Win
                session_stats['after_win']['total'] += 1
                if switched:
                    session_stats['after_win']['switches'] += 1
            else:  # Loss
                session_stats['after_loss']['total'] += 1
                if switched:
                    session_stats['after_loss']['switches'] += 1
        
        # Calculate switch rates for this session
        if session_stats['after_win']['total'] > 0:
            session_stats['after_win']['rate'] = (session_stats['after_win']['switches'] / session_stats['after_win']['total'] * 100)
        if session_stats['after_loss']['total'] > 0:
            session_stats['after_loss']['rate'] = (session_stats['after_loss']['switches'] / 
                                                 session_stats['after_loss']['total'] * 100)
        
        # Add session data to overall results
        results['session_results'].append(session_stats)
        results['after_win']['switches'] += session_stats['after_win']['switches']
        results['after_win']['total'] += session_stats['after_win']['total']
        results['after_loss']['switches'] += session_stats['after_loss']['switches']
        results['after_loss']['total'] += session_stats['after_loss']['total']
        results['sessions_analyzed'] += 1
        
        # Print session results
        print(f"  Session {sess} results:")
        print(f"    After win: {session_stats['after_win']['switches']}/{session_stats['after_win']['total']} " +
              f"switches ({session_stats['after_win']['rate']:.1f}%)")
        print(f"    After loss: {session_stats['after_loss']['switches']}/{session_stats['after_loss']['total']} " +
              f"switches ({session_stats['after_loss']['rate']:.1f}%)")
    
    # Calculate overall switch rates
    if results['after_win']['total'] > 0:
        results['after_win']['rate'] = results['after_win']['switches'] / results['after_win']['total'] * 100
    if results['after_loss']['total'] > 0:
        results['after_loss']['rate'] = results['after_loss']['switches'] / results['after_loss']['total'] * 100
    
    # Statistical testing: Chi-square test for independence
    contingency_table = np.array([
        [results['after_win']['switches'], results['after_win']['total'] - results['after_win']['switches']],
        [results['after_loss']['switches'], results['after_loss']['total'] - results['after_loss']['switches']]
    ])
    
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
        results['statistics'] = {
            'chi2': chi2,
            'p_value': p_chi2,
            'dof': dof,
            'expected': expected
        }
        
        # Fisher's exact test (more appropriate for smaller samples)
        odds_ratio, p_fisher = fisher_exact(contingency_table)
        results['statistics']['fisher_p_value'] = p_fisher
        results['statistics']['odds_ratio'] = odds_ratio
        
    except Exception as e:
        print(f"Error in statistical test: {e}")
        results['statistics'] = {'error': str(e)}
    
    # Print overall results with statistical test
    print("\n=== Overall Switch Probability Analysis ===")
    print(f"Subject: {subject_id}")
    print(f"Sessions analyzed: {results['sessions_analyzed']}")
    print(f"After win: {results['after_win']['switches']}/{results['after_win']['total']} " +
          f"switches ({results['after_win']['rate']:.1f}%)")
    print(f"After loss: {results['after_loss']['switches']}/{results['after_loss']['total']} " +
          f"switches ({results['after_loss']['rate']:.1f}%)")
    
    # Print statistical test results
    if 'statistics' in results and 'error' not in results['statistics']:
        print("\nStatistical Tests:")
        print(f"Chi-square test: χ² = {results['statistics']['chi2']:.2f}, p = {results['statistics']['p_value']:.5f}")
        print(f"Fisher's exact test: p = {results['statistics']['fisher_p_value']:.5f}, " + 
              f"odds ratio = {results['statistics']['odds_ratio']:.2f}")
        
        sig_level = 0.05
        if results['statistics']['p_value'] < sig_level or results['statistics']['fisher_p_value'] < sig_level:
            # Determine which condition has higher switch rate
            if results['after_win']['rate'] > results['after_loss']['rate']:
                print("\nResult: Subject significantly more likely to switch after wins than losses")
            else:
                print("\nResult: Subject significantly more likely to switch after losses than wins")
        else:
            print("\nResult: No significant difference in switch behavior after wins vs losses")
    
    # Create visualization
    if results['sessions_analyzed'] > 0:
        plt.figure(figsize=(12, 6))
        
        # If analyzing multiple sessions, plot session-by-session rates
        if len(results['session_results']) > 1:
            # Extract data for plotting
            sessions = list(range(1, results['sessions_analyzed'] + 1))
            win_rates = [s['after_win']['rate'] for s in results['session_results']]
            loss_rates = [s['after_loss']['rate'] for s in results['session_results']]
            
            # Plot session data
            plt.subplot(1, 2, 1)
            plt.plot(sessions, win_rates, 'o-', color='green', label='After Win')
            plt.plot(sessions, loss_rates, 'o-', color='red', label='After Loss')
            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Session Number')
            plt.ylabel('Switch Rate (%)')
            plt.title('Switch Rates Across Sessions')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Bar plot of overall rates on right side
            plt.subplot(1, 2, 2)
            
        # Create bar chart (standalone for single session, or right panel for multiple)
        bars = plt.bar([1, 2], 
                      [results['after_win']['rate'], results['after_loss']['rate']], 
                      color=['green', 'red'])
        
        # Add labels to bars
        for bar, count, total in zip(bars, 
                                   [results['after_win']['switches'], results['after_loss']['switches']],
                                   [results['after_win']['total'], results['after_loss']['total']]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{count}/{total}\n({height:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        plt.xticks([1, 2], ['After Win', 'After Loss'])
        plt.ylabel('Switch Rate (%)')
        plt.title('Overall Switch Probabilities')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add statistical annotation if we have statistics
        if 'statistics' in results and 'p_value' in results['statistics']:
            p_val = min(results['statistics']['p_value'], results['statistics']['fisher_p_value'])
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "n.s."
                
            # Draw a line connecting the bars with significance
            height = max(results['after_win']['rate'], results['after_loss']['rate']) + 5
            plt.plot([1, 2], [height, height], 'k-', linewidth=1.5)
            plt.text(1.5, height + 2, sig_text, ha='center', va='bottom', fontsize=14)
            
            # Add p-value text
            plt.text(1.5, height + 8, f"p = {p_val:.4f}", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        save_figure(plt.gcf(), subject_id, "pooled", "switch_probabilities")
        plt.show()
    
    return results


def analyze_switch_probability_quartiles(subject_id, session_date=None, win_loss=False, behavior_df=None):
    """
    Analyze photometry signals binned by switch probability quartiles for a single session or pooled across sessions.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    session_date : str, optional
        Specific session to analyze. If None, analyze all sessions.
    win_loss : bool, optional
        Whether to split by rewarded/unrewarded trials

    Returns:
    --------
    dict: Analysis results including quartile bins and switch probability values
    """
    all_plotting_data = []
    all_switch_probs = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                           if os.path.isdir(os.path.join(subject_path, d)) and
                           os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

        # Process each session separately
        for session_date in sessions:
            session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(
                    f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate switch probabilities for this session
            behavior_data = session_result['behavioral_data']
            choices = np.array(behavior_data['choice'])

            # Filter out missed trials
            non_miss_mask = choices != 'M'
            filtered_choices = choices[non_miss_mask]

            # Skip if not enough choices after filtering
            if len(filtered_choices) < 10:
                print(f"Skipping {subject_id}/{session_date}, insufficient non-missed trials.")
                continue

            # Calculate moving average switch probability with window size 15
            window_size = 15
            switch_probs = []
            overall_switch_rate = 0

            # First calculate overall switch rate for the session
            for i in range(1, len(filtered_choices)):
                if filtered_choices[i] != filtered_choices[i - 1]:
                    overall_switch_rate += 1
            if len(filtered_choices) > 1:
                overall_switch_rate /= (len(filtered_choices) - 1)

            # Calculate moving average switch probabilities
            for i in range(len(filtered_choices)):
                if i < window_size:
                    # For early trials, use available data plus weighted overall rate
                    switches = 0
                    comparisons = 0

                    for j in range(1, i + 1):
                        if filtered_choices[j] != filtered_choices[j - 1]:
                            switches += 1
                        comparisons += 1

                    if comparisons > 0:
                        recent_rate = switches / comparisons
                    else:
                        recent_rate = overall_switch_rate

                    # Weight: how much of the window size we have available
                    available_weight = comparisons / window_size
                    switch_prob = (available_weight * recent_rate +
                                   (1 - available_weight) * overall_switch_rate)
                else:
                    # For later trials, use full window
                    switches = 0
                    for j in range(i - window_size + 1, i + 1):
                        if j > 0 and filtered_choices[j] != filtered_choices[j - 1]:
                            switches += 1
                    switch_prob = switches / window_size

                switch_probs.append(switch_prob)

            switch_probs = np.array(switch_probs)

            # Map these probabilities to the valid trials with photometry data
            non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                      if idx in session_result["non_m_trials"]])

            # Get the original indices for non-missed trials
            orig_non_miss_indices = np.where(non_miss_mask)[0]

            # Create mapping from valid photometry trial to behavior index
            trial_switch_probs = []
            for idx in non_m_indices:
                # Find the behavior index for this photometry trial
                if idx < len(orig_non_miss_indices):
                    behavior_idx = orig_non_miss_indices[idx]
                    if behavior_idx < len(switch_probs):
                        trial_switch_probs.append(switch_probs[behavior_idx])
                    else:
                        trial_switch_probs.append(overall_switch_rate)
                else:
                    trial_switch_probs.append(overall_switch_rate)

            # Store data for this session
            all_plotting_data.append(session_result['plotting_data'])
            all_switch_probs.extend(trial_switch_probs)
            all_reward_outcomes.append(session_result["reward_outcomes"][non_m_indices])
            time_axis = session_result['time_axis']

        plot_title = f'Pooled Photometry by Switch Probability Quartiles: {subject_id}'
        plotting_data = np.vstack(all_plotting_data)
        switch_probs = np.array(all_switch_probs)
        reward_outcomes = np.concatenate(all_reward_outcomes)

    else:
        # Single session analysis
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Calculate switch probabilities for this session
        behavior_data = session_result['behavioral_data']
        choices = np.array(behavior_data['choice'])

        # Filter out missed trials
        non_miss_mask = choices != 'M'
        filtered_choices = choices[non_miss_mask]

        # Skip if not enough choices after filtering
        if len(filtered_choices) < 10:
            print(f"Insufficient non-missed trials in {subject_id}/{session_date}")
            return None

        # Calculate moving average switch probability with window size 15
        window_size = 15
        switch_probs = []
        overall_switch_rate = 0

        # First calculate overall switch rate for the session
        for i in range(1, len(filtered_choices)):
            if filtered_choices[i] != filtered_choices[i - 1]:
                overall_switch_rate += 1
        if len(filtered_choices) > 1:
            overall_switch_rate /= (len(filtered_choices) - 1)

        # Calculate moving average switch probabilities
        for i in range(len(filtered_choices)):
            if i < window_size:
                # For early trials, use available data plus weighted overall rate
                switches = 0
                comparisons = 0

                for j in range(1, i + 1):
                    if filtered_choices[j] != filtered_choices[j - 1]:
                        switches += 1
                    comparisons += 1

                if comparisons > 0:
                    recent_rate = switches / comparisons
                else:
                    recent_rate = overall_switch_rate

                # Weight: how much of the window size we have available
                available_weight = comparisons / window_size
                switch_prob = (available_weight * recent_rate +
                               (1 - available_weight) * overall_switch_rate)
            else:
                # For later trials, use full window
                switches = 0
                for j in range(i - window_size + 1, i + 1):
                    if j > 0 and filtered_choices[j] != filtered_choices[j - 1]:
                        switches += 1
                switch_prob = switches / window_size

            switch_probs.append(switch_prob)

        switch_probs = np.array(switch_probs)

        # Map these probabilities to the valid trials with photometry data
        non_m_indices = np.array([i for i, idx in enumerate(session_result["valid_trials"])
                                  if idx in session_result["non_m_trials"]])

        # Get the original indices for non-missed trials
        orig_non_miss_indices = np.where(non_miss_mask)[0]

        # Create mapping from valid photometry trial to behavior index
        trial_switch_probs = []
        for idx in non_m_indices:
            # Find the behavior index for this photometry trial
            if idx < len(orig_non_miss_indices):
                behavior_idx = orig_non_miss_indices[idx]
                if behavior_idx < len(switch_probs):
                    trial_switch_probs.append(switch_probs[behavior_idx])
                else:
                    trial_switch_probs.append(overall_switch_rate)
            else:
                trial_switch_probs.append(overall_switch_rate)

        plotting_data = session_result['plotting_data']
        switch_probs = np.array(trial_switch_probs)
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        time_axis = session_result['time_axis']
        plot_title = f'LC Signal by Switch Probability Quartiles: {subject_id} - {session_date}'

    # Create quartile bins based on switch probabilities
    quartile_bins = pd.qcut(switch_probs, q=4, labels=False)

    # Calculate average switch probability for each quartile
    quartile_averages = []
    for quartile in range(4):
        quartile_avg = np.mean(switch_probs[quartile_bins == quartile])
        quartile_averages.append(quartile_avg)

    # Print average switch probabilities for each quartile
    print(f"\nAverage switch probabilities by quartile:")
    for quartile in range(4):
        print(f"Quartile {quartile + 1}: {quartile_averages[quartile]:.4f}")

    # Create the plot
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']  # From lowest to highest switch probability

    if win_loss:
        for quartile in range(4):
            quartile_rewarded = (quartile_bins == quartile) & (reward_outcomes == 1)
            quartile_unrewarded = (quartile_bins == quartile) & (reward_outcomes == 0)

            if np.sum(quartile_rewarded) > 0:
                rewarded_avg = np.mean(plotting_data[quartile_rewarded], axis=0)
                rewarded_sem = calculate_sem(plotting_data[quartile_rewarded], axis=0)
                plt.fill_between(time_axis,
                                 rewarded_avg - rewarded_sem,
                                 rewarded_avg + rewarded_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, rewarded_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} Rewarded (n={np.sum(quartile_rewarded)})')

            if np.sum(quartile_unrewarded) > 0:
                unrewarded_avg = np.mean(plotting_data[quartile_unrewarded], axis=0)
                unrewarded_sem = calculate_sem(plotting_data[quartile_unrewarded], axis=0)
                plt.plot(time_axis, unrewarded_avg,
                         color=colors[quartile], linewidth=2, linestyle='--',
                         label=f'Quartile {quartile + 1} Unrewarded (n={np.sum(quartile_unrewarded)})')
    else:
        for quartile in range(4):
            quartile_trials = quartile_bins == quartile
            if np.sum(quartile_trials) > 0:
                quartile_avg = np.mean(plotting_data[quartile_trials], axis=0)
                quartile_sem = calculate_sem(plotting_data[quartile_trials], axis=0)

                plt.fill_between(time_axis,
                                 quartile_avg - quartile_sem,
                                 quartile_avg + quartile_sem,
                                 color=colors[quartile], alpha=0.3)
                plt.plot(time_axis, quartile_avg,
                         color=colors[quartile], linewidth=2,
                         label=f'Quartile {quartile + 1} (n={np.sum(quartile_trials)})')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')

    # Add text with quartile averages at the bottom of the plot
    quartile_text = "Average switch probabilities: " + ", ".join(
        [f"Q{q + 1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"switch_probability_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled",
                f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()

    return {
        'quartile_bins': quartile_bins,
        'switch_probs': switch_probs,
        'quartile_averages': quartile_averages
    }


def analyze_signal_state_effects_on_switching(subject_id, signal_window='pre_cue', condition='loss', behavior_df=None):
    """
    Perform multivariate logistic regression analysis to determine if photometry signal
    predicts choice switching independent of behavioral state.

    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
    signal_window : str, optional (default='pre_cue')
        Time window to use for calculating average photometry signal:
        - 'pre_cue': -0.75s to -0.25s before lick
        - 'early_post': +1s to +2s after lick
        - 'late_post': +3.5s to +4.5s after lick
    condition : str, optional (default='loss')
        Which trial outcome to analyze: 'loss' or 'win'

    Returns:
    --------
    dict: Results of the logistic regression analysis including model coefficients and statistics
    """
    if behavior_df is None:
        try:
            behavior_df = load_filtered_behavior_data("MatchingPennies")
            if behavior_df is None:
                print("Error loading behavior data")
                return None
        except Exception as e:
            print(f"Error loading behavior data: {e}")
            return None
        
    subject_df = behavior_df[behavior_df['subjid'] == subject_id]
    if subject_df.empty:
        print(f"No behavioral data found for subject {subject_id}")
        return None

    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    # Define time windows
    time_windows = {
        'pre_cue': (-0.75, -0.25),
        'early_post': (1.0, 2.0),
        'late_post': (3.5, 4.5)
    }

    if signal_window not in time_windows:
        raise ValueError(f"Invalid signal_window. Choose from: {list(time_windows.keys())}")

    # Get time window bounds
    window_start, window_end = time_windows[signal_window]
    is_pre_cue_analysis = signal_window == 'pre_cue'

    # Set target reward based on condition
    target_reward = 1 if condition == 'win' else 0

    print(f"Analyzing signal and state effects on switching for {subject_id}...")
    print(f"Condition: {condition} trials, Signal window: {signal_window} ({window_start}s to {window_end}s)")

    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Data containers
    trial_data = []  # Will store everything we need for regression
    time_axis = None

    sessions = sorted(subject_df['date'].unique())

    state_data_available = 'p_stochastic' in subject_df.columns
    if state_data_available:
        print("State probability data available in behavioral dataframe")
    else:
        print("No state probability data found in behavioral dataframe")

    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date, behavior_df=behavior_df)
        if not session_result:
            continue

        # Store time axis from first valid session
        if time_axis is None:
            time_axis = session_result['time_axis']

        # Get behavioral data
        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        choices = np.array(behavior_data['choice'])

        # Skip sessions with too few trials
        if len(choices) < 5:
            print(f"Skipping {subject_id}/{session_date}, insufficient trials")
            continue

        # Get valid trials with photometry data
        valid_trials = session_result['valid_trials']
        epoched_data = session_result['epoched_data'][valid_trials]

        # Find time indices for the specified window
        window_idx_start = np.where(time_axis >= window_start)[0][0]
        window_idx_end = np.where(time_axis <= window_end)[0][-1]

        # Get state probabilities for this session from the filtered dataframe
        session_df = subject_df[subject_df['date'] == session_date]
        if session_df.empty:
            print(f"No behavioral data found in filtered dataframe for {subject_id}/{session_date}")
            continue
        else:
            print(f"Found {len(session_df)} trials in the filtered dataframe for {subject_id}/{session_date}")

        # Only include non-missed trials in analysis
        non_miss_mask = choices != 'M'
        non_miss_trials = np.where(non_miss_mask)[0]

        # Process each valid trial based on condition
        for i, trial_idx in enumerate(valid_trials):
            # Skip if this isn't a valid trial index
            if trial_idx >= len(choices):
                continue

            # Skip missed trials
            if choices[trial_idx] == 'M':
                continue

            # Check if current trial matches our condition (win/loss)
            if is_pre_cue_analysis:
                # For pre-cue, we need this to be after our target condition
                if trial_idx == 0 or rewards[trial_idx - 1] != target_reward:
                    continue

                # Need both current and previous trials to be non-missed
                if trial_idx - 1 not in non_miss_trials:
                    continue

                prev_trial_idx = trial_idx - 1
                curr_trial_idx = trial_idx

                # Calculate if choice switched from previous trial
                choice_switched = int(choices[prev_trial_idx] != choices[curr_trial_idx])

                # Get average photometry signal in the window
                signal_value = np.mean(epoched_data[i][window_idx_start:window_idx_end])

            else:  # Post-cue analysis
                # For post-cue, current trial must match our condition
                if rewards[trial_idx] != target_reward:
                    continue

                # Need a next trial that's non-missed
                if trial_idx >= len(choices) - 1 or choices[trial_idx + 1] == 'M':
                    continue

                curr_trial_idx = trial_idx
                next_trial_idx = trial_idx + 1

                # Calculate if next choice switched from current
                choice_switched = int(choices[curr_trial_idx] != choices[next_trial_idx])

                # Get average photometry signal in the window
                signal_value = np.mean(epoched_data[i][window_idx_start:window_idx_end])

            # Get state probabilities (if available)
            state_info = {
                'p_stochastic': 0,
                'p_leftbias': 0,
                'p_rightbias': 0
            }

            if len(session_df) > curr_trial_idx:
                # Extract probabilities from the filtered dataframe
                try:
                    state_info['p_stochastic'] = float(session_df.iloc[curr_trial_idx].get('p_stochastic', 0) or 0)
                    state_info['p_leftbias'] = float(session_df.iloc[curr_trial_idx].get('p_leftbias', 0) or 0)
                    state_info['p_rightbias'] = float(session_df.iloc[curr_trial_idx].get('p_rightbias', 0) or 0)
                except (IndexError, ValueError, TypeError) as e:
                    print(f"Error extracting state probabilities for trial {curr_trial_idx}: {e}")

            # Create trial data record with all necessary information
            trial_data.append({
                'session': session_date,
                'trial_idx': curr_trial_idx,
                'signal': signal_value,
                'switch': choice_switched,
                'p_stochastic': state_info['p_stochastic'],
                'p_leftbias': state_info['p_leftbias'],
                'p_rightbias': state_info['p_rightbias'],
                'p_biased': max(state_info['p_leftbias'], state_info['p_rightbias'])
            })

    # Check if we have enough data
    if len(trial_data) < 10:
        print(f"Insufficient data for analysis: only {len(trial_data)} valid trials")
        return None

    # Create DataFrame and prepare for regression
    data = pd.DataFrame(trial_data)

    # Print basic data summary
    print(f"\nAnalysis summary for {subject_id}:")
    print(f"  Total trials analyzed: {len(data)}")
    print(f"  Overall switch rate: {data['switch'].mean() * 100:.1f}%")

    # Check for constant values or zero variance and print state probability distribution
    print("\nState probability distributions:")
    state_use_for_model = {}

    for state in ['p_stochastic', 'p_biased']:
        mean_val = data[state].mean()
        median_val = data[state].median()
        min_val = data[state].min()
        max_val = data[state].max()
        std_val = data[state].std()
        print(f"  {state}: Mean={mean_val:.3f}, Median={median_val:.3f}, Range=[{min_val:.3f}, {max_val:.3f}], Std={std_val:.3f}")
        
        # Determine if this state has enough variation to use in model
        if min_val == max_val or std_val < 0.001:  # Effectively constant
            state_use_for_model[state] = False
            print(f"  Warning: {state} has no variation. Excluding from model.")
        else:
            state_use_for_model[state] = True

    # Add z-scored variables with proper handling of zero variance and NaN checking
    data['signal_z'] = (data['signal'] - data['signal'].mean()) / (data['signal'].std() or 1)

    # Clean and standardize state variables properly
    for state in ['p_stochastic', 'p_biased']:
        if state_use_for_model[state]:
            # Replace any NaN values first
            data[state] = data[state].fillna(data[state].mean())
            
            # Calculate z-score with safeguards
            mean_val = data[state].mean()
            std_val = data[state].std()
            
            # Only standardize if we have meaningful variation
            if std_val > 0.001:
                data[f'{state}_z'] = (data[state] - mean_val) / std_val
                
                # Check for inf/NaN and replace with zeros if found
                data[f'{state}_z'] = data[f'{state}_z'].replace([np.inf, -np.inf], 0).fillna(0)
            else:
                # If std is too small, just set all values to 0
                data[f'{state}_z'] = 0
        else:
            # No variation - set all values to 0
            data[f'{state}_z'] = 0

    # Create a function to fit model safely
    def fit_model_safely(model_formula, data):
        try:
            return model_formula.fit(disp=0)  # Turn off convergence messages
        except PerfectSeparationError:
            print("Warning: Perfect separation detected. Some coefficients will be inaccurate.")
            # Use a more robust method
            return model_formula.fit(method='bfgs', disp=0)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix detected. Using regularized fit.")
            # Add tiny regularization
            return model_formula.fit_regularized(alpha=0.01, disp=0)
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None

    # Run a series of logistic regression models

    # Model 1: Signal only - this should always work
    model1_formula = sm.Logit(data['switch'], sm.add_constant(data['signal_z']))
    model1 = fit_model_safely(model1_formula, data)

    # Model 2: State probabilities only (using only variables with variation)
    if state_use_for_model['p_stochastic'] or state_use_for_model['p_biased']:
        # Select variables with variation
        state_vars = []
        if state_use_for_model['p_stochastic']:
            state_vars.append('p_stochastic_z')
        if state_use_for_model['p_biased']:
            state_vars.append('p_biased_z')

        if state_vars:
            model2_formula = sm.Logit(data['switch'], sm.add_constant(data[state_vars]))
            model2 = fit_model_safely(model2_formula, data)
        else:
            print("No state variables have sufficient variation. Skipping state-only model.")
            model2 = None
    else:
        print("No state variables have sufficient variation. Skipping state-only model.")
        model2 = None

    # Model 3: Signal + State probabilities
    if state_use_for_model['p_stochastic'] or state_use_for_model['p_biased']:
        # Create combined list of variables
        model3_vars = ['signal_z']
        if state_use_for_model['p_stochastic']:
            model3_vars.append('p_stochastic_z')
        if state_use_for_model['p_biased']:
            model3_vars.append('p_biased_z')

        model3_formula = sm.Logit(data['switch'], sm.add_constant(data[model3_vars]))
        model3 = fit_model_safely(model3_formula, data)
    else:
        print("No state variables have sufficient variation. Model 3 will be identical to Model 1.")
        model3 = model1  # Just use signal-only model

    # Model 4: Signal + State + Interaction (only if we have state vars)
    if state_use_for_model['p_stochastic'] or state_use_for_model['p_biased']:
        # Create interaction terms for variables with variation
        model4_vars = ['signal_z']
        interaction_terms = []

        if state_use_for_model['p_stochastic']:
            model4_vars.append('p_stochastic_z')
            data['signal_x_stoch'] = data['signal_z'] * data['p_stochastic_z']
            interaction_terms.append('signal_x_stoch')

        if state_use_for_model['p_biased']:
            model4_vars.append('p_biased_z')
            data['signal_x_biased'] = data['signal_z'] * data['p_biased_z']
            interaction_terms.append('signal_x_biased')

        # Add interaction terms to model variables
        model4_vars.extend(interaction_terms)

        model4_formula = sm.Logit(data['switch'], sm.add_constant(data[model4_vars]))
        model4 = fit_model_safely(model4_formula, data)
    else:
        print("No state variables have sufficient variation. Skipping full interaction model.")
        model4 = None

    # Compute odds ratios for all models
    def get_odds_ratios(model):
        if model is None:
            return None
        return np.exp(model.params)

    odds_ratios1 = get_odds_ratios(model1)
    odds_ratios2 = get_odds_ratios(model2)
    odds_ratios3 = get_odds_ratios(model3)
    odds_ratios4 = get_odds_ratios(model4)

    # Print model results
    print("\n=== Model 1: Signal Only ===")
    if model1 is not None:
        print(model1.summary())
        print("\nOdds Ratios (Signal Only):")
        for param, odds in zip(model1.params.index, odds_ratios1):
            print(f"  {param}: {odds:.3f}")
    else:
        print("Model could not be fitted.")

    if model2 is not None:
        print("\n=== Model 2: State Probabilities Only ===")
        print(model2.summary())
        print("\nOdds Ratios (State Probabilities Only):")
        for param, odds in zip(model2.params.index, odds_ratios2):
            print(f"  {param}: {odds:.3f}")
    else:
        print("\n=== Model 2: State Probabilities Only ===")
        print("Model could not be fitted due to insufficient variation in state variables.")

    if model3 is not None and model3 is not model1:
        print("\n=== Model 3: Signal + State Probabilities ===")
        print(model3.summary())
        print("\nOdds Ratios (Signal + State Probabilities):")
        for param, odds in zip(model3.params.index, odds_ratios3):
            print(f"  {param}: {odds:.3f}")
    else:
        print("\n=== Model 3: Signal + State Probabilities ===")
        if model3 is model1:
            print("Model identical to Model 1 (Signal Only) due to insufficient variation in state variables.")
        else:
            print("Model could not be fitted.")

    if model4 is not None:
        print("\n=== Model 4: Signal + State + Interaction ===")
        print(model4.summary())
        print("\nOdds Ratios (Signal + State + Interaction):")
        for param, odds in zip(model4.params.index, odds_ratios4):
            print(f"  {param}: {odds:.3f}")
    else:
        print("\n=== Model 4: Signal + State + Interaction ===")
        print("Model could not be fitted due to insufficient variation in state variables.")

    # Perform model comparison
    from scipy import stats

    def compare_models(larger_model, smaller_model, label1, label2):
        """Compare nested models using likelihood ratio test"""
        if larger_model is None or smaller_model is None:
            print(f"Cannot compare {label1} vs {label2} - one or both models missing.")
            return None

        # Skip if models are identical
        if larger_model is smaller_model:
            print(f"Cannot compare {label1} vs {label2} - models are identical.")
            return None

        try:
            lr_stat = -2 * (smaller_model.llf - larger_model.llf)
            df_diff = larger_model.df_model - smaller_model.df_model

            if df_diff <= 0:
                print(f"Cannot compare {label1} vs {label2} - models not properly nested.")
                return None

            p_value = stats.chi2.sf(lr_stat, df_diff)

            print(f"\nLikelihood Ratio Test: {label1} vs {label2}")
            print(f"  Chi-squared: {lr_stat:.3f}")
            print(f"  Degrees of freedom: {df_diff}")
            print(f"  p-value: {p_value:.6f}")

            if p_value < 0.05:
                print(f"  Result: {label1} is significantly better than {label2}")
            else:
                print(f"  Result: No significant improvement of {label1} over {label2}")

            return {"lr_stat": lr_stat, "df_diff": df_diff, "p_value": p_value}
        except Exception as e:
            print(f"Error comparing models: {e}")
            return None

    # Compare models only if appropriate
    comp1 = None
    comp2 = None
    comp3 = None

    # Model 3 vs Model 1 (signal + state vs signal only)
    if model3 is not None and model3 is not model1 and model1 is not None:
        comp1 = compare_models(model3, model1, "Signal+State", "Signal Only")

    # Model 3 vs Model 2 (signal + state vs state only)
    if model3 is not None and model2 is not None and model3 is not model2:
        comp2 = compare_models(model3, model2, "Signal+State", "State Only")

    # Model 4 vs Model 3 (full interaction vs signal + state)
    if model4 is not None and model3 is not None and model4 is not model3:
        comp3 = compare_models(model4, model3, "Signal+State+Interaction", "Signal+State")

    # Calculate McFadden's pseudo-R² for each model
    def mcfadden_r2(model):
        """Calculate McFadden's pseudo-R² for a fitted model"""
        if model is None:
            return None
        return 1 - (model.llf / model.llnull)

    r2_model1 = mcfadden_r2(model1)
    r2_model2 = mcfadden_r2(model2)
    r2_model3 = mcfadden_r2(model3)
    r2_model4 = mcfadden_r2(model4)

    print("\nMcFadden's Pseudo-R²:")
    print(f"  Model 1 (Signal Only): {r2_model1:.4f}" if r2_model1 is not None else "  Model 1: Not available")
    print(f"  Model 2 (State Only): {r2_model2:.4f}" if r2_model2 is not None else "  Model 2: Not available")
    print(f"  Model 3 (Signal + State): {r2_model3:.4f}" if r2_model3 is not None else "  Model 3: Not available")
    print(
        f"  Model 4 (Signal + State + Interaction): {r2_model4:.4f}" if r2_model4 is not None else "  Model 4: Not available")

    # Determine best model based on AIC
    # Get AIC for each model (if available)
    aic_values = []
    model_names = []
    models = [model1, model2, model3, model4]
    model_labels = ["Signal Only", "State Only", "Signal + State", "Signal + State + Interaction"]

    for model, label in zip(models, model_labels):
        if model is not None:
            # Skip duplicate models (e.g., if model3 is model1)
            if model not in [m for m, _ in zip(models[:models.index(model)], model_labels[:models.index(model)]) if
                             m is not None]:
                aic_values.append(model.aic)
                model_names.append(label)

    if aic_values:
        best_model_idx = np.argmin(aic_values)
        print(f"\nBest model based on AIC: {model_names[best_model_idx]}")
        print(f"  AIC values: {[f'{aic:.2f}' for aic in aic_values]}")
    else:
        print("\nNo models available for AIC comparison")
        best_model_idx = None

    # Visualize the results only if we have models to visualize
    if model1 is not None:
        plt.figure(figsize=(15, 10))

        # First subplot: Effect of signal on switch probability
        plt.subplot(2, 2, 1)

        # Range of signal values for prediction (use standardized values)
        signal_range = np.linspace(-2.5, 2.5, 100)  # z-scores from -2.5 to 2.5

        # Create prediction data frames for different models
        pred_data_model1 = pd.DataFrame({'signal_z': signal_range})
        pred_data_model1 = sm.add_constant(pred_data_model1)

        # Make predictions for model 1
        pred_model1 = model1.predict(pred_data_model1)

        # Plot predictions for signal-only model
        plt.plot(signal_range, pred_model1, 'b-', linewidth=2.5, label="Signal Only Model")

        # Create prediction data for model 3 (Signal + State) if it exists and has state variables
        if model3 is not None and model3 is not model1:
            # Only show prediction lines if there's actual variation in state variables
            has_state_vars = False

            # Create predictions for different levels of state probabilities
            # Use mean, low (-1SD), and high (+1SD) values for the state probabilities
            scenarios = []

            if state_use_for_model['p_stochastic'] and state_use_for_model['p_biased']:
                # Both states have variation
                has_state_vars = True
                stoch_mean = data['p_stochastic_z'].mean()
                stoch_sd = data['p_stochastic_z'].std()
                biased_mean = data['p_biased_z'].mean()
                biased_sd = data['p_biased_z'].std()

                # Define typical scenarios
                scenarios = [
                    {'name': 'High Stochastic, Low Biased', 'stoch': stoch_mean + stoch_sd,
                     'biased': biased_mean - biased_sd, 'color': 'green'},
                    {'name': 'Low Stochastic, High Biased', 'stoch': stoch_mean - stoch_sd,
                     'biased': biased_mean + biased_sd, 'color': 'red'},
                    {'name': 'Average State Probs', 'stoch': stoch_mean, 'biased': biased_mean, 'color': 'purple'}
                ]
            elif state_use_for_model['p_stochastic']:
                # Only stochastic has variation
                has_state_vars = True
                stoch_mean = data['p_stochastic_z'].mean()
                stoch_sd = data['p_stochastic_z'].std()

                scenarios = [
                    {'name': 'High Stochastic', 'stoch': stoch_mean + stoch_sd,
                     'biased': 0, 'color': 'green'},
                    {'name': 'Low Stochastic', 'stoch': stoch_mean - stoch_sd,
                     'biased': 0, 'color': 'red'},
                    {'name': 'Average Stochastic', 'stoch': stoch_mean,
                     'biased': 0, 'color': 'purple'}
                ]
            elif state_use_for_model['p_biased']:
                # Only biased has variation
                has_state_vars = True
                biased_mean = data['p_biased_z'].mean()
                biased_sd = data['p_biased_z'].std()

                scenarios = [
                    {'name': 'High Biased', 'stoch': 0,
                     'biased': biased_mean + biased_sd, 'color': 'green'},
                    {'name': 'Low Biased', 'stoch': 0,
                     'biased': biased_mean - biased_sd, 'color': 'red'},
                    {'name': 'Average Biased', 'stoch': 0,
                     'biased': biased_mean, 'color': 'purple'}
                ]

            # Plot prediction lines if we have state variables with variation
            if has_state_vars:
                for scenario in scenarios:
                    # Create prediction data with the right variables
                    pred_data = {'signal_z': signal_range}

                    if state_use_for_model['p_stochastic']:
                        pred_data['p_stochastic_z'] = scenario['stoch']

                    if state_use_for_model['p_biased']:
                        pred_data['p_biased_z'] = scenario['biased']

                    # Convert to DataFrame and add constant
                    pred_data = pd.DataFrame(pred_data)
                    pred_data = sm.add_constant(pred_data)

                    # Make predictions using model 3
                    try:
                        pred_probs = model3.predict(pred_data)

                        # Plot prediction line
                        plt.plot(signal_range, pred_probs, color=scenario['color'], linewidth=2.5,
                                 label=scenario['name'])
                    except Exception as e:
                        print(f"Error generating predictions for {scenario['name']}: {e}")

        # Add reference line at 50% probability
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

        # Add labels and title
        plt.xlabel('Signal (z-scored)', fontsize=12)
        plt.ylabel('Probability of Switching', fontsize=12)
        plt.title(f'Predicted Switch Probability by {signal_window} Signal and State', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)

        # Second subplot: Actual switching rates by signal quartile
        plt.subplot(2, 2, 2)

        # Create quartiles of signal
        try:
            data['signal_quartile'] = pd.qcut(data['signal'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

            # Only create stochastic quartiles if there's variation
            if state_use_for_model['p_stochastic'] and data['p_stochastic'].std() > 0:
                try:
                    data['stoch_quartile'] = pd.qcut(data['p_stochastic'],
                                                     q=4,
                                                     labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                                     duplicates='drop')
                except Exception as e:
                    print(f"Could not create stochastic quartiles: {e}")
                    data['stoch_quartile'] = "All data"
            else:
                data['stoch_quartile'] = "All data"

            # Calculate switch rates for each combination of signal quartile and stochastic quartile
            switch_rates = data.groupby(['signal_quartile', 'stoch_quartile']).agg(
                switch_rate=('switch', 'mean'),
                count=('switch', 'count')
            ).reset_index()

            # Convert to percentages
            switch_rates['switch_rate'] = switch_rates['switch_rate'] * 100

            # Plot separate lines for each stochastic probability quartile
            stoch_levels = sorted(data['stoch_quartile'].unique())
            stoch_colors = plt.cm.viridis(np.linspace(0, 1, len(stoch_levels)))

            for i, stoch_q in enumerate(stoch_levels):
                stoch_data = switch_rates[switch_rates['stoch_quartile'] == stoch_q]
                if not stoch_data.empty:
                    plt.plot(stoch_data['signal_quartile'], stoch_data['switch_rate'],
                             'o-', color=stoch_colors[i], linewidth=2.5, label=f"Stochastic Prob: {stoch_q}")

                    # Add count labels
                    for _, row in stoch_data.iterrows():
                        plt.text(row['signal_quartile'], row['switch_rate'] + 2, f"n={row['count']}",
                                 color=stoch_colors[i], ha='center', fontsize=8)
        except Exception as e:
            print(f"Error creating quartile plot: {e}")
            plt.text(0.5, 0.5, "Could not create quartile plot", ha='center', va='center')

        plt.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Signal Quartile', fontsize=12)
        plt.ylabel('Switch Rate (%)', fontsize=12)
        plt.title('Actual Switch Rates by Signal Quartile and State Probability', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 100)

        # Third subplot: Coefficient forest plot (only if we have model 3 with state vars)
        plt.subplot(2, 2, 3)

        if model3 is not None and model3 is not model1:
            try:
                # Get coefficients and confidence intervals from model 3
                coefs = model3.params[1:]  # Skip intercept
                conf_int = model3.conf_int()
                conf_int = conf_int.loc[coefs.index]

                # Calculate odds ratios and confidence intervals for odds ratios
                odds_ratios = np.exp(coefs)
                odds_ratio_ci = np.exp(conf_int)

                # Create forest plot
                y_pos = np.arange(len(coefs))

                plt.errorbar(odds_ratios, y_pos, xerr=[odds_ratios - odds_ratio_ci.iloc[:, 0],
                                                       odds_ratio_ci.iloc[:, 1] - odds_ratios],
                             fmt='o', capsize=5, color='blue', markersize=8)

                # Add vertical line at odds ratio = 1 (no effect)
                plt.axvline(x=1, color='red', linestyle='--', linewidth=1.5)

                # Add labels
                plt.yticks(y_pos, coefs.index)
                plt.xlabel('Odds Ratio (log scale)', fontsize=12)
                plt.title('Odds Ratios with 95% Confidence Intervals', fontsize=14)

                # Use log scale for x-axis to better visualize odds ratios
                plt.xscale('log')
                plt.grid(True, alpha=0.3)

                # Add significance markers
                for i, p_val in enumerate(model3.pvalues[1:]):
                    marker = '*' * sum([p_val < threshold for threshold in [0.05, 0.01, 0.001]])
                    if marker:
                        plt.text(odds_ratios.iloc[i] * 1.1, i, marker, fontsize=14)
            except Exception as e:
                print(f"Error creating forest plot: {e}")
                plt.text(0.5, 0.5, "Could not create forest plot", ha='center', va='center')
        else:
            if model3 is model1:
                plt.text(0.5, 0.5, "Model 3 identical to Model 1 - no state effects to show", ha='center', va='center')
            else:
                plt.text(0.5, 0.5, "Model 3 not available", ha='center', va='center')

        # Fourth subplot: Model comparison
        plt.subplot(2, 2, 4)

        if aic_values:
            try:
                # Plot AIC values
                bar_positions = np.arange(len(aic_values))
                bar_colors = ['blue', 'green', 'orange', 'red'][:len(aic_values)]
                bars = plt.bar(bar_positions, aic_values, color=bar_colors)
                plt.ylabel('AIC (lower is better)', fontsize=12)
                plt.title('Model Comparison', fontsize=14)
                plt.xticks(bar_positions, model_names, rotation=15)
                plt.grid(True, axis='y', alpha=0.3)

                # Highlight best model
                if best_model_idx is not None:
                    bars[best_model_idx].set_color('purple')

                # Add R² values on top of bars
                r2_values = []
                for model_name in model_names:
                    if model_name == "Signal Only":
                        r2_values.append(r2_model1)
                    elif model_name == "State Only":
                        r2_values.append(r2_model2)
                    elif model_name == "Signal + State":
                        r2_values.append(r2_model3)
                    elif model_name == "Signal + State + Interaction":
                        r2_values.append(r2_model4)

                for i, (r2, bar) in enumerate(zip(r2_values, bars)):
                    if r2 is not None:
                        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - 5,
                                 f'R² = {r2:.3f}', ha='center', fontsize=10, color='white', fontweight='bold')
            except Exception as e:
                print(f"Error creating model comparison plot: {e}")
                plt.text(0.5, 0.5, "Could not create model comparison plot", ha='center', va='center')
        else:
            plt.text(0.5, 0.5, "No models available for comparison", ha='center', va='center')

        plt.tight_layout()

        # Save the figure
        save_figure(plt.gcf(), subject_id, "pooled", f"signal_state_regression_{signal_window}_{condition}")

        plt.show()

    # Create a summary of the results
    summary = {
        'subject_id': subject_id,
        'signal_window': signal_window,
        'condition': condition,
        'total_trials': len(data),
        'state_variation': state_use_for_model
    }

    # Add model results to summary (if models exist)
    summary['models'] = {}

    if model1 is not None:
        summary['models']['signal_only'] = {
            'coefficients': model1.params.to_dict(),
            'pvalues': model1.pvalues.to_dict(),
            'odds_ratios': odds_ratios1.to_dict(),
            'aic': model1.aic,
            'r2': r2_model1
        }

    if model2 is not None:
        summary['models']['state_only'] = {
            'coefficients': model2.params.to_dict(),
            'pvalues': model2.pvalues.to_dict(),
            'odds_ratios': odds_ratios2.to_dict(),
            'aic': model2.aic,
            'r2': r2_model2
        }

    if model3 is not None and model3 is not model1:
        summary['models']['signal_and_state'] = {
            'coefficients': model3.params.to_dict(),
            'pvalues': model3.pvalues.to_dict(),
            'odds_ratios': odds_ratios3.to_dict(),
            'aic': model3.aic,
            'r2': r2_model3
        }

    if model4 is not None:
        summary['models']['full_model'] = {
            'coefficients': model4.params.to_dict(),
            'pvalues': model4.pvalues.to_dict(),
            'odds_ratios': odds_ratios4.to_dict(),
            'aic': model4.aic,
            'r2': r2_model4
        }

    # Add model comparisons to summary (if available)
    if comp1 is not None or comp2 is not None or comp3 is not None:
        summary['model_comparisons'] = {}
        if comp1 is not None:
            summary['model_comparisons']['signal_and_state_vs_signal'] = comp1
        if comp2 is not None:
            summary['model_comparisons']['signal_and_state_vs_state'] = comp2
        if comp3 is not None:
            summary['model_comparisons']['full_model_vs_signal_and_state'] = comp3

    # Add best model info if available
    if best_model_idx is not None:
        summary['best_model'] = {
            'name': model_names[best_model_idx],
            'index': best_model_idx + 1
        }

    return summary