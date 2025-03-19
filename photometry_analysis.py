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
CODE_VERSION = "1.0.4"  # Increment this when making analysis changes --> will force recomputation of all data
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


def load_behavior_data(subject_id, session_date):
    """Load behavioral data from Parquet and filter it by subject and session date. Ignore any session where "ignore" is not 0
    Calculates correction of sound_time_stamps for the lick_time_stamps, so epochs are aligned by moment of first lick"""

    try:
        # Load the parquet file with pyarrow
        df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

        df['date'] = df['date'].astype(str)  # Ensure date is a string
        session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
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


def process_session(subject_id, session_date, force_recompute=False, use_global_cache=True):
    """Process a single session for a given subject"""

    cache_key = f"{subject_id}/{session_date}"
    if use_global_cache and cache_key in _SESSION_CACHE and not force_recompute:
        return _SESSION_CACHE[cache_key]
    
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

        behavior_data = load_behavior_data(subject_id, session_date)
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

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('ΔF/F', fontsize=12)
    ax1.set_title(f'Photometry Response: {analysis_result["subject_id"]} - {analysis_result["session_date"]}',
                  fontsize=14)
    ax1.set_xlim([-pre_cue_time, post_cue_time])
    ax1.legend(loc='upper right')

    # Add statistics text box
    stats_text = (f"Trials: {len(analysis_result['non_m_trials'])} (excluding missed trials)\n"
                  f"Peak: {np.max(analysis_result['trial_average']):.4f}\n"
                  f"Baseline: {np.mean(analysis_result['trial_average'][:pre_cue_samples]):.4f}")
    ax1.text(-pre_cue_time + 0.2, np.max(analysis_result['trial_average']) * 0.9, stats_text,
             bbox=dict(facecolor='white', alpha=0.7))

    # Second subplot: Heatmap (if enabled)
    if show_heatmap:
        ax2 = axes[1]
        trial_data = analysis_result['plotting_data']
        im = ax2.imshow(trial_data,
                        aspect='auto',
                        extent=[-pre_cue_time, post_cue_time, 0, len(trial_data)],
                        cmap='viridis',
                        interpolation='nearest')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Trial Number', fontsize=12)
        ax2.set_title('Trial-by-Trial Heatmap', fontsize=14)
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


def analyze_pooled_data(subject_id, win_loss=False, force_recompute=False, fig=None):
    """Analyze and visualize pooled data for a subject without loading from saved files"""
    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(12, 7))

    # Check for saved pooled results
    if not force_recompute:
        saved_results = check_saved_pooled_results(subject_id, win_loss)
        if saved_results is not None:
            # Recreate the figure from saved results
            plt.figure(figsize=(12, 7))

            if win_loss:
                # Rewarded data
                if saved_results.get('rewarded_avg') is not None:
                    rewarded_avg = saved_results['rewarded_avg']
                    rewarded_sem = saved_results['rewarded_sem']
                    plt.fill_between(saved_results['time_axis'],
                                    rewarded_avg - rewarded_sem,
                                    rewarded_avg + rewarded_sem,
                                    color='lightgreen', alpha=0.4, label='Rewarded ± SEM')
                    plt.plot(saved_results['time_axis'], rewarded_avg,
                            color='green', linewidth=2.5, label='Rewarded Avg')

                # Unrewarded data
                if saved_results.get('unrewarded_avg') is not None:
                    unrewarded_avg = saved_results['unrewarded_avg']
                    unrewarded_sem = saved_results['unrewarded_sem']
                    plt.fill_between(saved_results['time_axis'],
                                    unrewarded_avg - unrewarded_sem,
                                    unrewarded_avg + unrewarded_sem,
                                    color='lightsalmon', alpha=0.4, label='Unrewarded ± SEM')
                    plt.plot(saved_results['time_axis'], unrewarded_avg,
                            color='darkorange', linewidth=2.5, label='Unrewarded Avg')
            else:
                pooled_sem = saved_results['pooled_sem']
                plt.fill_between(saved_results['time_axis'],
                                 saved_results['pooled_average'] - pooled_sem,  
                                 saved_results['pooled_average'] + pooled_sem,  
                                 color='lightgreen', alpha=0.4,
                                 label='Mean ± SEM')  

                # Plot session averages
            num_sessions = len(saved_results['session_dates'])
            blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))

            for idx, session_avg in enumerate(saved_results.get('session_averages', [])):
                plt.plot(saved_results['time_axis'], session_avg,
                         alpha=0.6, linewidth=1, linestyle='-',
                         color=blue_colors[idx],
                         label=f"Session {saved_results['session_dates'][idx]}" if idx < 5 else "_nolegend_")

            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('ΔF/F', fontsize=12)
            plt.title(f'Pooled Photometry Response: {subject_id} ({len(saved_results["session_dates"])} sessions)',
                      fontsize=14)
            plt.xlim([-pre_cue_time, post_cue_time])

            # Recreate the stats text box
            stats_text = (f"Total Sessions: {len(saved_results['session_dates'])}\n"
                          f"Total Trials: {saved_results['total_trials']}\n"
                          f"Peak: {np.max(saved_results['pooled_average']):.4f}\n"
                          f"Baseline: {np.mean(saved_results['pooled_average'][:pre_cue_samples]):.4f}")
            plt.text(-pre_cue_time + 0.2, np.max(saved_results['pooled_average']) * 1.2, stats_text,
                     bbox=dict(facecolor='white', alpha=0.7))

            # Limit legend items if too many sessions
            if len(saved_results['session_dates']) > 5:
                handles, labels = plt.gca().get_legend_handles_labels()
                limited_handles = handles[:8]
                limited_labels = labels[:8]
                limited_labels.append(f"+ {len(saved_results['session_dates']) - 5} more sessions")
                plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=10)
            else:
                plt.legend(loc='upper right', fontsize=10)

            plt.tight_layout()

            # Save figure
            save_figure(plt.gcf(), subject_id, "pooled",
                        f"pooled_results{'_winloss' if win_loss else ''}")

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

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    num_sessions = len(sessions)
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, num_sessions))  # Create blue gradient

    for idx, session_date in enumerate(sessions):
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            result = process_session(subject_id, session_date)
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

    if win_loss:
        rewarded_data = []
        unrewarded_data = []

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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(f'Pooled Photometry Response: {subject_id} ({len(all_sessions)} sessions)', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])

    # Limit legend items if too many sessions
    if len(all_sessions) > 5:
        handles, labels = plt.gca().get_legend_handles_labels()
        limited_handles = handles[:8]
        limited_labels = labels[:8]
        limited_labels.append(f"+ {len(all_sessions) - 5} more sessions")
        plt.legend(limited_handles, limited_labels, loc='upper right', fontsize=10)
    else:
        plt.legend(loc='upper right', fontsize=10)

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
    save_figure(plt.gcf(), subject_id, "pooled", f"pooled_results{'_winloss' if win_loss else ''}")

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


# Function aliases with clear names for Jupyter notebook cells
def analyze_specific_session(subject_id, session_date, show_heatmap=False, win_loss=False):
    """Analyze and visualize a specific session"""
    print(f"Analyzing session {subject_id}/{session_date}...")
    analysis_result = process_session(subject_id, session_date)
    if analysis_result:
        return plot_session_results(analysis_result, show_heatmap=show_heatmap, win_loss=win_loss)
    return None


def pooled_results(subject_id, win_loss=False):
    """Analyze and visualize pooled results for a subject"""
    print(f"Analyzing pooled results for subject {subject_id}...")
    return analyze_pooled_data(subject_id, win_loss=win_loss)


def all_results(win_loss=False, force_recompute=False):
    """Analyze and visualize results for all subjects"""
    print("Analyzing all subjects...")
    return analyze_all_subjects(win_loss=win_loss, force_recompute=force_recompute)


def analyze_reward_rate_quartiles(subject_id, session_date=None, win_loss=False):
    """
    Analyze photometry signals binned by reward rate quartiles for a single session or pooled across sessions
    """
    all_plotting_data = []
    all_reward_rates = []
    all_reward_outcomes = []
    time_axis = None
    plot_title = ''

    if session_date is None:
        # Get all sessions for pooled analysis
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                         if os.path.isdir(os.path.join(subject_path, d)) and
                         os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
        
        # Process each session separately to maintain session-specific reward rate context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Calculate reward rates for this session
            behavior_data = session_result['behavioral_data']
            rewards = np.array(behavior_data['reward'])
            window_size = max(int(len(rewards) * 0.1), 1)
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
        session_result = process_session(subject_id, session_date)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        behavior_data = session_result['behavioral_data']
        rewards = np.array(behavior_data['reward'])
        window_size = max(int(len(rewards) * 0.1), 1)
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
        plot_title = f'Photometry by Reward Rate Quartiles: {subject_id} - {session_date}'

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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    
    # Add text with quartile averages at the bottom of the plot
    quartile_text = "Average reward rates: " + ", ".join([f"Q{q+1}: {avg:.4f}" for q, avg in enumerate(quartile_averages)])
    plt.figtext(0.5, 0.01, quartile_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom

    # Save the figure
    fig_name = f"reward_rate_quartiles{'_pooled' if session_date is None else ''}"
    save_figure(plt.gcf(), subject_id, session_date or "pooled", 
               f"{fig_name}{'_winloss' if win_loss else ''}")

    plt.show()

    return {
        'quartile_bins': quartile_bins,
        'reward_rates': reward_rates
    }


def analyze_comp_confidence_quartiles(subject_id, session_date=None, win_loss=False):
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
        sessions = sorted([d for d in os.listdir(subject_path)
                           if os.path.isdir(os.path.join(subject_path, d)) and
                           os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

        # Process each session separately to maintain session-specific confidence context
        for session_date in sessions:
            session_result = process_session(subject_id, session_date)
            if not session_result:
                continue

            if len(session_result['non_m_trials']) < 100:
                print(f"Skipping {subject_id}/{session_date}, less than 100 valid trials ({len(session_result['non_m_trials'])}).")
                continue

            # Get behavioral data for this session
            behavior_data = session_result['behavioral_data']

            # Load the parquet file to get min_pvalue data
            try:
                df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
                df['date'] = df['date'].astype(str)  # Ensure date is a string
                session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]

                if session_data.empty:
                    print(f"No p-value data found for {subject_id} on {session_date}")
                    continue

                # Extract p-values and calculate confidence
                p_values = session_data['min_pvalue'].values
                min_p_value = 1e-12
                p_values = np.maximum(p_values, min_p_value)
                confidence = -np.log10(p_values)

                # Calculate moving average confidence with window size 15
                window_size = 15
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
        session_result = process_session(subject_id, session_date)
        if not session_result:
            print(f"Could not process session {subject_id}/{session_date}")
            return None

        # Load the parquet file to get min_pvalue data
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            session_data = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]

            if session_data.empty:
                print(f"No p-value data found for {subject_id} on {session_date}")
                return None

            # Extract p-values and calculate confidence
            p_values = session_data['min_pvalue'].values
            min_p_value = 1e-12
            p_values = np.maximum(p_values, min_p_value)
            confidence = -np.log10(p_values)

            # Calculate moving average confidence with window size 15
            window_size = 15
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
            plot_title = f'Photometry by Computer Confidence Quartiles: {subject_id} - {session_date}'

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

    return {
        'quartile_bins': quartile_bins,
        'confidence_rates': confidence_rates
    }

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

def plot_per_session_win_loss(subject_id):
    """
    Plot win/loss traces for each session of a subject with choice history
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
        
    Returns:
    --------
    dict: Dictionary of session win-loss analyses
    """
    # Find all sessions for the subject
    subject_path = os.path.join(base_dir, subject_id)
    sessions = sorted([d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and
                os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

    # Find max peak for consistent y-axis scaling
    max_peak = float('-inf')
    min_peak = float('inf')
    session_analyses = {}
    valid_sessions = []

    # First pass: find valid sessions and determine y-axis scaling
    for session_date in sessions:
        session_result = process_session(subject_id, session_date)
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

        # Get reward outcomes and photometry data
        reward_outcomes = session_result["reward_outcomes"][non_m_indices]
        session_plots = session_result['plotting_data']

        # Separate rewarded and unrewarded trials
        rewarded_trials = session_plots[reward_outcomes == 1]
        unrewarded_trials = session_plots[reward_outcomes == 0]

        # Compute average rewarded and unrewarded signals
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(rewarded_avg + rewarded_sem))
            min_peak = min(min_peak, np.min(rewarded_avg - rewarded_sem))
            
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            max_peak = max(max_peak, np.max(unrewarded_avg + unrewarded_sem))
            min_peak = min(min_peak, np.min(unrewarded_avg - unrewarded_sem))

    # Add 20% padding to y-axis limits
    y_range = max_peak - min_peak
    y_max = max_peak + 0.1 * y_range
    y_min = min_peak - 0.1 * y_range

    # Calculate number of sessions and rows/columns for subplots
    n_sessions = len(valid_sessions)  # Use count of valid sessions
    if n_sessions == 0:
        print(f"No valid sessions found for {subject_id} (all have < 100 trials)")
        return {}
        
    n_cols = 3  # Always 3 columns
    n_rows = (n_sessions + n_cols - 1) // n_cols  # Ceiling division for number of rows needed
    
    # Create figure with proper size
    fig = plt.figure(figsize=(18, 5*n_rows))  # Wider figure, height based on number of rows
    
    # Create GridSpec to control the layout - 2 rows per session (photometry + choice)
    gs = plt.GridSpec(n_rows*2, n_cols, height_ratios=[2, 1] * n_rows)  # Repeat [2,1] pattern for each row
    
    # Process only valid sessions
    for i, session_date in enumerate(valid_sessions):
        row = i // n_cols  # Row index (integer division)
        col = i % n_cols   # Column index
        
        session_result = process_session(subject_id, session_date)
        if not session_result:
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

        # Create subplot for photometry data (top row for this session)
        ax_photo = fig.add_subplot(gs[row*2, col])
        
        # Photometry data - rewarded
        if len(rewarded_trials) > 0:
            rewarded_avg = np.mean(rewarded_trials, axis=0)
            rewarded_sem = calculate_sem(rewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       rewarded_avg - rewarded_sem,
                       rewarded_avg + rewarded_sem,
                       color='lightgreen', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], rewarded_avg,
                 color='green', linewidth=2,
                 label=f'Win (n={len(rewarded_trials)})')
        
        # Photometry data - unrewarded
        if len(unrewarded_trials) > 0:
            unrewarded_avg = np.mean(unrewarded_trials, axis=0)
            unrewarded_sem = calculate_sem(unrewarded_trials, axis=0)
            ax_photo.fill_between(session_result['time_axis'],
                       unrewarded_avg - unrewarded_sem,
                       unrewarded_avg + unrewarded_sem,
                       color='lightsalmon', alpha=0.3)
            ax_photo.plot(session_result['time_axis'], unrewarded_avg,
                 color='darkorange', linewidth=2,
                 label=f'Loss (n={len(unrewarded_trials)})')
        
        # Add vertical line at cue onset
        ax_photo.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax_photo.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Set y-axis limit consistent across all photometry subplots
        ax_photo.set_ylim([y_min, y_max])
        
        # Labels and formatting for photometry subplot
        ax_photo.set_xlabel('Time (s)')
        ax_photo.set_ylabel('ΔF/F')
        ax_photo.set_title(f'Session {session_date} (n={len(rewarded_trials) + len(unrewarded_trials)} trials)')
        ax_photo.legend(loc='upper right')
        
        # Store analysis results
        session_analyses[session_date] = {
            'rewarded_avg': rewarded_avg if len(rewarded_trials) > 0 else None,
            'unrewarded_avg': unrewarded_avg if len(unrewarded_trials) > 0 else None,
            'rewarded_n': len(rewarded_trials),
            'unrewarded_n': len(unrewarded_trials),
            'time_axis': session_result['time_axis']
        }
        
        # Create subplot for choice history (bottom row)
        ax_choice = fig.add_subplot(gs[row*2 + 1, col])
        
        # Extract choices and rewards
        choices = session_result['behavioral_data']['choice']
        rewards = session_result['behavioral_data']['reward']
        
        # Plot choice history
        for j, choice in enumerate(choices):
            if choice == 'L':
                ax_choice.plot([j + 1, j + 1], [0, 1], 'r-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, 1, 'ro', markersize=8, fillstyle='none')
            elif choice == 'R':
                ax_choice.plot([j + 1, j + 1], [0, -1], 'b-', linewidth=1.5)
                if rewards[j] == 1:
                    ax_choice.plot(j + 1, -1, 'bo', markersize=8, fillstyle='none')
        
        # Add the middle line
        ax_choice.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Set the y-axis limits and labels
        ax_choice.set_ylim(-1.5, 1.5)
        ax_choice.set_yticks([-1, 0, 1])
        ax_choice.set_yticklabels(['Right', '', 'Left'])
        
        # Set the x-axis and title
        ax_choice.set_xlabel('Trial Number')
        ax_choice.set_title('Choice History')
        ax_choice.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, subject_id, "all_sessions", "per_session_win_loss_with_choices")
    
    plt.show()
    
    return session_analyses


def analyze_session_win_loss_difference_gap(subject_id, session_date=None, comp_conf=False, df=None, sem=True):
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
    # If no specific session provided, get all sessions for the subject
    if session_date is None:
        subject_path = os.path.join(base_dir, subject_id)
        sessions = sorted([d for d in os.listdir(subject_path)
                           if os.path.isdir(os.path.join(subject_path, d)) and
                           os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])
    else:
        sessions = [session_date]

    # Load dataframe if needed for confidence calculations and not provided
    if comp_conf and df is None:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting

    # Store results for each session
    session_differences = {}
    session_confidences = {}

    # Process each session
    for idx, session_date in enumerate(sessions):
        # Process the session
        session_result = process_session(subject_id, session_date)
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
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
                
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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Rewarded - Unrewarded ΔF/F', fontsize=12)
    
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


def analyze_previous_outcome_effect(subject_id):
    """
    Analyze photometry signals based on previous and current trial outcomes.
    
    Parameters:
    -----------
    subject_id : str
        The identifier for the subject
        
    Returns:
    --------
    dict: Analysis results
    """
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Process each session and collect results
    all_sessions = []
    all_plotting_data = []
    all_prev_rewards = []
    all_curr_rewards = []
    session_dates = []

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d)) and
                      os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    # Process sessions in chronological order
    for session_date in sessions:
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            result = process_session(subject_id, session_date)
            if result:
                if len(result['non_m_trials']) < 100: 
                    print(f"Skipping {subject_id}/{session_date} due to low number of trials")
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
                
                # Store data
                all_plotting_data.append(session_plots)
                all_curr_rewards.append(curr_rewards)
                all_prev_rewards.append(prev_rewards)
    
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
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Define colors and labels
    colors = {
        'prev_win_curr_win': 'darkgreen',
        'prev_win_curr_loss': 'indianred', 
        'prev_loss_curr_win': 'mediumseagreen',
        'prev_loss_curr_loss': 'firebrick'
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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)
    plt.title(f'Photometry by Previous Trial Outcome: {subject_id} ({len(all_sessions)} sessions)', fontsize=14)
    plt.xlim([-pre_cue_time, post_cue_time])
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), subject_id, "pooled", "previous_outcome_effect")
    
    plt.show()
    
    # Return analysis results
    return {
        'subject_id': subject_id,
        'session_dates': session_dates,
        'time_axis': time_axis,
        'condition_data': condition_data
    }


def analyze_win_stay_lose_switch(subject_id, session_date=None, df=None):
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

    if df is None:
        try: 
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            return None

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
        # Get all sessions from parquet file for this subject
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            sessions = df[df['subjid'] == subject_id]['date'].unique()
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            return None
    else:
        sessions = [session_date]

    # Process each session
    for sess in sessions:
        print(f"Analyzing WSLS for {subject_id}/{sess}...")
        
        # Get behavior data directly from parquet file
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)
            session_df = df[(df['subjid'] == subject_id) & (df['date'] == sess) & (df['ignore'] == 0)]
            
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
        plt.title(f'Win-Stay, Lose-Switch Analysis Across Sessions: {subject_id}')
        plt.xticks(session_numbers)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add data table below the plot
        table_data = []
        for i, s in enumerate(session_data):
            table_data.append([
                f"Session {i+1}",
                f"{s['win_stay_count']}/{s['win_trials']} ({s['win_stay_pct']:.1f}%)",
                f"{s['lose_switch_count']}/{s['lose_trials']} ({s['lose_switch_pct']:.1f}%)",
                f"{s['total_wsls_count']}/{s['total_trials']} ({s['wsls_pct']:.1f}%)"
            ])
        
        plt.table(cellText=table_data,
                  colLabels=['Session', 'Win-Stay', 'Lose-Switch', 'Overall WSLS'],
                  loc='bottom',
                  bbox=[0, -0.65, 1, 0.5])
        
        plt.subplots_adjust(bottom=0.4)  # Make room for the table
        
        # Save figure
        save_figure(plt.gcf(), subject_id, "all_sessions", "wsls_across_sessions")
        plt.show()
    
    # Store session data in results
    wsls_results['session_data'] = session_data
    return wsls_results


def analyze_loss_streaks_before_win(subject_id, skipped_missed=True, only_1_5=False):
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
        If False, plot all categories

    Returns:
    --------
    dict: Analysis results for different loss streak lengths
    """
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None

    # Store data for each loss streak category
    streak_data = {
        '1_loss': [],  # T0 loss, T-1 no loss
        '2_loss': [],  # T0 & T-1 loss, T-2 no loss
        '3_loss': [],  # T0, T-1, T-2 loss, T-3 no loss
        '4_loss': [],  # T0, T-1, T-2, T-3 loss, T-4 no loss
        '5plus_loss': []  # T0 loss preceded by 4+ losses
    }

    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d)) and
                       os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])

    time_axis = None  # Will be set from the first valid session

    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date)
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
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('ΔF/F', fontsize=12)

    missed_text = "excluding" if skipped_missed else "including"
    plot_cat_text = "1_and_5" if only_1_5 else "all_cats"
    plt.title(f'Photometry Signal by Loss Streak Length Before Win: {subject_id} ({missed_text} missed trials)',
              fontsize=14)
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

def analyze_session_win_loss_difference_heatmap(subject_id, comp_conf=False):
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
    sessions = sorted([d for d in os.listdir(subject_path)
                       if os.path.isdir(os.path.join(subject_path, d)) and
                       os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

    # Store results for each session
    session_differences = []
    session_dates = []
    time_axis = None
    peak_differences = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date)
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
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
                
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
    ax_heatmap.set_xlabel('Time (s)', fontsize=12)
    ax_heatmap.set_ylabel('Session', fontsize=12)
    ax_heatmap.set_title(f'Win-Loss Signal Difference Across Sessions: {subject_id} (sorted by {sort_type})', fontsize=14)

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

    # Plot peak differences across sessions (bottom)
    ax_peaks = fig.add_subplot(gs[1])

    # Create x-axis values (session numbers from 1 to n)
    session_numbers = np.arange(1, len(session_dates) + 1)

    # Create blue gradient for bars
    blue_colors = plt.cm.Blues(np.linspace(0.3, 1, len(session_dates)))

    # Plot peak differences
    ax_peaks.bar(session_numbers, peak_differences, color=blue_colors, alpha=0.7)

    # Add trend line
    if len(peak_differences) > 1:
        z = np.polyfit(session_numbers, peak_differences, 1)
        p = np.poly1d(z)
        ax_peaks.plot(session_numbers, p(session_numbers), 'r--', linewidth=2,
                      label=f'Trend: {z[0]:.4f}')

        # Add correlation coefficient
        corr, p_val = scipy.stats.pearsonr(session_numbers, peak_differences)
        ax_peaks.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}',
                      transform=ax_peaks.transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and formatting
    ax_peaks.set_xlabel('Session Number', fontsize=12)
    ax_peaks.set_ylabel('Peak Win-Loss Difference', fontsize=12)
    ax_peaks.set_title('Peak Difference Magnitude by Session', fontsize=14)
    ax_peaks.set_xticks(session_numbers)
    ax_peaks.set_xticklabels(session_numbers)  # Use session numbers
    ax_peaks.grid(True, axis='y', alpha=0.3)

    if len(peak_differences) > 1:
        ax_peaks.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

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


def analyze_session_average_heatmap(subject_id, comp_conf=False):
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
    sessions = sorted([d for d in os.listdir(subject_path)
                       if os.path.isdir(os.path.join(subject_path, d)) and
                       os.path.exists(os.path.join(subject_path, d, "deltaff.npy"))])

    # Store results for each session
    session_averages = []
    session_dates = []
    time_axis = None
    peak_averages = []
    session_confidences = {}

    # Load parquet data for confidence calculations if needed
    if comp_conf:
        try:
            df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
            df['date'] = df['date'].astype(str)  # Ensure date is a string
            print(f"Loaded parquet data for confidence calculations")
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            comp_conf = False  # Fallback to chronological sorting

    # Process each session
    for session_date in sessions:
        # Process the session
        session_result = process_session(subject_id, session_date)
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
                session_df = df[(df['subjid'] == subject_id) & (df['date'] == session_date) & (df["ignore"] == 0)]
                
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
    ax_heatmap.set_title(f'Average Photometry Signal Across Sessions: {subject_id} (sorted by {sort_type})', fontsize=14)

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

    # Plot peak averages across sessions (bottom)
    ax_peaks = fig.add_subplot(gs[1])

    # Create x-axis values (session numbers from 1 to n)
    session_numbers = np.arange(1, len(session_dates) + 1)

    # Create gradient colors for bars based on peak values
    norm = plt.Normalize(min(peak_averages), max(peak_averages))
    colors = plt.cm.viridis(norm(peak_averages))

    # Plot peak averages
    ax_peaks.bar(session_numbers, peak_averages, color=colors, alpha=0.7)

    # Add trend line
    if len(peak_averages) > 1:
        z = np.polyfit(session_numbers, peak_averages, 1)
        p = np.poly1d(z)
        ax_peaks.plot(session_numbers, p(session_numbers), 'r--', linewidth=2,
                      label=f'Trend: {z[0]:.4f}')

        # Add correlation coefficient
        corr, p_val = scipy.stats.pearsonr(session_numbers, peak_averages)
        ax_peaks.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}',
                      transform=ax_peaks.transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and formatting
    ax_peaks.set_xlabel('Session Number', fontsize=12)
    ax_peaks.set_ylabel('Peak Signal Amplitude', fontsize=12)
    ax_peaks.set_title('Peak Signal Magnitude by Session', fontsize=14)
    ax_peaks.set_xticks(session_numbers)
    ax_peaks.set_xticklabels(session_numbers)  # Use session numbers
    ax_peaks.grid(True, axis='y', alpha=0.3)

    if len(peak_averages) > 1:
        ax_peaks.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    sort_suffix = "by_comp_conf" if comp_conf else "chronological"
    save_figure(fig, subject_id, "all_sessions", f"average_signal_heatmap_{sort_suffix}")

    plt.show()

    # Return analysis results
    return {
        'subject_id': subject_id,
        'session_dates': session_dates,
        'time_axis': time_axis,
        'session_averages': session_averages,
        'peak_averages': peak_averages,
        'session_confidences': session_confidences if comp_conf else None
    }


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

def analyze_signal_quartiles_by_outcome(subject_id, signal_window='pre_cue', condition='loss', plot_verification=True):
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
    
    # Find all session directories for this subject
    subject_dir = os.path.join(base_dir, subject_id)
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return None
    
    # Determine analysis mode based on time window
    is_pre_cue_analysis = signal_window == 'pre_cue'
    
    # Store trial data and corresponding behavior
    all_trials_data = []         # Photometry data for all selected trials
    all_trial_signals = []       # Average signal in window for selected trials
    all_choice_switches = []     # Boolean: True if switch in choice
    all_next_reward = []         # Boolean: True if next trial rewarded 
    time_axis = None             # Will be set from the first valid session
    
    # Set the reward value we're looking for based on condition
    target_reward = 1 if condition == 'win' else 0
    
    # Sort sessions chronologically
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d)) and
                       os.path.exists(os.path.join(subject_dir, d, "deltaff.npy"))])
    
    # Process each session
    for session_date in sessions:
        print(f"Processing {subject_id}/{session_date}...")
        session_result = process_session(subject_id, session_date)
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
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[curr_trial_idx])  # Current trial's reward outcome
                
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
                
                # Store the data
                all_trials_data.append(curr_photometry)
                all_trial_signals.append(window_signal)
                all_choice_switches.append(choice_switched)
                all_next_reward.append(non_miss_rewards[next_trial_idx])  # Next trial's reward outcome
    
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
        else:
            quartile_switch_rates.append(0)
            quartile_trial_counts.append(0)
            quartile_reward_rates.append(0)
            
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
        
    # Create verification plot if requested
    if plot_verification:
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Plot average photometry traces by quartile
        ax1 = fig.add_subplot(gs[0])
        colors = ['blue', 'green', 'orange', 'red']  # Colors for quartiles
        
        # Plot each quartile's average trace
        for quartile in range(4):
            quartile_mask = quartile_labels == quartile
            if np.sum(quartile_mask) > 0:
                quartile_data = all_trials_data[quartile_mask]
                quartile_avg = np.mean(quartile_data, axis=0)
                quartile_sem = calculate_sem(quartile_data, axis=0)
                
                ax1.fill_between(time_axis,
                               quartile_avg - quartile_sem,
                               quartile_avg + quartile_sem,
                               color=colors[quartile], alpha=0.3)
                ax1.plot(time_axis, quartile_avg,
                       color=colors[quartile], linewidth=2,
                       label=f'Quartile {quartile+1} (n={quartile_trial_counts[quartile]})')
                       
        # Highlight the time window used for sorting
        ax1.axvspan(window_start, window_end, color='gray', alpha=0.3, label='Sorting Window')
        
        # Add reference lines
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Lick Timing')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('ΔF/F', fontsize=12)
        
        if is_pre_cue_analysis:
            ax1.set_title(f'Trials After {outcome_label} Sorted by Pre-Cue Signal: {subject_id}', fontsize=14)
        else:
            ax1.set_title(f'{outcome_label} Trials Sorted by {signal_window} Signal: {subject_id}', fontsize=14)
            
        ax1.legend(loc='upper right')
        ax1.set_xlim([-pre_cue_time, post_cue_time])
        
        # Plot switch rates by quartile
        ax2 = fig.add_subplot(gs[1])
        bars = ax2.bar(range(1, 5), quartile_switch_rates, color=colors, alpha=0.7)
        
        # Add trial counts as text on bars
        for bar, count in zip(bars, quartile_trial_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'n={count}', ha='center', va='bottom', fontsize=10)
                   
        ax2.set_xlabel('Signal Quartile', fontsize=12)
        ax2.set_ylabel('Switch Rate (%)', fontsize=12)
        
        if is_pre_cue_analysis:
            ax2.set_title(f'% Trials Where Choice Switched From Previous {outcome_label} Trial', fontsize=14)
        else:
            ax2.set_title('% Trials Where Choice Switched on Next Trial', fontsize=14)
            
        ax2.set_ylim(0, 100)
        ax2.set_xticks(range(1, 5))
        ax2.set_xticklabels([f'Q{i+1}' for i in range(4)])
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot reward rates by quartile
        ax3 = fig.add_subplot(gs[2])
        bars = ax3.bar(range(1, 5), quartile_reward_rates, color=colors, alpha=0.7)
        
        # Add trial counts as text on bars
        for bar, count in zip(bars, quartile_trial_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'n={count}', ha='center', va='bottom', fontsize=10)
                   
        ax3.set_xlabel('Signal Quartile', fontsize=12)
        ax3.set_ylabel('Reward Rate (%)', fontsize=12)
        
        if is_pre_cue_analysis:
            ax3.set_title('% Trials That Were Rewarded', fontsize=14)
        else:
            ax3.set_title('% Next Trials That Were Rewarded', fontsize=14)
            
        ax3.set_ylim(0, 100)
        ax3.set_xticks(range(1, 5))
        ax3.set_xticklabels([f'Q{i+1}' for i in range(4)])
        ax3.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        condition_str = condition  # 'win' or 'loss'
        mode_suffix = f"after_{condition_str}" if is_pre_cue_analysis else f"{condition_str}_trials"
        save_figure(fig, subject_id, "pooled", f"{mode_suffix}_{signal_window}_quartiles")
        
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

def analyze_switch_probabilities(subject_id, session_date=None):
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
        # Get all sessions for this subject
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
        session_result = process_session(subject_id, sess)
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


def analyze_switch_probability_quartiles(subject_id, session_date=None, win_loss=False):
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
            session_result = process_session(subject_id, session_date)
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
        session_result = process_session(subject_id, session_date)
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
        plot_title = f'Photometry by Switch Probability Quartiles: {subject_id} - {session_date}'

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


def analyze_normalized_quartile_effects(subject_id):
    """
    Analyze whether the quartile effect on switch probability is stronger than expected
    given the baseline differences in win-stay/lose-switch behavior.
    """
    print(f"Analyzing normalized quartile effects for {subject_id}...")

    # First, get baseline win/loss switch probabilities
    switch_probs = analyze_switch_probabilities(subject_id)

    if not switch_probs or 'after_win' not in switch_probs or 'after_loss' not in switch_probs:
        print("Could not obtain baseline switch probabilities")
        return None

    baseline_win_switch = switch_probs['after_win']['rate']
    baseline_loss_switch = switch_probs['after_loss']['rate']

    print(f"\nBaseline switch probabilities:")
    print(f"After win: {baseline_win_switch:.2f}%")
    print(f"After loss: {baseline_loss_switch:.2f}%")

    # Analyze pre-cue quartiles for win and loss trials
    loss_results = analyze_signal_quartiles_by_outcome(subject_id, signal_window='pre_cue',
                                                       condition='loss', plot_verification=False)
    win_results = analyze_signal_quartiles_by_outcome(subject_id, signal_window='pre_cue',
                                                      condition='win', plot_verification=False)

    if not loss_results or not win_results:
        print("Could not obtain quartile analysis results")
        return None

    # Debugging: Print keys to help identify the correct one
    print("Available keys in loss_results:", list(loss_results.keys()))
    print("Available keys in win_results:", list(win_results.keys()))

    # Create normalized effects (how much each quartile deviates from baseline)
    loss_normalized = []
    win_normalized = []

    for i in range(4):
        # For loss trials: (quartile_switch_rate - baseline_loss_switch) / baseline_loss_switch
        if baseline_loss_switch > 0:
            loss_normalized.append((loss_results['quartile_switch_rates'][i] - baseline_loss_switch) /
                                   baseline_loss_switch * 100)
        else:
            loss_normalized.append(0)

        # For win trials: (quartile_switch_rate - baseline_win_switch) / baseline_win_switch
        if baseline_win_switch > 0:
            win_normalized.append((win_results['quartile_switch_rates'][i] - baseline_win_switch) /
                                  baseline_win_switch * 100)
        else:
            win_normalized.append(0)

    # Calculate the slope of the effect across quartiles
    loss_slope = (loss_normalized[3] - loss_normalized[0]) / 3
    win_slope = (win_normalized[3] - win_normalized[0]) / 3

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Actual switch rates by quartile
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 5), loss_results['quartile_switch_rates'], 'o-', color='red',
             label=f'Loss trials (baseline: {baseline_loss_switch:.1f}%)')
    plt.plot(range(1, 5), win_results['quartile_switch_rates'], 'o-', color='green',
             label=f'Win trials (baseline: {baseline_win_switch:.1f}%)')

    # Add counts to points
    for i, (loss_rate, loss_count, win_rate, win_count) in enumerate(zip(
            loss_results['quartile_switch_rates'], loss_results['quartile_trial_counts'],
            win_results['quartile_switch_rates'], win_results['quartile_trial_counts'])):
        plt.text(i + 1, loss_rate + 2, f"n={loss_count}", color='red', ha='center', fontsize=8)
        plt.text(i + 1, win_rate - 4, f"n={win_count}", color='green', ha='center', fontsize=8)

    plt.xlabel('Signal Quartile')
    plt.ylabel('Switch Rate (%)')
    plt.title('Actual Switch Rates by Pre-Cue Signal Quartile')
    plt.xticks(range(1, 5))
    plt.ylim(0, max(max(loss_results['quartile_switch_rates']),
                    max(win_results['quartile_switch_rates'])) + 10)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Normalized effects (percentage change from baseline)
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 5), loss_normalized, 'o-', color='red',
             label=f'Loss trials (slope: {loss_slope:.2f}%/quartile)')
    plt.plot(range(1, 5), win_normalized, 'o-', color='green',
             label=f'Win trials (slope: {win_slope:.2f}%/quartile)')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    plt.xlabel('Signal Quartile')
    plt.ylabel('Normalized Effect (%)')
    plt.title('Normalized Effect: Change from Baseline Switch Probability')
    plt.xticks(range(1, 5))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_figure(plt.gcf(), subject_id, "pooled", "normalized_quartile_effects")
    plt.show()

    # Statistical analysis: bootstrap comparison of slopes
    print("\nBootstrap analysis of slope differences:")
    n_bootstrap = 1000
    slope_diffs = []

    # Check if trial-level data is available in all_trials_data
    if 'all_trials_data' in loss_results and 'all_trials_data' in win_results:
        print("Using trial-level data from all_trials_data")

        # Extract trial-level data - structure depends on what's in all_trials_data
        # Let's inspect what's in all_trials_data
        if loss_results['all_trials_data'] is not None and len(loss_results['all_trials_data']) > 0:
            print("all_trials_data keys for a sample trial:",
                  list(loss_results['all_trials_data'][0].keys()) if isinstance(loss_results['all_trials_data'][0],
                                                                                dict) else "Not a dictionary")

        # Assuming all_trials_data contains trial dictionaries with 'signal' and 'switch' keys
        # You might need to adjust these keys based on actual data structure
        try:
            # Extract signals and switches from all_trials_data
            loss_signals = np.array([trial.get('signal', 0) for trial in loss_results['all_trials_data']])
            loss_switches = np.array([trial.get('switch', 0) for trial in loss_results['all_trials_data']])

            win_signals = np.array([trial.get('signal', 0) for trial in win_results['all_trials_data']])
            win_switches = np.array([trial.get('switch', 0) for trial in win_results['all_trials_data']])

            print(f"Extracted {len(loss_signals)} loss trials and {len(win_signals)} win trials")

            # Alternative method: recreate the data by using quartiles and sampling switch rates
            # This is a fallback method if we can't directly extract trial data
        except (AttributeError, KeyError, TypeError) as e:
            print(f"Error extracting trial data: {e}")
            print("Falling back to simulated bootstrap based on quartile rates")

            # Method 2: Simulate trial data from quartile statistics
            loss_signals = []
            loss_switches = []
            win_signals = []
            win_switches = []

            # For each quartile
            for q in range(4):
                # Get number of trials
                loss_n = loss_results['quartile_trial_counts'][q]
                win_n = win_results['quartile_trial_counts'][q]

                # Get switch rate (as proportion, not percentage)
                loss_switch_rate = loss_results['quartile_switch_rates'][q] / 100
                win_switch_rate = win_results['quartile_switch_rates'][q] / 100

                # Create simulated trials
                # Signals within quartile (just use quartile number as signal value)
                loss_signals.extend([q] * loss_n)
                win_signals.extend([q] * win_n)

                # Switch decisions (1 for switch, 0 for stay)
                loss_switches.extend(np.random.binomial(1, loss_switch_rate, size=loss_n))
                win_switches.extend(np.random.binomial(1, win_switch_rate, size=win_n))

            loss_signals = np.array(loss_signals)
            loss_switches = np.array(loss_switches)
            win_signals = np.array(win_signals)
            win_switches = np.array(win_switches)

            print(f"Created simulated data: {len(loss_signals)} loss trials and {len(win_signals)} win trials")

    else:
        print("No trial-level data available, simulating bootstrap from quartile statistics")
        # Same simulation code as in the except block above
        loss_signals = []
        loss_switches = []
        win_signals = []
        win_switches = []

        # For each quartile
        for q in range(4):
            # Get number of trials
            loss_n = loss_results['quartile_trial_counts'][q]
            win_n = win_results['quartile_trial_counts'][q]

            # Get switch rate (as proportion, not percentage)
            loss_switch_rate = loss_results['quartile_switch_rates'][q] / 100
            win_switch_rate = win_results['quartile_switch_rates'][q] / 100

            # Create simulated trials
            # Signals within quartile (just use quartile index + random variation as signal)
            quartile_signal = q + 0.5  # Center of quartile
            loss_signals.extend([quartile_signal] * loss_n)
            win_signals.extend([quartile_signal] * win_n)

            # Switch decisions (1 for switch, 0 for stay)
            loss_switches.extend(np.random.binomial(1, loss_switch_rate, size=loss_n))
            win_switches.extend(np.random.binomial(1, win_switch_rate, size=win_n))

        loss_signals = np.array(loss_signals)
        loss_switches = np.array(loss_switches)
        win_signals = np.array(win_signals)
        win_switches = np.array(win_switches)

        print(f"Created simulated data: {len(loss_signals)} loss trials and {len(win_signals)} win trials")

    # Setup for bootstrap
    loss_trials = {
        'signals': loss_signals,
        'switches': loss_switches
    }

    win_trials = {
        'signals': win_signals,
        'switches': win_switches
    }

    # Bootstrap resampling
    for _ in range(n_bootstrap):
        # Resample loss trials
        loss_idx = np.random.choice(len(loss_trials['signals']), len(loss_trials['signals']), replace=True)
        loss_resampled_signals = loss_trials['signals'][loss_idx]
        loss_resampled_switches = loss_trials['switches'][loss_idx]

        # Resample win trials
        win_idx = np.random.choice(len(win_trials['signals']), len(win_trials['signals']), replace=True)
        win_resampled_signals = win_trials['signals'][win_idx]
        win_resampled_switches = win_trials['switches'][win_idx]

        # Calculate quartiles for resampled data
        loss_quartiles = pd.qcut(loss_resampled_signals, 4, labels=False)
        win_quartiles = pd.qcut(win_resampled_signals, 4, labels=False)

        # Calculate switch rates per quartile
        loss_rates = []
        win_rates = []

        for q in range(4):
            loss_q_mask = loss_quartiles == q
            win_q_mask = win_quartiles == q

            if np.sum(loss_q_mask) > 0:
                loss_q_rate = np.mean(loss_resampled_switches[loss_q_mask]) * 100
                loss_rates.append(loss_q_rate)
            else:
                loss_rates.append(0)

            if np.sum(win_q_mask) > 0:
                win_q_rate = np.mean(win_resampled_switches[win_q_mask]) * 100
                win_rates.append(win_q_rate)
            else:
                win_rates.append(0)

        # Normalize by baseline
        if baseline_loss_switch > 0 and baseline_win_switch > 0:
            loss_norm = [(r - baseline_loss_switch) / baseline_loss_switch * 100 for r in loss_rates]
            win_norm = [(r - baseline_win_switch) / baseline_win_switch * 100 for r in win_rates]

            # Calculate slopes
            loss_bs_slope = (loss_norm[3] - loss_norm[0]) / 3
            win_bs_slope = (win_norm[3] - win_norm[0]) / 3

            # Store difference in slopes
            slope_diffs.append(loss_bs_slope - win_bs_slope)

    # Calculate confidence interval and p-value
    slope_diffs = np.array(slope_diffs)
    mean_diff = np.mean(slope_diffs)
    ci_low = np.percentile(slope_diffs, 2.5)
    ci_high = np.percentile(slope_diffs, 97.5)

    # P-value from bootstrap distribution
    p_value = np.mean(slope_diffs <= 0) if mean_diff > 0 else np.mean(slope_diffs >= 0)
    p_value = min(p_value, 1 - p_value) * 2  # Two-tailed

    print("\nStatistical comparison of slopes:")
    slope_diff = loss_slope - win_slope

    print(f"Loss trials slope: {loss_slope:.2f}%/quartile")
    print(f"Win trials slope: {win_slope:.2f}%/quartile")
    print(f"Difference in slopes: {loss_slope - win_slope:.2f}%/quartile")

    # Print bootstrap results
    print("\nBootstrap analysis results:")
    print(f"Mean difference from bootstrap: {mean_diff:.2f}%/quartile")
    print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Bootstrap p-value: {p_value:.4f}")

    # Statistical interpretation
    if p_value < 0.05:
        if mean_diff > 0:
            print(
                "Conclusion: The effect of pre-cue signal on switching is significantly stronger for loss trials (p < 0.05)")
        else:
            print(
                "Conclusion: The effect of pre-cue signal on switching is significantly stronger for win trials (p < 0.05)")
    else:
        print("Conclusion: No significant difference in the normalized effect strength between win and loss trials")

    # Return results with bootstrap stats
    return {
        'subject_id': subject_id,
        'baseline_win_switch': baseline_win_switch,
        'baseline_loss_switch': baseline_loss_switch,
        'loss_normalized': loss_normalized,
        'win_normalized': win_normalized,
        'loss_slope': loss_slope,
        'win_slope': win_slope,
        'slope_difference': loss_slope - win_slope,
        'bootstrap_mean_diff': mean_diff,
        'bootstrap_ci_low': ci_low,
        'bootstrap_ci_high': ci_high,
        'bootstrap_p_value': p_value
    }