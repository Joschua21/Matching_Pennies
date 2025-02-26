from random import choice

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import glob
import warnings
import pandas as pd
from pytz.exceptions import NonExistentTimeError

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
CODE_VERSION = "1.0.3"  # Increment this when making analysis changes

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
    """Load behavioral data from Parquet and filter it by subject and session date."""
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


def process_session(subject_id, session_date, force_recompute=False):
    """Process a single session for a given subject"""

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

    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset')
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
            for i, session_avg in enumerate(saved_results.get('session_averages', [])):
                plt.plot(saved_results['time_axis'], session_avg,
                         alpha=0.4, linewidth=1, linestyle='-',
                         label=f"Session {saved_results['session_dates'][i]}" if i < 5 else "_nolegend_")

            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset')
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

    for session_date in os.listdir(subject_dir):
        session_path = os.path.join(subject_dir, session_date)
        if os.path.isdir(session_path) and os.path.exists(os.path.join(session_path, "deltaff.npy")):
            print(f"Processing {subject_id}/{session_date}...")
            result = process_session(subject_id, session_date)
            if result:
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

    # Plot individual session averages as thin lines
    for i, session in enumerate(all_sessions):
        plt.plot(time_axis, session['trial_average'],
                 alpha=0.4, linewidth=1, linestyle='-',
                 label=f"Session {session['session_date']}" if i < 5 else "_nolegend_")

        # Add a vertical line at the cue onset (time=0)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset')
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