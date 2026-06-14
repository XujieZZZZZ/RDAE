from param import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def visualize_executor_usage(job_dags, file_path, carbon_schedule=None):
    exp_completion_time = int(np.ceil(np.max([
        j.completion_time for j in job_dags])))

    job_durations = \
        [job_dag.completion_time - \
        job_dag.start_time for job_dag in job_dags]

    executor_occupation = np.zeros(exp_completion_time)
    executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                executor_occupation[
                    int(task.start_time) : \
                    int(task.finish_time)] += 1
        num_jobs_in_system[
            int(job_dag.start_time) : \
            int(job_dag.completion_time)] += 1

    executor_usage = \
        np.sum(executor_occupation) / np.sum(executor_limit)

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    # plt.plot(executor_occupation)
    # plt.fill_between(range(len(executor_occupation)), 0,
    #                  executor_occupation)
    plt.plot(moving_average(executor_occupation, 10000))

    plt.ylabel('Number of busy executors')
    plt.title('Executor usage: ' + str(executor_usage) + \
              '\n average completion time: ' + \
              str(np.mean(job_durations)))

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)

    if carbon_schedule is not None:
        # compute the carbon footprint by multiplying the number of busy executors times the carbon intensity
        carbon_usage = 0
        for i, busy in enumerate(executor_occupation): # executor occupation is the number of busy executors at each millisecond of the simulation
            # get carbon intensity by rounding the current time down to the nearest 60000 
            key = i - (i % 60000)
            if key in carbon_schedule.keys():
                carbon_usage += busy * carbon_schedule[key]
            else:
                carbon_usage += busy * carbon_schedule[list(carbon_schedule.keys())[0]]
    print("File name: ", file_path, " Carbon usage: ", carbon_usage)



def visualize_dag_time(job_dags, executors, plot_total_time=None, plot_type='stage'):

    dags_makespan = 0
    all_tasks = []
    # 1. compute each DAG's finish time
    # so that we can visualize it later
    dags_finish_time = []
    dags_duration = []
    for dag in job_dags:
        dag_finish_time = 0
        for node in dag.nodes:
            for task in node.tasks:
                all_tasks.append(task)
                if task.finish_time > dag_finish_time:
                    dag_finish_time = task.finish_time
        dags_finish_time.append(dag_finish_time)
        assert dag_finish_time == dag.completion_time
        dags_duration.append(dag_finish_time - dag.start_time)

    # 2. visualize them in a canvas
    if plot_total_time is None:
        canvas = np.ones([len(executors), int(max(dags_finish_time))]) * args.canvas_base
    else:
        canvas = np.ones([len(executors), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # job_dag -> base

    for job_dag in job_dags:
        bases[job_dag] = base
        base += job_dag.num_nodes

    for task in all_tasks:

        start_time = int(task.start_time)
        finish_time = int(task.finish_time)
        exec_id = task.executor.idx

        if plot_type == 'stage':

            canvas[exec_id, start_time : finish_time] = \
                bases[task.node.job_dag] + task.node.idx

        elif plot_type == 'app':
            canvas[exec_id, start_time : finish_time] = \
                job_dags.index(task.node.job_dag)

    return canvas, dags_finish_time, dags_duration


def visualize_dag_time_save_pdf(
        job_dags, executors, file_path, plot_total_time=None, plot_type='stage'):
    
    canvas, dag_finish_time, dags_duration = \
        visualize_dag_time(job_dags, executors, plot_total_time, plot_type)

    fig = plt.figure()

    # canvas
    plt.imshow(canvas, interpolation='nearest', aspect='auto')
    # plt.colorbar()
    # each dag finish time
    for finish_time in dag_finish_time:
        plt.plot([finish_time, finish_time],
                 [- 0.5, len(executors) - 0.5], 'r')
    plt.title('average DAG completion time: ' + str(np.mean(dags_duration)))
    fig.savefig(file_path)
    plt.close(fig)




def visualize_carbon(carbon_time_list):
    schemes = [entry[0] for entry in carbon_time_list]


    times = [entry[1]/10000 for entry in carbon_time_list]
    carbon_usage = [entry[2] for entry in carbon_time_list]

    x = range(len(schemes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(x, times, width, label="Time (s)", color='cornflowerblue')
    ax1.set_xlabel("Scheduling Algorithm")
    ax1.set_ylabel("Time (s)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks([pos + width / 2 for pos in x])
    ax1.set_xticklabels(schemes)

    ax2 = ax1.twinx()
    ax2.bar([pos + width for pos in x], carbon_usage, width, label="Carbon Usage", color='mediumseagreen')
    ax2.set_ylabel("Carbon Usage", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    print("generating plot of carbon usage over time....")
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles + handles2, labels + labels2, loc='upper left')

    plt.title("Comparison of Time and Carbon Usage by Scheduling Algorithm")
    fig.tight_layout()
    fig.savefig("./simulator/results/barplot_carbon_usage.png")
    plt.close(fig)


def visualize_carbon_usage(job_dags, file_path, carbon_schedule):
    """
    Visualizes the carbon usage over time for the given job DAGs, similar to visualize_executor_usage.
    It calculates the carbon usage dynamically and produces a bar plot comparing time and carbon usage.
    
    Args:
    - job_dags: List of job DAGs executed during the simulation.
    - file_path: Path to save the plot.
    - carbon_schedule: Dictionary of carbon intensities keyed by time intervals.
    """
    exp_completion_time = int(np.ceil(np.max([
        j.completion_time for j in job_dags])))

    executor_occupation = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                executor_occupation[
                    int(task.start_time):int(task.finish_time)] += 1

    carbon_usage = 0
    for i, busy in enumerate(executor_occupation):
        key = i - (i % 60000)
        if key in carbon_schedule.keys():
            carbon_usage += busy * carbon_schedule[key]
        else:
            carbon_usage += busy * carbon_schedule[list(carbon_schedule.keys())[0]]

    total_time = np.sum(executor_occupation) / len(executor_occupation)

    schemes = ["Total Time", "Carbon Usage"]
    values = [total_time, carbon_usage]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(schemes))
    ax1.bar(x, values, color=['cornflowerblue', 'mediumseagreen'])

    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Values", color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(schemes)
    plt.title("Carbon Usage and Execution Time")

    print(f"File saved: {file_path}, Carbon Usage: {carbon_usage}")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)

def visualize_carbon_usage_aggregated(scheme_data, file_path):
    """
    Generates a combined bar plot for all schemes with total time and total carbon usage.
    
    Args:
    - scheme_data: List of tuples in the format (scheme_name, total_time, total_carbon_usage).
    - file_path: Path to save the plot.
    """
    # Extract scheme names, times, and carbon usage
    schemes = [entry[0] for entry in scheme_data]
    times = [entry[1]/60000 for entry in scheme_data]
    carbon_usage = np.array([entry[2] for entry in scheme_data]) / scheme_data[0][2]
    job_durations = [entry[3] for entry in scheme_data]

    # readable scheme names
    scheme_names = {
        "spark_fifo": "Spark FIFO",
        "dynamic_partition": "Weighted Fair",
        "decima": "Decima",
        "green_hadoop": "GreenHadoop",
        "cap_fifo": "CAP on\n Spark FIFO",
        "fifo": "FIFO",
        "cap_partition": "CAP on\n Weighted Fair",
        "cap_decima": "CAP on\n Decima",
        "pcaps": "PCAPS",
        "newalgo": "newalgo"
    }
    schemes_friendly = [scheme_names[scheme] for scheme in schemes]

    x = range(len(schemes))
    width = 0.35

    # Create the bar plot
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    ax1.bar(x, times, width, label="Time (s)", color='cornflowerblue')
    ax1.set_xlabel("Scheduling Algorithms")
    ax1.set_ylabel("Time (hours)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks([pos + width / 2 for pos in x])
    ax1.set_xticklabels(schemes_friendly)

    ax2 = ax1.twinx()
    ax2.bar([pos + width for pos in x], carbon_usage, width, label="Carbon Usage", color='mediumseagreen')
    ax2.set_ylabel("Carbon Usage (normalized to FIFO)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    plt.title("End-to-End Completion Time and Carbon Footprint by Scheduler")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)

    # once everything is done, print the results for carbon and time to the console    
    # normalize everything to fifo (first scheme) so that we can compare the results
    print("Results (normalized to FIFO) for each scheduler:")
    for i in range(len(schemes)):
        name = scheme_names[schemes[i]].replace('\n', '')
        print(f"{name + ':': <25} Carbon: {carbon_usage[i] / carbon_usage[0]:.4f}, End-to-end Time: {times[i] / times[0]:.4f}, Avg. Job Duration: {job_durations[i] / job_durations[0]:.4f}")
    

