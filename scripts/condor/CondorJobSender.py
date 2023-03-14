import htcondor
from htcondor import dags
import networkx as nx
import yaml
import os, stat, shutil

"""
Most useful scripts are those starting with the word 'send'
"""
project_dir = os.path.dirname(__file__)
file = open(project_dir + '/configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()
directories = docs['directories']
parameters = docs['parameters']

script_dir = directories['working_dir']
log_dir = directories['log_dir']
error_dir = directories['std_error_dir']
output_dir = directories['std_output_dir']
exec_dir = directories['condor_executables_dir']
python_path = directories['python_path_remote']
julia_path = directories['julia_path_remote']


def send_job_to_condor(py_script_path, job_name, **kargs):
    """
    :param py_script_path: The path of the script to be sent
    :param job_name: some name for the job. Will also be used for output/error files
    :param kargs: All arguments the condor system needs. Amongst them: log, output, error, request_cpus,
                    requestMemory (in mb), universe and Arguments.
                    All names correspond to parameters of the same name in the condor shell scripts.
                    Arguments has a format of a string of argument by the order the script takes them.
    :return:
    """
    if 'Arguments' in kargs.keys():
        num_arguments = len(kargs['Arguments'].split(" "))
    else:
        num_arguments = 0
    executable_path = create_executable_from_py_script(py_script_path, job_name, num_arguments)

    submit_format = create_job_submit_format(executable_path, job_name, **kargs)
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        submit_format.queue(txn)
    return


def send_batch_of_jobs_to_condor(py_script_path, job_name, batch_parameters, **kargs):
    """
    :param py_script_path:
    :param job_name:
    :param batch_parameters: This comes in a format of a list of dictionaries, each holding all parameters a run needs.
            For example: [{'x':'1', 'y':'2'}]. This will be used with an 'Arguments' containing $(x) and $(y)
    :param kargs:
    :return:
    """
    if 'Arguments' in kargs.keys():
        num_arguments = len(kargs['Arguments'].split(" "))
    else:
        num_arguments = 0
    executable_path = create_executable_from_py_script(py_script_path, job_name, num_arguments)

    submit_format = create_job_submit_format(executable_path, job_name, **kargs)
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        submit_format.queue_with_itemdata(txn, 1, iter(batch_parameters))
    return


def create_job_submit_format(executalePath, job_name, **kargs):
    job_submit_format = htcondor.Submit()

    job_submit_format['Executable'] = executalePath

    if 'log' in kargs.keys():
        job_submit_format['log'] = kargs['log']
    else:
        job_submit_format['log'] = log_dir
    job_submit_format['log'] = job_submit_format['log'] + job_name + '-$(ProcID).log'

    if 'output' in kargs.keys():
        job_submit_format['output'] = kargs['output']
    else:
        job_submit_format['output'] = output_dir
    job_submit_format['output'] = job_submit_format['output'] + job_name + '-$(ProcID).out'

    if 'error' in kargs.keys():
        job_submit_format['error'] = kargs['error']
    else:
        job_submit_format['error'] = error_dir
    job_submit_format['error'] = job_submit_format['error'] + job_name + '-$(ProcID).error'

    if 'request_cpus' in kargs.keys():
        job_submit_format['request_cpus'] = kargs['request_cpus']

    if 'requestMemory' in kargs.keys():
        job_submit_format['requestMemory'] = kargs['requestMemory']

    # Arguments must be in string form
    if 'Arguments' in kargs.keys():
        job_submit_format['Arguments'] = kargs['Arguments']

    job_submit_format['universe'] = parameters['universe']

    return job_submit_format


def create_executable_from_py_script(py_script_path, job_name, num_arguments):
    if py_script_path.split('.')[1] == 'py':
        language_path = python_path
    else:
        language_path = julia_path
    executable_path = exec_dir + job_name + '.sh'
    shell_script = '#!/bin/bash\n'
    shell_script = shell_script + 'source /Local/ph_keselman/julia-1.6.1/bin/setup.sh\n'
    shell_script = shell_script + 'chmod u+x ' + py_script_path + '\n'
    shell_script = shell_script + language_path + ' ' + py_script_path + ' '
    for i in range(num_arguments):
        shell_script = shell_script + '${' + str(i + 1) + '} '

    file = open(executable_path, 'w')
    file.write(shell_script)
    file.close()
    os.chmod(executable_path, stat.S_IRWXU)
    return executable_path


def create_job_submit_format_from_python_script(py_script_path, job_name, **kargs):
    if 'Arguments' in kargs.keys():
        num_arguments = len(kargs['Arguments'].split(" "))
    else:
        num_arguments = 0
    executable_path = create_executable_from_py_script(py_script_path, job_name, num_arguments)

    submit_format = create_job_submit_format(executable_path, job_name, **kargs)

    return submit_format


def create_dag_file(dag_graph, dag_dir_name, information_dict):
    """
    :param dag_graph: a DAG networkx graph representing the dependencies between the different jobs,
                        where a job is specified by 'job_name'
    :param dag_dir_name: Directory for the dag. Will be overwritten.
    :param information_dict: a dictionary of dictionaries: has a key for each 'job_name'.
            in information_dict['job_name'] there are keys for
                the python script path (py_script_path)
                the batch parameters (batch_parameters). Will be set to [] by default
                'kargs_dict' is a dictionary holding all parameters for running a job as specified in send_job scripts.
    :return:
    """
    nodes = list(nx.topological_sort(dag_graph))
    # layers = []
    dag = dags.DAG()
    for job_name in nodes:
        job_submit = create_job_submit_format_from_python_script(information_dict[job_name]['py_script_path'], job_name,
                                                                 **information_dict[job_name]['kargs_dict'])
        if 'batch_parameters' not in information_dict[job_name].keys():
            information_dict[job_name]['batch_parameters'] = [{}]
        layer = dag.layer(
            name=job_name,
            submit_description=job_submit,
            vars=information_dict[job_name]['batch_parameters']
        )
        # layers.append(layer)
        parents = list(dag_graph.predecessors(job_name))
        if parents:
            for parent in parents:
                layer.add_parents(dag.glob(parent))

    print(dag.describe())
    if not os.path.exists(exec_dir + dag_dir_name):
        os.mkdir(exec_dir + dag_dir_name)
    shutil.rmtree(exec_dir + dag_dir_name, ignore_errors=True)
    dag_file = dags.write_dag(dag, exec_dir + dag_dir_name)
    return dag_file


def send_dag_job(dag_graph, dag_dir_name, information_dict):
    """
        :param dag_graph: a DAG networkx graph representing the dependencies between the different jobs,
                            where a job is specified by 'job_name'
                            for example:
                            graph = nx.Digraph()
                            graph.add_edges_from([('job1','job2'),('job2','job3')])
        :param dag_dir_name: Directory for the dag. Will be overwritten.
        :param information_dict: a dictionary of dictionaries: has a key for each 'job_name'.
                in information_dict['job_name'] there are keys for
                    the python script path (py_script_path)
                    the batch_parameters (batch_parameters). Will be set to [] by default
                    'kargs_dict' is a dictionary holding all parameters for running a job as specified in send_job scripts.
        :return:
        """
    dag_file = create_dag_file(dag_graph, dag_dir_name, information_dict)
    dag_submit = htcondor.Submit.from_dag(str(dag_file), {'force': 1})
    os.chdir(exec_dir + dag_dir_name)
    schedd = htcondor.Schedd()

    with schedd.transaction() as txn:
        dag_submit.queue(txn)
    return
