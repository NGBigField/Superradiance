if __name__ == "__main__":
    import pathlib, sys 
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from sys import argv

# Import DictWriter class from CSV module
from csv import DictWriter

from scripts.condor.job_plot_system_size import main as job_plot_system_size
from scripts.condor.job_optimize import main as optimize
from scripts.condor.job_movie    import main as movie



NUM_EXPECTED_ARGS = 5


# A main function to parse inputs:
def main():

    ## Check function call:
    assert len(argv)==NUM_EXPECTED_ARGS, f"Expected {NUM_EXPECTED_ARGS} arguments. Got {len(argv)}."

    ## Parse args:
    print(f"The {NUM_EXPECTED_ARGS} arguments are:")

    this_func_name = argv[0]
    print(f"1: this_func_name={this_func_name!r}")

    output_file = argv[1]
    print(f"2: output_file={output_file!r}")

    seed = int(argv[2])
    print(f"3: seed={seed}")

    variation = int(argv[3])
    print(f"4: variation={variation}")

    job_type = argv[4]
    print(f"5: job_type={job_type}")

    ## Call job:        # "movie"\"optimize"
    if job_type=="movie":
        res = movie(variation)
    elif job_type=="optimize":
        res = optimize(variation, seed)
    elif job_type=="plot_system_size":
        res = job_plot_system_size(variation, seed)
    else: 
        raise ValueError(f"Not an expected job_type={job_type!r}")
            
            
    print(f"res={res}")

    ## check output:
    assert isinstance(res, dict), f"result must be of type `dict`! got {type(res)}"

    ## Write result:
    with open( output_file ,'a') as f:
        dict_writer = DictWriter(f, fieldnames=list( res.keys() ) )
        dict_writer.writerow(res)
        f.close()


if __name__ == "__main__":
    main()

