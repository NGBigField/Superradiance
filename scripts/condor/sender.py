import pathlib, sys, os

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# Import DictWriter class from CSV module
from csv import DictWriter

def main(
    num_seeds:int=2, 
    num_variations:int=4,
    job_type="plot_results"  # "movie"\"plot_results"\"optimize"\"plot_system_size"\"emitted_light"
):

    ## Define paths and names:
    sep = os.sep
    this_folder_path = pathlib.Path(__file__).parent.__str__()
    #
    script_fullpath     = this_folder_path+sep+"worker.py"
    results_fullpath    = this_folder_path+sep+"results.csv"
    output_files_prefix = "superradiance-"+job_type
    #
    print(f"script_fullpath={script_fullpath!r}")
    print(f"results_fullpath={results_fullpath!r}")
    print(f"output_files_prefix={output_files_prefix!r}")
    print(f"job_type={job_type!r}")

    ## Define job params:
    job_params = []

    for variation in range(num_variations):
        variation = f"{variation}"

        for resolution in [200, 500, 1000, 2000, 4000]:
            resolution = f"{resolution}"

            for seed in range(num_seeds):
                seed = f"{seed}"

                job_params.append( dict(
                    outfile=results_fullpath,
                    variation=variation,
                    seed=seed,
                    resolution=resolution,
                    job_type=job_type
                ))

    for params in job_params:
        print(params)

    ## Prepare output file:
    fieldnames = ["variation", "seed", "score", "theta", "resolution"]
    with open( results_fullpath ,'a') as f:        
        dict_writer = DictWriter(f, fieldnames=fieldnames)
        dict_writer.writerow({field:field for field in fieldnames})
        f.close()

    ## Call condor:
    import CondorJobSender
    CondorJobSender.send_batch_of_jobs_to_condor(
        script_fullpath,
        output_files_prefix,
        job_params,
        request_cpus='8',
        requestMemory='16gb',
        Arguments='$(outfile) $(seed) $(variation) $(job_type) $(resolution)'
    )

    print("Called condor successfully")



if __name__=="__main__":
    main()