import pathlib, sys, os

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# Import DictWriter class from CSV module
from csv import DictWriter

def main(num_runs:int=1, num_variations:int=4):

    ## Define paths and names:
    sep = os.sep
    this_folder_path = pathlib.Path(__file__).parent.__str__()
    #
    script_fullpath     = this_folder_path+sep+"worker.py"
    results_fullpath    = this_folder_path+sep+"results.csv"
    output_files_prefix = 'superradiance'
    #
    print(f"script_fullpath={script_fullpath}")
    print(f"results_fullpath={results_fullpath}")
    print(f"output_files_prefix={output_files_prefix}")

    ## Define job params:
    job_params = []
    for seed in range(num_runs):
        seed = f"{seed}"

        for variation in range(num_variations):
            variation = f"{variation}"

            job_params.append( dict(
                outfile=results_fullpath,
                variation=variation,
                seed=seed,
            ))

    for params in job_params:
        print(params)

    ## Prepare output file:
    fieldnames = ["variation", "seed", "score", "theta"]
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
        requestMemory='5gb',
        Arguments='$(outfile) $(seed) $(variation)'
    )

    print("Called condor successfully")



if __name__=="__main__":
    main()