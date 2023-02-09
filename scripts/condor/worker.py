if __name__ == "__main__":
    import pathlib, sys, os 
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from sys import argv

# Import DictWriter class from CSV module
from csv import DictWriter

from job import main as func


# A main function to parse inputs:
def main():

    num_expected_args = 4
    assert len(argv)==num_expected_args, f"Expected {num_expected_args} arguments. Got {len(argv)}."

    print("The arguments are:")

    this_func_name = argv[0]
    print(f"this_func_name={this_func_name}")

    output_file = argv[1]
    print(f"output_file={output_file}")

    seed = int(argv[2])
    print(f"seed={seed}")

    variation = int(argv[3])
    print(f"variation={variation}")

    res = func(variation, seed)
    print(f"res={res}")

    assert isinstance(res, dict), f"result must be of type `dict`! got {type(res)}"

    with open( output_file ,'a') as f:
        dict_writer = DictWriter(f, fieldnames=list( res.keys() ) )
        dict_writer.writerow(res)
        f.close()


if __name__ == "__main__":
    main()

