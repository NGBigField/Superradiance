import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from scripts.optimize.gkp_square import main as square_gkp
from scripts.optimize.gkp_hex    import main as hex_gkp
from scripts.optimize.cat4_thin  import main as cat4
from scripts.optimize.cat2       import main as cat2



def main(variation:int=2, seed:int=0, num_total_attempts:int=1):
    
    if variation==0:
        result = square_gkp(num_total_attempts=num_total_attempts)
        name = "square_gkp"
    elif variation==1:
        result = hex_gkp(num_total_attempts=num_total_attempts)
        name = "hex_gkp"
    elif variation==2:
        result = cat4(num_total_attempts=num_total_attempts)
        name = "cat4"
    elif variation==3:
        result = cat2(num_total_attempts=num_total_attempts)
        name = "cat2"
    else:
        raise ValueError(f"Not a supported variation: {variation}")

    print("Result:")
    print(result)

    return dict(
        variation = name,
        seed = seed,
        score = result.score,
        theta = result.operation_params
    )


if __name__ == "__main__":
    main()

