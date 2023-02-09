if __name__ == "__main__":
    import pathlib, sys, os 
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from scripts.main_gkp_square import learn_sx2_pulses as square_gkp
from scripts.main_gkp_hex import learn_sx2_pulses as hex_gkp


def main(variation:int=0, seed:int=0):
    
    if variation==0:
        result = square_gkp()
    elif variation==1:
        result = hex_gkp()
    else:
        raise ValueError(f"Not a supported variation: {variation}")

    print("Result:")
    print(result)

    return dict(
        seed = seed,
        variation = variation,
        score = result.score,
        theta = result.operation_params
    )


if __name__ == "__main__":
    main()

