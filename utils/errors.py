# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
# For error handling:
import traceback

# ==================================================================================== #
#|                            Define Useful Error Types                               |#
# ==================================================================================== #

class ProtectedPropertyError(Exception): ...
class QuantumTheoryError(Exception): ...



# ==================================================================================== #
#|                               Declared Functions                                   |#
# ==================================================================================== #
def print_traceback(e: Exception) -> None:
    s = get_traceback(e)
    s = strings.add_color(s, strings.PrintColors.RED)
    print(s)

def get_traceback(e: Exception) -> str:
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)