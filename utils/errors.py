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
    print(get_traceback(e))

def get_traceback(e) -> str:
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)