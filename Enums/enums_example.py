from enum import Enum, auto



class Direction(Enum):
    up    = auto()
    down  = auto()
    left  = auto()
    right = auto()


def f(a: int, d: Direction = Direction.up) -> int:
    if d == Direction.up:
        out = a + 1
    elif d == Direction.down:
        out = a + 2
    elif d == Direction.left:
        out = a + 3
    elif d == Direction.right:
        out = a + 4
    return out


def main():
    a = f(10, d=Direction.left)
    print(f"a={a}")


if __name__ == "__main__":
    main()