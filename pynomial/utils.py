def _check_arguments(x, n, conf):

    if (conf < 0) or (conf > 1):
        raise ValueError(f"conf should be between 0 and 1 but got {conf}")

    if (not isinstance(x, int)) and (not x.is_integer()):
        raise ValueError(
            f"x should be an int datatype or convertable to int but got {x}"
        )

    if (not isinstance(n, int)) and (not n.is_integer()):
        raise ValueError(
            f"n should be an int datatype of convertable to int but got {n}"
        )

    if (n < x) or (x < 0) or (n < 0):
        raise ValueError("n and x should be positive integers with 0<= x <=n")
