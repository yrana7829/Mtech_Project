def get_qrange(num_bits):

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    return qmin, qmax
