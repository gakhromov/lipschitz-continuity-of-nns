"""List of functions to convert a convolution operation to a linear matrix 
multiplication."""
import numpy as np
import scipy
from scipy.sparse import csr_array, lil_array


def get_conv2d_matrix(kernel: np.array,
                      padding: int,
                      stride: int,
                      img_size: int) -> csr_array:
    """Compute the matrix K that represents the convolution operation.
    One can perform convolution by doing
    ```python
    >>> K = get_conv2d_matrix(kernel, padding, stride, img_size).toarray()
    >>> f_conv = lambda x: (K @ x.flatten()).reshape(output_channel, 
                                                     img_size, 
                                                     img_size)
    # f_conv(x) === torch.nn.Conv2d(input_channel, 
    #                               output_channel, 
    #                               kernel_size, 
    #                               stride=stride, 
    #                               padding=padding, 
    #                               bias=False)(x)
    ```

    Parameters
    ----------
    kernel
        Convolution kernel of size (output_channel, input_channel, kernel_size, 
        kernel_size).
    padding
        Convolution padding.
    stride
        Convolution stride.
    img_size
        The size of the square image (aka height or width).

    Returns
    -------
        Matrix K that represents the linear operation of the convolution.

    Limitations
    -----------
        This function only works for square images, kernels, paddings and 
        strides (meaning that the amount of padding and stride step in both axes
        is the same).
    """
    kernel_size = kernel.shape[-1]
    inp_ch = kernel.shape[1]
    out_ch = kernel.shape[0]

    out_img_size = (img_size + 2 * padding - kernel_size) // stride + 1

    # placeholder for the final K matrix
    # LIL is efficent for modification
    K = lil_array((out_img_size**2 * out_ch, img_size**2 * inp_ch))

    toeplitz_matrices = []
    for out_ch_i in range(out_ch):
        for inp_ch_i in range(inp_ch):
            toeplitz_matrices_curr = get_toeplitz_matrices(
                kernel[out_ch_i, inp_ch_i], padding, stride, img_size
            )
            toeplitz_matrices += toeplitz_matrices_curr

    # get doubly blocked matrix indices for just the 2d image
    # -1 indicates matrix with zeros
    row_size = img_size + padding
    row = np.hstack([np.arange(kernel_size), [-1] * (row_size - kernel_size)])

    col_size = img_size + 2 * padding - kernel_size + 1
    col = np.hstack([[0], [-1] * (col_size - 1)])

    index_matrix = scipy.linalg.toeplitz(c=col, r=row)
    # if there is non-zero padding, remove the first pad columns
    # if there is non-one stride, take rows with step stride
    index_matrix = index_matrix[::stride, padding:]

    # horizontally pad the input channels
    concat_inp = []
    for inp_ch_i in range(inp_ch):
        t = index_matrix.copy()
        t[np.where(t >= 0)] = t[np.where(t >= 0)] + kernel_size * inp_ch_i
        concat_inp.append(t)

    index_matrix = np.concatenate(concat_inp, axis=1)

    # vertically pad the output channels
    concat_out = []
    for out_ch_i in range(out_ch):
        t = index_matrix.copy()
        t[np.where(t >= 0)] = t[np.where(t >= 0)] + \
            kernel_size * inp_ch * out_ch_i
        concat_out.append(t)

    index_matrix = np.concatenate(concat_out, axis=0)

    toep_rows = toeplitz_matrices[0].shape[0]
    toep_cols = toeplitz_matrices[0].shape[1]

    # get the doubly blocked Toeplitz matrix
    for i in range(index_matrix.shape[0]):
        for j in range(index_matrix.shape[1]):
            # do not do anything with zeros
            if index_matrix[i, j] == -1:
                continue
            # fill the kernel with toeplitz matrices' values otherwise
            K[
                i * toep_rows: (i + 1) * toep_rows,
                j * toep_cols: (j + 1) * toep_cols
            ] = toeplitz_matrices[index_matrix[i, j]]

    # convert to CSR since we usually compute the norm of this matrix afterwards
    return csr_array(K)


def get_toeplitz_matrices(
    k_curr_channel: np.array, padding: int, stride: int, img_size: int
) -> list[np.array]:
    """This function makes small Toeplitz matrices for each row of the image
    for the particular input and output channel.
    Used by the `get_conv2d_matrix` function.

    Parameters
    ----------
    k_curr_channel
        Matrix, representing the entries of the kernel tensor for the particular
        input and output channel.
    padding
        Convolution padding.
    stride
        Convolution stride.
    img_size
        The size of the square image (aka height or width).

    Returns
    -------
        An array of Toeplitz matrices for each row of a particular
        input and output channel of the kernel wrt. to input image.
    """
    toeplitz_matrices = []
    kernel_size = k_curr_channel.shape[0]

    # create a matrix for each row of the kernel
    for i in range(kernel_size):
        row_size = img_size + padding
        row = np.hstack([k_curr_channel[i], [0] * (row_size - kernel_size)])

        col_size = img_size + 2 * padding - kernel_size + 1
        col = np.hstack([[k_curr_channel[i][0]], [0] * (col_size - 1)])

        toeplitz = scipy.linalg.toeplitz(c=col, r=row)
        # if there is non-zero padding, remove the first pad columns
        # if there is non-one stride, take rows with step stride
        toeplitz = toeplitz[::stride, padding:]
        toeplitz_matrices.append(toeplitz)

    return toeplitz_matrices
