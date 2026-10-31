using System;
using System.Collections.Generic;
using System.Text;

namespace NetInfer;

/// <summary>
/// Tensor is a N dimensionnal matrix generalisation
/// Each dimension have a length define in Stride.
/// Ex:
/// Shape = [2, 3, 4] means the Tensor is 3D and Width=2, Height=3, Depth=4
/// Shape = [] means the Tensor is a scalar
/// Stride is the number of element to skip to go to the next element in the dimension.
/// Ex for a 3D Tensor with Shape = [2, 3, 4], Stride = [3 * 4, 4, 1]
/// 
/// Tensor can be itself a slice of another Tensor:
/// * in this case the Offset is the index of the first element of the slice in the original Tensor.
/// * Shape and Stride can also have less dimensions and be smaller than the original Tensor.
/// </summary>
internal class Tensor<T>
{
    int[] Shape;
    int[] Stride;
    T[] Data;
    int Offset;
}
