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
internal record Tensor
{
    public required ulong[] Shape { get; init; }
    public required ulong[] Stride { get; init; }
    public required byte[] Data { get; init; }
    public required ulong Offset { get; init; }
    public required ulong ValueSize { get; init; }
    public required TensorType Type { get; init; }

    public static ulong[] StrideFromShape(ulong[] shape, ulong valueSize)
    {
        var strides = new ulong[shape.Length];
        var stride = valueSize;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            var dim = shape[i];
            strides[i] = stride;
            stride *= dim;
        }
        return strides;
    }

    /// <summary>
    /// Project a Tensor on a dimension by fixing the index of this dimension.
    /// </summary>
    public Tensor OrthogonalProject(int dimension, ulong index)
    {
        if (dimension < 0 || dimension >= Shape.Length)
            throw new ArgumentOutOfRangeException(nameof(dimension));
        if (index >= Shape[dimension])
            throw new ArgumentOutOfRangeException(nameof(index));

        var newShape = new ulong[Shape.Length - 1];
        var newStride = new ulong[Stride.Length - 1];
        for (int i = 0, j = 0; i < Shape.Length; i++, j++)
        {
            if (i == dimension) continue;
            newShape[j] = Shape[i];
            newStride[j] = Stride[i];
        }
        return new Tensor
        {
            Shape = newShape,
            Stride = newStride,
            Data = Data,
            Offset = Offset + index * Stride[dimension],
            Type = Type,
            ValueSize = ValueSize,
        };
    }

    /// <summary>
    /// Copy the tensor to a new one dense
    /// </summary>
    /// <returns></returns>
    public Tensor CopyToNew()
    {
        var newData = new byte[ValueSize * Shape.Aggregate(1UL, (acc, x) => acc * x)];
        var newStride = StrideFromShape(Shape, ValueSize);
        CopyToNewRecursive(newStride, newData, (long)Offset, 0, 0);

        return new Tensor
        {
            Shape = Shape,
            Stride = newStride,
            Data = newData,
            Offset = 0,
            Type = Type,
            ValueSize = ValueSize,
        };
    }

    private void CopyToNewRecursive(ulong[] newStride, byte[] newData, long srcOffset, long dstOffset, int dimension)
    {
        var srcStride = Stride[dimension];
        var dstStride = newStride[dimension];
        if (dimension == Shape.Length)
        {
            // Copy the value
            Array.Copy(Data, srcOffset, newData, dstOffset, (long)ValueSize);
        }
        else if (dstStride == srcStride && dstStride < int.MaxValue)
        {
            // If the stride is the same, we can copy the whole remaining dimensions at once
            Array.Copy(Data, srcOffset, newData, dstOffset, (int)dstStride);
        }
        else
        {
            for (long i = 0; i < (long)Shape[dimension]; i++)
                CopyToNewRecursive(newStride, newData, srcOffset + i * (long)srcStride, dstOffset + i * (long)dstStride, dimension + 1);
        }
    }
}
