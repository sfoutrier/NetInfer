using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
internal record Tensor : IEquatable<Tensor>
{
    public required ulong[] Shape { get; init; }
    public required ulong[] Stride { get; init; }
    public required byte[] Data { get; init; }
    public required ulong Offset { get; init; }
    public required ulong ValueSize { get; init; }
    public required TensorType Type { get; init; }

    public virtual bool Equals(Tensor? other)
    {
        if (ReferenceEquals(this, other))
            return true;

        if (other is null ||
            Type != other.Type ||
            ValueSize != other.ValueSize ||
            !Shape.AsSpan().SequenceEqual(other.Shape))
            return false;

        var elementCount = Shape.Aggregate(1UL, static (count, dimension) => checked(count * dimension));
        for (ulong i = 0; i < elementCount; i++)
        {
            if (!GetElementSpan(i).SequenceEqual(other.GetElementSpan(i)))
                return false;
        }

        return true;
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Type);
        hash.Add(ValueSize);
        hash.Add(Shape.Length);
        foreach (var dimension in Shape)
            hash.Add(dimension);

        var elementCount = Shape.Aggregate(1UL, static (count, dimension) => checked(count * dimension));
        for (ulong i = 0; i < elementCount; i++)
        {
            foreach (var valueByte in GetElementSpan(i))
                hash.Add(valueByte);
        }

        return hash.ToHashCode();
    }

    private ReadOnlySpan<byte> GetElementSpan(ulong linearOffset)
    {
        var byteOffset = Offset;
        for (var dimension = Shape.Length - 1; dimension >= 0; dimension--)
        {
            var dimensionSize = Shape[dimension];
            byteOffset += linearOffset % dimensionSize * Stride[dimension];
            linearOffset /= dimensionSize;
        }

        return Data.AsSpan(checked((int)byteOffset), checked((int)ValueSize));
    }

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

    /// <summary>
    /// Performs matrix multiplication: result = this * other
    /// </summary>
    public Tensor MatMul<T>(Tensor other)
        where T : INumber<T>, IAdditionOperators<T, T, T>, IMultiplyOperators<T, T, T>
    {
        var (m, k) = Shape switch
        {
            [var mVal, var kVal] => (mVal, kVal),
            _ => throw new ArgumentException("Invalid tensor shape for matrix multiplication.")
        };

        var (kOther, n) = other.Shape switch
        {
            [var kVal, var nVal] => (kVal, nVal),
            _ => throw new ArgumentException("Invalid tensor shape for matrix multiplication.")
        };

        if (k != kOther)
            throw new ArgumentException($"Dimension mismatch: {k} != {kOther}");

        if (ValueSize != (ulong)Unsafe.SizeOf<T>() || other.ValueSize != ValueSize)
            throw new ArgumentException("Tensor value size does not match the matrix multiplication type.");

        var resultData = new byte[checked((int)(m * n * ValueSize))];

        // Simple MatMul Kernel
        ulong iOffset = Offset;
        ulong resultOffset = 0;
        for (ulong i = 0; i < m; i++)
        {
            ulong jOffset = other.Offset;
            for (ulong j = 0; j < n; j++)
            {
                T sum = default!;
                ulong leftOffset = iOffset;
                ulong rightOffset = jOffset;
                for (ulong l = 0; l < k; l++)
                {
                    T left = Unsafe.ReadUnaligned<T>(ref Data[leftOffset]);
                    T right = Unsafe.ReadUnaligned<T>(ref other.Data[rightOffset]);
                    sum += left * right;
                    leftOffset += Stride[1];
                    rightOffset += other.Stride[0];
                }
                Unsafe.WriteUnaligned(ref resultData[resultOffset], sum);

                resultOffset += ValueSize;
                jOffset += other.Stride[1];
            }

            iOffset += Stride[0];
        }

        return this with
        {
            Shape = [m, n],
            Stride = [checked(n * ValueSize), ValueSize],
            Data = resultData,
            Offset = 0,
        };
    }

    public Tensor Transpose()
    {
        var (m, k) = Shape switch
        {
            [var mVal, var kVal] => (mVal, kVal),
            _ => throw new NotSupportedException("Only 2D Transpose is supported for now.")
        };

        var (sm, sk) = Stride switch
        {
            [var smVal, var skVal] => (smVal, skVal),
            _ => throw new NotSupportedException("Only 2D Transpose is supported for now.")
        };

        return this with { Shape = [k, m], Stride = [sk, sm] };
    }

    public static Tensor FromArray<T>(T[] array)
        where T : struct, INumber<T>
    {
        var valueSize = (ulong)Unsafe.SizeOf<T>();
        var data = new byte[(ulong)array.Length * valueSize];

        MemoryMarshal.Cast<T, byte>(array.AsSpan()).CopyTo(data);

        return new Tensor
        {
            Shape = [checked((ulong)array.Length)],
            Stride = [valueSize],
            Data = data,
            Offset = 0,
            ValueSize = valueSize,
            Type = typeof(T) == typeof(float) ? TensorType.Float32 :
                typeof(T) == typeof(Half) ? TensorType.Float16 :
                throw new NotSupportedException($"Type {typeof(T)} is not supported."),
        };
    }

    public static Tensor FromArray<T>(T[][] array)
        where T : struct, INumber<T>
    {
        var valueSize = (ulong)Unsafe.SizeOf<T>();
        var rows = (ulong)array.Length;
        var cols = (ulong)array[0].Length;

        var data = new byte[rows * cols * valueSize];

        var offset = 0ul;
        var rowSize = cols * valueSize;
        foreach (var row in array)
        {
            if ((ulong)row.Length != cols)
                throw new ArgumentException("All rows must have the same length.");

            MemoryMarshal.Cast<T, byte>(row.AsSpan()).CopyTo(data.AsSpan().Slice((int)offset, (int)rowSize));
            offset += rowSize;
        }

        return new Tensor
        {
            Shape = [rows, cols],
            Stride = [cols * valueSize, valueSize],
            Data = data,
            Offset = 0,
            ValueSize = valueSize,
            Type = GetTensorType<T>(),
        };
    }

    private static TensorType GetTensorType<T>()
        where T : struct, INumber<T>
    {
        if (typeof(T) == typeof(float))
            return TensorType.Float32;
        else if (typeof(T) == typeof(Half))
            return TensorType.Float16;
        else
            throw new NotSupportedException($"Type {typeof(T)} is not supported.");
    }
}
