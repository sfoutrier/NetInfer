namespace NetInfer.Test;

public class TensorTest
{
    [Test]
    public void Float32MatMulMultipliesSingleElements()
    {
        var left = Tensor.FromArray([[2f]]);
        var right = Tensor.FromArray([[3f]]);
        var expected = Tensor.FromArray([[6f]]);

        var result = left.MatMul<float>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float32MatMulMultipliesSquareMatrices()
    {
        var left = Tensor.FromArray([[1f, 2f], [3f, 4f]]);
        var right = Tensor.FromArray([[5f, 6f], [7f, 8f]]);
        var expected = Tensor.FromArray([[19f, 22f], [43f, 50f]]);

        var result = left.MatMul<float>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float32MatMulMultipliesRectangularMatrices()
    {
        var left = Tensor.FromArray([[1f, 2f, 3f]]);
        var right = Tensor.FromArray([[4f], [5f], [6f]]);
        var expected = Tensor.FromArray([[32f]]);

        var result = left.MatMul<float>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float32MatMulSupportsTransposedInput()
    {
        var left = Tensor.FromArray([[1f, 2f, 3f], [4f, 5f, 6f]]).Transpose();
        var right = Tensor.FromArray([[10f], [100f]]);
        var expected = Tensor.FromArray([[410f], [520f], [630f]]);

        var result = left.MatMul<float>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float16MatMulMultipliesSingleElements()
    {
        var left = Tensor.FromArray(ToHalf([[2f]]));
        var right = Tensor.FromArray(ToHalf([[3f]]));
        var expected = Tensor.FromArray(ToHalf([[6f]]));

        var result = left.MatMul<Half>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float16MatMulMultipliesSquareMatrices()
    {
        var left = Tensor.FromArray(ToHalf([[1f, 2f], [3f, 4f]]));
        var right = Tensor.FromArray(ToHalf([[5f, 6f], [7f, 8f]]));
        var expected = Tensor.FromArray(ToHalf([[19f, 22f], [43f, 50f]]));

        var result = left.MatMul<Half>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float16MatMulMultipliesRectangularMatrices()
    {
        var left = Tensor.FromArray(ToHalf([[1f, 2f, 3f]]));
        var right = Tensor.FromArray(ToHalf([[4f], [5f], [6f]]));
        var expected = Tensor.FromArray(ToHalf([[32f]]));

        var result = left.MatMul<Half>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void Float16MatMulSupportsTransposedInput()
    {
        var left = Tensor.FromArray(ToHalf([[1f, 2f, 3f], [4f, 5f, 6f]])).Transpose();
        var right = Tensor.FromArray(ToHalf([[10f], [100f]]));
        var expected = Tensor.FromArray(ToHalf([[410f], [520f], [630f]]));

        var result = left.MatMul<Half>(right);
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void EqualsMatchesSameValuesInSeparateStorage()
    {
        var first = Tensor.FromArray([1f, 2f, 3f]);
        var second = Tensor.FromArray([1f, 2f, 3f]);

        Assert.That(first.Equals(second), Is.True);
        Assert.That(first.GetHashCode(), Is.EqualTo(second.GetHashCode()));
    }

    [Test]
    public void EqualsMatchesSameLogicalValuesWithDifferentStrides()
    {
        var transposed = Tensor.FromArray(
        [
            [1f, 2f],
            [3f, 4f],
        ]).Transpose();
        var dense = Tensor.FromArray(
        [
            [1f, 3f],
            [2f, 4f],
        ]);

        Assert.That(transposed, Is.EqualTo(dense));
        Assert.That(transposed.GetHashCode(), Is.EqualTo(dense.GetHashCode()));
    }

    [Test]
    public void EqualsRejectsDifferentShapeTypeOrValues()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2 });

        Assert.That(tensor.Equals(Tensor.FromArray(new float[] { 1, 3 })), Is.False);
        Assert.That(tensor.Equals(Tensor.FromArray(new float[][] { [1], [2] })), Is.False);
        Assert.That(tensor.Equals(Tensor.FromArray([(Half)1, (Half)2])), Is.False);
        Assert.That(tensor.Equals(null), Is.False);
    }

    private static Half[] ToHalf(float[] values) => values.Select(static value => (Half)value).ToArray();
    private static Half[][] ToHalf(float[][] values) => values.Select(static row => ToHalf(row)).ToArray();
}
