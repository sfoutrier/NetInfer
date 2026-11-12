using System.Text;

namespace NetInfer;

internal record struct GgufHeader(uint Magic, uint Version, ulong NumTensors, ulong NumKV);

public enum GgufType : uint
{
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

abstract record GgufMetadata(string Key)
{
    public abstract string GetValueAsString();

    public static GgufMetadata ReadMetadata(BinaryReader reader)
    {
        var keyLen = reader.ReadUInt64();
        var keyBytes = reader.ReadBytes((int)keyLen);
        var key = Encoding.UTF8.GetString(keyBytes, 0, keyBytes.Length);

        var type = (GgufType)reader.ReadUInt32();
        switch (type)
        {
            case GgufType.UInt8:
                var byteValue = reader.ReadByte();
                return new GgufMetadataUint8(key, byteValue);
            case GgufType.Int8:
                var sbyteValue = reader.ReadSByte();
                return new GgufMetadataInt8(key, sbyteValue);
            case GgufType.UInt16:
                var ushortValue = reader.ReadUInt16();
                return new GgufMetadataUint16(key, ushortValue);
            case GgufType.Int16:
                var shortValue = reader.ReadInt16();
                return new GgufMetadataInt16(key, shortValue);
            case GgufType.UInt32:
                var uintValue = reader.ReadUInt32();
                return new GgufMetadataUint32(key, uintValue);
            case GgufType.Int32:
                var intValue = reader.ReadInt32();
                return new GgufMetadataInt32(key, intValue);
            case GgufType.String:
                string stringValue = ReadString(reader);
                return new GgufMetadataString(key, stringValue);
            case GgufType.UInt64:
                var ulongValue = reader.ReadUInt64();
                return new GgufMetadataUint64(key, ulongValue);
            case GgufType.Int64:
                var longValue = reader.ReadInt64();
                return new GgufMetadataInt64(key, longValue);
            case GgufType.Float64:
                var floatValue = reader.ReadDouble();
                return new GgufMetadataFloat64(key, floatValue);
            case GgufType.Float32:
                var floatValue32 = reader.ReadSingle();
                return new GgufMetadataFloat32(key, floatValue32);
            case GgufType.Bool:
                var boolValue = reader.ReadBoolean();
                return new GgufMetadataBool(key, boolValue);
            case GgufType.Array:
                return ReadArray(reader, key);

            default: throw new NotSupportedException($"Unsupported metadata type: {type}");
        }
    }

    public static string ReadString(BinaryReader reader)
    {
        var valueLen = reader.ReadUInt64();
        var valueBytes = reader.ReadBytes((int)valueLen);
        var stringValue = Encoding.UTF8.GetString(valueBytes, 0, valueBytes.Length);
        return stringValue;
    }

    private static T[] Read<T>(BinaryReader reader, Func<BinaryReader, T> valueReader)
    {
        var len = reader.ReadUInt64();
        var values = new T[len];
        for (ulong i = 0; i < len; i++)
            values[i] = valueReader(reader);
        return values;
    }

    private static GgufMetadata ReadArray(BinaryReader reader, string key)
    {
        var subType = (GgufType)reader.ReadUInt32();

        switch (subType)
        {
            case GgufType.UInt8:
                var byteValue = reader.ReadByte();
                return new GgufMetadataArray<byte>(key, GgufType.UInt8, Read(reader, reader => reader.ReadByte()));
            case GgufType.Int8:
                return new GgufMetadataArray<sbyte>(key, GgufType.Int8, Read(reader, reader => reader.ReadSByte()));
            case GgufType.UInt16:
                return new GgufMetadataArray<ushort>(key, GgufType.UInt16, Read(reader, reader => reader.ReadUInt16()));
            case GgufType.Int16:
                return new GgufMetadataArray<short>(key, GgufType.Int16, Read(reader, reader => reader.ReadInt16()));
            case GgufType.UInt32:
                return new GgufMetadataArray<uint>(key, GgufType.UInt32, Read(reader, reader => reader.ReadUInt32()));
            case GgufType.Int32:
                return new GgufMetadataArray<int>(key, GgufType.Int32, Read(reader, reader => reader.ReadInt32()));
            case GgufType.UInt64:
                return new GgufMetadataArray<ulong>(key, GgufType.UInt64, Read(reader, reader => reader.ReadUInt64()));
            case GgufType.Int64:
                return new GgufMetadataArray<long>(key, GgufType.Int64, Read(reader, reader => reader.ReadInt64()));
            case GgufType.Float32:
                return new GgufMetadataArray<float>(key, GgufType.Float32, Read(reader, reader => reader.ReadSingle()));
            case GgufType.Float64:
                return new GgufMetadataArray<double>(key, GgufType.Float64, Read(reader, reader => reader.ReadDouble()));
            case GgufType.Bool:
                return new GgufMetadataArray<bool>(key, GgufType.Bool, Read(reader, reader => reader.ReadBoolean()));
            case GgufType.String:
                return new GgufMetadataArray<string>(key, GgufType.String, Read(reader, ReadString));

            default: throw new NotSupportedException($"Unsupported metadata type: {subType}");
        }
    }
}

record GgufMetadataUint8(string Key, byte Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}

record GgufMetadataInt8(string Key, sbyte Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataUint16(string Key, ushort Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataInt16(string Key, short Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataUint32(string Key, uint Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataInt32(string Key, int Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataUint64(string Key, ulong Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataInt64(string Key, long Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataBool(string Key, bool Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataFloat32(string Key, float Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataFloat64(string Key, double Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}
record GgufMetadataArray<T>(string Key, GgufType Type, T[] Values) : GgufMetadata(Key)
{
    private const int MAX_DIPLAYED = 10;
    override public string GetValueAsString() => Values.Length switch
    {
        > 5 => $"[{string.Join(", ", Values[..MAX_DIPLAYED])}, ... {Values.Length - MAX_DIPLAYED} more]",
        _ => $"[{string.Join(", ", Values)}]",
    };

    public override string ToString() => $"GgufMetadataArray<{Type}>({Key}, [{GetValueAsString()}])";
}
record GgufMetadataString(string Key, string Value) : GgufMetadata(Key)
{
    override public string GetValueAsString() => Value.ToString();
}

public enum TensorType : uint
{
    Float32 = 0,
    Float16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    Int8 = 24,
    Int16 = 25,
    Int32 = 26,
    Int64 = 27,
    Float64 = 28,
    IQ1_M = 29,
    BFloat16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

record TensorInfo(string Name, ulong[] Dimensions, TensorType Type, ulong Offset)
{
    public static TensorInfo[] ReadTensorInfos(BinaryReader reader, ulong numTensors)
    {
        var tensorInfos = new TensorInfo[numTensors];
        for (ulong i = 0; i < numTensors; i++)
        {
            var name = GgufMetadata.ReadString(reader);

            var numDimensions = reader.ReadUInt32();
            var dimensions = new ulong[numDimensions];
            for (ulong j = 0; j < numDimensions; j++)
                dimensions[j] = reader.ReadUInt64();
            var type = (TensorType)reader.ReadUInt32();
            var offset = reader.ReadUInt64();
            tensorInfos[i] = new TensorInfo(name, dimensions, type, offset);
        }
        return tensorInfos;
    }
}

internal record GgufFile(GgufHeader Header, GgufMetadata[] Metadata, TensorInfo[] TensorsInfos, Dictionary<string, Tensor> Tensors)
{
    public static GgufFile Load(string path)
    {
        using var file = File.OpenRead(path);
        using var reader = new BinaryReader(file);
        var magic = reader.ReadUInt32();
        var version = reader.ReadUInt32();
        var numTensors = reader.ReadUInt64();
        var numKV = reader.ReadUInt64();

        var header = new GgufHeader(magic, version, numTensors, numKV);
        var metadata = LoadMetadata(reader, numKV);
        var tensors = TensorInfo.ReadTensorInfos(reader, numTensors);
        var tensorDict = LoadTensors(reader, tensors);
        return new GgufFile(header, metadata, tensors, tensorDict);
    }

    private static ulong SizeFromTensorType(TensorType type) => type switch
    {
        TensorType.Float32 => 4,
        TensorType.Float16 => 2,
        TensorType.Q8_0 => 1,
        TensorType.Q8_1 => 1,
        TensorType.Q8_K => 1,
        TensorType.Int8 => 1,
        TensorType.Int16 => 2,
        TensorType.Int32 => 4,
        TensorType.Int64 => 8,
        TensorType.Float64 => 8,
        TensorType.BFloat16 => 2,
        _ => throw new NotSupportedException($"Unsupported tensor type: {type}"),
    };

    public static Dictionary<string, Tensor> LoadTensors(BinaryReader reader, TensorInfo[] tensorInfos)
    {
        var result = new Dictionary<string, Tensor>();
        foreach (var tensorInfo in tensorInfos)
        {
            ulong valueSize = SizeFromTensorType(tensorInfo.Type);
            ulong stride = valueSize;
            var strides = new ulong[tensorInfo.Dimensions.Length];
            for (int i = tensorInfo.Dimensions.Length - 1; i >= 0; i--)
            {
                var dim = tensorInfo.Dimensions[i];
                strides[i] = stride;
                stride *= dim;
            }
            var data = reader.ReadBytes((int)stride);

            var tensor = new Tensor
            {
                Data = data,
                Offset = 0,
                Shape = tensorInfo.Dimensions,
                Stride = strides,
                Type = tensorInfo.Type,
                ValueSize = valueSize,
            };
            result[tensorInfo.Name] = tensor;
        }
        return result;
    }

    private static GgufMetadata[] LoadMetadata(BinaryReader reader, ulong numKV)
    {
        var result = new GgufMetadata[numKV];
        for (ulong i = 0; i < numKV; i++)
            result[i] = GgufMetadata.ReadMetadata(reader);
        return result;
    }
}