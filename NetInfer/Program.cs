using NetInfer;

GgufFile file = GgufFile.Load(
    """
    C:\Data\NetInfer\NetInfer\NetInfer\TinyStories-LLaMA2-20M-256h-4l-GQA.f16.gguf
    """);

foreach (var metadata in file.Metadata)
    Console.WriteLine(metadata);

foreach (var tensor in file.TensorsInfos)
{
    Console.WriteLine($"Tensor: {tensor.Name}");
    Console.WriteLine($"  Type: {tensor.Type}");
    Console.WriteLine($"  Offset: {tensor.Offset}");
    Console.WriteLine($"  Dimensions: {string.Join(", ", tensor.Dimensions)}");
}