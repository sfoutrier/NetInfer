namespace NetInfer.Inference;

internal class Gpt2Inference
{
    private readonly GgufFile _file;

    public Gpt2Inference(GgufFile file)
    {
        _file = file;
    }

    public void Prefill(IEnumerable<uint> tokens)
    {
        var embeddings = _file.Tensors["token_embd.weight"];
        var embeddingsLen = embeddings.Shape[0];

        var attentionK = _file.Tensors["blk.0.attn_k.weight"];
        var attentionQ = _file.Tensors["blk.0.attn_q.weight"];
        var attentionV = _file.Tensors["blk.0.attn_v.weight"];

        Tensor? x = null;
        Tensor? h = null;
        foreach (var token in tokens)
        {
            var embd = embeddings.OrthogonalProject(0, token);
            x ??= embd.CopyToNew();
        }
    }
}
