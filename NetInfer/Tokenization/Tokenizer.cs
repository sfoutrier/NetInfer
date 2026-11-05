using System.Text;
using System.Text.RegularExpressions;

namespace NetInfer.Tokenization;

/// <summary>
/// Implementation of the GPT-2 tokenizer
/// </summary>
internal class Tokenizer
{
    private static readonly Regex GPT2Regex = new Regex("""
        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        """, RegexOptions.Compiled);

    public static List<uint> Tokenize(string text, string[] tokens, int[] tokenTypes, string[] merges, uint unknownTokenId)
    {
        var tokenMap = tokens.Select((t, i) => new { Token = t, Id = i })
            .ToDictionary(x => x.Token, x => (uint)x.Id);

        var tokenIds = new List<uint>();

        foreach (var match in GPT2Regex.EnumerateMatches(text))
        {
            var tokenText = CharReplacement(text.AsSpan().Slice(match.Index, match.Length));

            // FIXME: we don’t apply BPE merging yet, as this would require to lookhead in the input text

            if (tokenMap.TryGetValue(tokenText, out var tokenId))
                tokenIds.Add(tokenId);
            else
                // TODO: I do think that we could add some magic here to try to find the closest token, but for now we just return the unknown token id
                tokenIds.Add(unknownTokenId);
        }

        return tokenIds;
    }

    /// <summary>
    /// Returns list of utf-8 byte and a corresponding list of unicode strings.
    /// The reversible bpe codes work on unicode strings.
    /// This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    /// When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    /// This is a signficant percentage of your normal, say, 32K bpe vocab.
    /// To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    /// And avoids mapping to whitespace/control characters the bpe code barfs on.
    /// /// </summary>
    private static Dictionary<byte, char> BytesToUnicode()
    {
        // [33..126] .. [161..172] .. [174..255]
        IEnumerable<char> preservedChars = [.. Range('!', '~'), .. Range('¡', '¬'), .. Range('®', 'ÿ')];
        var result = new Dictionary<byte, char>();
        foreach (var c in preservedChars)
            result[(byte)c] = c;

        var n = 0;
        for (char b = (char)0; b < 256; b++)
        {
            if (!result.ContainsKey((byte)b))
                result[(byte)b] = (char)(256 + n++);
        }
        return result;
    }

    private static IEnumerable<char> Range(char start, char end)
    {
        if (end < start)
        {
            var tmp = start;
            start = end;
            end = tmp;
        }

        for (char c = start; c <= end; c++)
            yield return c;
    }

    private static readonly Dictionary<byte, char> bytesToUnicode = BytesToUnicode();
    private static string CharReplacement(ReadOnlySpan<char> input)
    {
        var output = new StringBuilder();
        foreach (var c in input)
        {
            if (bytesToUnicode.TryGetValue((byte)c, out var replacement))
                output.Append(replacement);
            else
                output.Append(c);
        }
        return output.ToString();
    }
}
