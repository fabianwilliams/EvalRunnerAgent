using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace EvalRunnerAgent.Services;

public class OllamaEmbeddingService : ITextEmbeddingGenerationService
{
    private readonly string _model;
    private readonly string _endpoint;
    private readonly HttpClient _httpClient;

    public OllamaEmbeddingService(string model, string endpoint)
    {
        _model = model;
        _endpoint = endpoint;
        _httpClient = new HttpClient();
    }

    public IReadOnlyDictionary<string, object?> Attributes { get; } = new Dictionary<string, object?>
    {
        { "ServiceId", "ollama" },
        { "ApiType", "local" },
        { "ModelId", "nomic-embed-text:latest" } // Update if needed
    };

    public async Task<ReadOnlyMemory<float>> GenerateEmbeddingAsync(string input, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var list = await GenerateEmbeddingsAsync(new List<string> { input }, kernel, cancellationToken);
        return list[0];
    }

    public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(IList<string> data, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        Console.WriteLine($"🧠 [EmbeddingService] Generating embeddings for {data.Count} input(s)");

        var results = new List<ReadOnlyMemory<float>>();

        foreach (var input in data)
        {
            Console.WriteLine($"📎 Embedding input: {input}");

            var payload = new
            {
                model = _model,
                prompt = input
            };

            var json = JsonSerializer.Serialize(payload);
            Console.WriteLine($"📤 Payload: {json}");
            Console.WriteLine($"🌐 Posting to: {_endpoint}/embeddings");

            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync($"{_endpoint}/embeddings", content, cancellationToken);

            Console.WriteLine($"📥 Response Status: {response.StatusCode}");
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
            var parsed = JsonDocument.Parse(responseContent);
            var embeddingArray = parsed.RootElement
                .GetProperty("embedding")
                .EnumerateArray()
                .Select(e => e.GetSingle())
                .ToArray();

            results.Add(embeddingArray);
        }

        return results;
    }
}
