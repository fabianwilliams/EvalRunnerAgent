using Microsoft.Extensions.Configuration;
using EvalRunnerAgent.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Text.Json;



namespace EvalRunnerAgent;

class Program
{
    static async Task Main(string[] args)
    {
        // Adding flexibility for tuning the pass criteria: parse --threshold from command line
        double? thresholdOverride = null;
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "--threshold" && double.TryParse(args[i + 1], out double parsed))
            {
                thresholdOverride = parsed;
                Console.WriteLine($"📏 Overriding similarity threshold: {thresholdOverride}");
                break;
            }
        }

        Console.WriteLine("🚀 Starting Evaluation Agent...");

        // Load Configuration
        var configuration = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .Build();

        var openAiSection = configuration.GetSection("OpenAI");
        var openAiKey = openAiSection["ApiKey"];
        var modelId = openAiSection["ModelId"];
        var embeddingModel = openAiSection["EmbeddingModel"];

        // 🔐 Validate config values BEFORE using them
        if (string.IsNullOrEmpty(openAiKey) || string.IsNullOrEmpty(modelId) || string.IsNullOrEmpty(embeddingModel))
        {
            Console.WriteLine("❌ Missing OpenAI config. Check appsettings.json.");
            return;
        }

        var configThreshold = configuration.GetValue<double>("Eval:SimilarityThreshold", 0.85);
        var threshold = thresholdOverride ?? configThreshold;

        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(modelId: modelId, apiKey: openAiKey);
        var kernel = builder.Build();

        // Load evals
        var evalSet = LoadEvalSet("Data/evalset.json");

        var results = new List<EvalResult>();
        var chat = kernel.GetRequiredService<IChatCompletionService>();

        foreach (var eval in evalSet)
        {
            Console.WriteLine($"🔹 Running eval for input: {eval.Input}");

            ChatMessageContent? response = null;
            int maxRetries = 3;
            int retryDelayMs = 2000;

            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    response = await chat.GetChatMessageContentAsync(
                        new ChatHistory { new ChatMessageContent(AuthorRole.User, eval.Input) });

                    if (response != null)
                        break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Attempt {attempt} failed: {ex.Message}");
                    if (attempt < maxRetries) await Task.Delay(retryDelayMs);
                }
            }

            if (response == null)
            {
                Console.WriteLine("❌ Failed to get a response after retries. Skipping...");
                continue;
            }

            var modelOutput = response.Content.Trim();

            // ✅ Explicitly deconstruct return type to fix CS8130
            (bool passed, double score) = await EvaluateOutputAsync(modelOutput, eval.GroundTruth, embeddingModel, openAiKey, threshold);

            results.Add(new EvalResult
            {
                Input = eval.Input,
                GroundTruth = eval.GroundTruth,
                ModelOutput = modelOutput,
                Passed = passed,
                ThresholdUsed = threshold,
                SimilarityScore = score,
                Notes = passed ? "✅ Pass" : "❌ Fail"
            });
        }

        // Save results to file
        var timestamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmss");
        var outputPath = Path.Combine("Data", $"eval_results_{timestamp}.json");
        Directory.CreateDirectory("Data"); // Ensure Data folder exists

        var serializedResults = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(outputPath, serializedResults);

        Console.WriteLine($"\n📝 Results saved to {outputPath}");

        // Print summary
        var passCount = results.Count(r => r.Passed);
        var failCount = results.Count(r => !r.Passed);

        Console.WriteLine($"✅ Passed: {passCount} | ❌ Failed: {failCount}");
    }

    static List<EvalInput> LoadEvalSet(string filePath)
    {
        var json = File.ReadAllText(filePath);
        return JsonSerializer.Deserialize<List<EvalInput>>(json) ?? new List<EvalInput>();
    }

    /* Removing in favor of one that will now work with my Cosine similarity function
    static bool EvaluateOutput(string modelOutput, string groundTruth)
    {
        if (string.IsNullOrWhiteSpace(modelOutput) || string.IsNullOrWhiteSpace(groundTruth))
            return false;

        var cleanOutput = modelOutput.Trim().ToLowerInvariant();
        var cleanGroundTruth = groundTruth.Trim().ToLowerInvariant();

        return cleanOutput.Contains(cleanGroundTruth);
    }
    */

    // ✅ Embedding support via OpenAITextEmbeddingGenerationService
#pragma warning disable SKEXP0010
    static async Task<(bool passed, double score)> EvaluateOutputAsync(
        string modelOutput,
        string groundTruth,
        string embeddingModel,
        string openAiKey,
        double threshold = 0.85)
    {
        if (string.IsNullOrWhiteSpace(modelOutput) || string.IsNullOrWhiteSpace(groundTruth))
            return (false, 0.0);

        var embeddingService = new OpenAITextEmbeddingGenerationService(embeddingModel, openAiKey);

        var embeddingTasks = await Task.WhenAll(
            embeddingService.GenerateEmbeddingAsync(modelOutput),
            embeddingService.GenerateEmbeddingAsync(groundTruth)
        );

        // Convert ReadOnlyMemory<float> → double[] for cosine similarity
        var vectorA = embeddingTasks[0].ToArray().Select(x => (double)x).ToArray();
        var vectorB = embeddingTasks[1].ToArray().Select(x => (double)x).ToArray();

        var score = CosineSimilarity(vectorA, vectorB);
        Console.WriteLine($"🧠 Similarity Score: {score:F4}");

        return (score >= threshold, score);
    }
#pragma warning restore SKEXP0010

    static double CosineSimilarity(double[] vectorA, double[] vectorB)
    {
        double dot = 0.0;
        double magA = 0.0;
        double magB = 0.0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dot += vectorA[i] * vectorB[i];
            magA += Math.Pow(vectorA[i], 2);
            magB += Math.Pow(vectorB[i], 2);
        }

        magA = Math.Sqrt(magA);
        magB = Math.Sqrt(magB);

        return dot / (magA * magB);
    }
}
