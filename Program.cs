using Microsoft.Extensions.Configuration;
using EvalRunnerAgent.Models;
using EvalRunnerAgent.Services;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Text.Json;

#pragma warning disable SKEXP0010 // Suppress OpenAITextEmbeddingGenerationService experimental warning

namespace EvalRunnerAgent;

class Program
{
    static async Task Main(string[] args)
    {
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

        var useLocalModel = configuration.GetValue<bool>("Eval:UseLocalModels", false);

        var openAiKey = configuration["OpenAI:ApiKey"];
        var openAiModel = configuration["OpenAI:ModelId"];
        var openAiEmbeddingModel = configuration["OpenAI:EmbeddingModel"];

        var ollamaModel = configuration["Ollama:Model"];
        var ollamaEndpoint = configuration["Ollama:Endpoint"];
        var ollamaEmbeddingEndpoint = configuration["Ollama:EmbeddingEndpoint"];
        var ollamaEmbeddingModel = configuration["Ollama:EmbeddingModel"];

        var thresholdConfig = configuration.GetValue<double>("Eval:SimilarityThreshold", 0.75);
        var threshold = thresholdOverride ?? thresholdConfig;
        var gtWeight = configuration.GetValue<double>("Eval:GroundTruthWeight", 0.7);
        var criteriaWeight = configuration.GetValue<double>("Eval:EvalCriteriaWeight", 0.3);

        // Setup Kernel
        var kernelBuilder = Kernel.CreateBuilder();

        if (useLocalModel)
        {
            Console.WriteLine("🤖 Using Local Ollama Model for Chat Completion...");
            var endpoint = new Uri(ollamaEndpoint!);
            var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

            kernelBuilder.AddOpenAIChatCompletion(
                modelId: ollamaModel!,
                endpoint: endpoint,
                apiKey: string.Empty,
                httpClient: httpClient);
        }
        else
        {
            if (string.IsNullOrWhiteSpace(openAiModel) || string.IsNullOrWhiteSpace(openAiKey))
            {
                Console.WriteLine("❌ OpenAI modelId or apiKey is missing.");
                return;
            }

            Console.WriteLine("🌐 Using OpenAI Model for Chat Completion...");
            kernelBuilder.AddOpenAIChatCompletion(
                modelId: openAiModel!,
                apiKey: openAiKey);
        }



        var kernel = kernelBuilder.Build();
        var chat = kernel.GetRequiredService<IChatCompletionService>();

        // Setup Embedding Service
        ITextEmbeddingGenerationService embeddingService;
        if (useLocalModel)
        {
            embeddingService = new OllamaEmbeddingService(ollamaEmbeddingModel!, ollamaEmbeddingEndpoint!);
        }
        else
        {
            embeddingService = new OpenAITextEmbeddingGenerationService(openAiEmbeddingModel!, openAiKey!);
        }

        var evalSet = LoadEvalSet("Data/evalset.json");
        var results = new List<EvalResult>();

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
                    response = await chat.GetChatMessageContentAsync(new ChatHistory
                    {
                        new ChatMessageContent(AuthorRole.User, eval.Input!)
                    });

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

            var modelOutput = response.Content!.Trim();

            (bool passed, double finalScore, double gtScore, double criteriaScore) =
                await EvaluateOutputAsync(
                    modelOutput,
                    eval.GroundTruth!,
                    eval.EvalCriteria!,
                    embeddingService,
                    threshold,
                    gtWeight,
                    criteriaWeight);


            results.Add(new EvalResult
            {
                Input = eval.Input,
                GroundTruth = eval.GroundTruth,
                ModelOutput = modelOutput,
                Passed = passed,
                ThresholdUsed = threshold,
                SimilarityScore = finalScore,
                GroundTruthScore = gtScore,
                CriteriaScore = criteriaScore,
                Notes = passed ? "✅ Pass" : "❌ Fail"
            });
        }

        var timestamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmss");
        var outputPath = Path.Combine("Data", $"eval_results_{timestamp}.json");
        Directory.CreateDirectory("Data");

        var serializedResults = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(outputPath, serializedResults);

        Console.WriteLine($"\n📝 Results saved to {outputPath}");
        Console.WriteLine($"✅ Passed: {results.Count(r => r.Passed)} | ❌ Failed: {results.Count(r => !r.Passed)}");
    }

    static List<EvalInput> LoadEvalSet(string filePath)
    {
        var json = File.ReadAllText(filePath);
        return JsonSerializer.Deserialize<List<EvalInput>>(json) ?? new List<EvalInput>();
    }

    static async Task<(bool passed, double finalScore, double gtScore, double criteriaScore)> EvaluateOutputAsync(
        string modelOutput,
        string groundTruth,
        string evalCriteria,
        ITextEmbeddingGenerationService embeddingService,
        double threshold,
        double gtWeight,
        double criteriaWeight)
    {
        if (string.IsNullOrWhiteSpace(modelOutput) || string.IsNullOrWhiteSpace(groundTruth))
            return (false, 0.0, 0.0, 0.0);

        var embeddings = await Task.WhenAll(
            embeddingService.GenerateEmbeddingAsync(modelOutput),
            embeddingService.GenerateEmbeddingAsync(groundTruth),
            embeddingService.GenerateEmbeddingAsync(evalCriteria ?? "")
        );

        var outputVec = embeddings[0].ToArray().Select(x => (double)x).ToArray();
        var gtVec = embeddings[1].ToArray().Select(x => (double)x).ToArray();
        var criteriaVec = embeddings[2].ToArray().Select(x => (double)x).ToArray();

        var gtScore = CosineSimilarity(outputVec, gtVec);
        var criteriaScore = string.IsNullOrWhiteSpace(evalCriteria) ? 0.0 : CosineSimilarity(outputVec, criteriaVec);
        var finalScore = (gtScore * gtWeight) + (criteriaScore * criteriaWeight);

        Console.WriteLine($"🧠 GT Score: {gtScore:F4}, Criteria Score: {criteriaScore:F4}, Weighted: {finalScore:F4}");

        return (finalScore >= threshold, finalScore, gtScore, criteriaScore);
    }

    static double CosineSimilarity(double[] vectorA, double[] vectorB)
    {
        double dot = 0.0, magA = 0.0, magB = 0.0;
        for (int i = 0; i < vectorA.Length; i++)
        {
            dot += vectorA[i] * vectorB[i];
            magA += Math.Pow(vectorA[i], 2);
            magB += Math.Pow(vectorB[i], 2);
        }
        return dot / (Math.Sqrt(magA) * Math.Sqrt(magB));
    }
}
