using Microsoft.Extensions.Configuration;
using EvalRunnerAgent.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Text.Json;

namespace EvalRunnerAgent;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("🚀 Starting Evaluation Agent...");

        // Load Configuration
        var configuration = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .Build();

        var openAiSection = configuration.GetSection("OpenAI");
        var openAiKey = openAiSection["ApiKey"];
        var modelId = openAiSection["ModelId"];

        if (string.IsNullOrEmpty(openAiKey))
        {
            Console.WriteLine("❌ OpenAI API key not found in appsettings.json. Exiting.");
            return;
        }

        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: modelId,
            apiKey: openAiKey);
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
                        new ChatHistory
                        {
                            new ChatMessageContent(AuthorRole.User, eval.Input)
                        });

                    if (response != null)
                        break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Attempt {attempt} failed: {ex.Message}");
                    if (attempt < maxRetries)
                    {
                        await Task.Delay(retryDelayMs);
                    }
                }
            }

            if (response == null)
            {
                Console.WriteLine("❌ Failed to get a response after retries. Skipping...");
                continue;
            }

            var modelOutput = response.Content.Trim();
            bool passed = EvaluateOutput(modelOutput, eval.GroundTruth);

            results.Add(new EvalResult
            {
                Input = eval.Input,
                GroundTruth = eval.GroundTruth,
                ModelOutput = modelOutput,
                Passed = passed,
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

    static bool EvaluateOutput(string modelOutput, string groundTruth)
    {
        if (string.IsNullOrWhiteSpace(modelOutput) || string.IsNullOrWhiteSpace(groundTruth))
            return false;

        var cleanOutput = modelOutput.Trim().ToLowerInvariant();
        var cleanGroundTruth = groundTruth.Trim().ToLowerInvariant();

        return cleanOutput.Contains(cleanGroundTruth);
    }
}
