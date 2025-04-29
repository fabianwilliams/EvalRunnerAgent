namespace EvalRunnerAgent.Models;

public class EvalResult
{
    public required string Input { get; set; }
    public required string GroundTruth { get; set; }
    public required string ModelOutput { get; set; }

    public bool Passed { get; set; }
    public double ThresholdUsed { get; set; }
    public double SimilarityScore { get; set; }
    public double GroundTruthScore { get; set; }
    public double CriteriaScore { get; set; }

    public required string Notes { get; set; }
}
