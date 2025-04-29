namespace EvalRunnerAgent.Models;

public class EvalInput
{
    public required string Input { get; set; }
    public required string GroundTruth { get; set; }
    public required string EvalCriteria { get; set; }
}
