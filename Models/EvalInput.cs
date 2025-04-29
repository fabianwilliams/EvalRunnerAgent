namespace EvalRunnerAgent.Models
{
    public class EvalInput
    {
        public string? Input { get; set; }
        public string? GroundTruth { get; set; }
        public string EvalCriteria { get; set; }
    }
}
