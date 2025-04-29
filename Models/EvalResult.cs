namespace EvalRunnerAgent.Models
{
    public class EvalResult
    {
        public string Input { get; set; }
        public string GroundTruth { get; set; }
        public string ModelOutput { get; set; }
        public double SimilarityScore { get; set; }
        public bool Passed { get; set; }
        public double ThresholdUsed { get; set; }
        public string Notes { get; set; }
    }
}
