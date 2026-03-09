/**
 * stageConfig
 * Stage order and labels — kept separate from PipelineContext.jsx so that
 * plain-object exports don't prevent Vite Fast Refresh on the provider.
 */

export const STAGE_ORDER = [
  "ingestion",
  "validation",
  "eda",
  "cleaning",
  "feature_engineering",
  "normalisation",
  "splitting",
  "training",
  "evaluation",
  "tuning",
  "explainability",
  "deployment",
  "monitoring"
]

export const STAGE_LABELS = {
  ingestion:           { label: "Load Your Data",       icon: "Upload" },
  validation:          { label: "Check Data Quality",   icon: "ShieldCheck" },
  eda:                 { label: "Explore Your Data",    icon: "BarChart2" },
  cleaning:            { label: "Clean Your Data",      icon: "Sparkles" },
  feature_engineering: { label: "Prepare Features",     icon: "Wrench" },
  normalisation:       { label: "Scale Your Data",      icon: "Sliders" },
  splitting:           { label: "Divide Your Data",     icon: "Scissors" },
  training:            { label: "Train the Model",      icon: "Brain" },
  evaluation:          { label: "Measure Performance",  icon: "Target" },
  tuning:              { label: "Fine-Tune the Model",  icon: "SlidersHorizontal" },
  explainability:      { label: "Understand the Model", icon: "Lightbulb" },
  deployment:          { label: "Deploy the Model",     icon: "Rocket" },
  monitoring:          { label: "Monitor Performance",  icon: "Activity" }
}
