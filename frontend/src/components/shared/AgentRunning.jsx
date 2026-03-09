import { Brain } from "lucide-react"

/**
 * AgentRunning
 * Shown while a pipeline agent is executing.
 * Never shows a blank screen — always shows an informative message.
 *
 * Props:
 *   stageName — plain English stage name
 *   message   — what the agent is doing right now
 *   progress  — 0-100 (optional, shows progress bar when provided)
 */
export default function AgentRunning({ stageName, message, progress }) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={`Running: ${stageName}`}
      className="flex flex-col items-center justify-center py-20 px-8 animate-fade-in"
    >
      {/* Spinner with brain icon in centre */}
      <div className="relative mb-6">
        <div
          className="w-16 h-16 rounded-full border-4 border-blue-100 border-t-[#1B3A5C]"
          style={{ animation: "spin 1s linear infinite" }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <Brain size={22} className="text-[#1B3A5C]" />
        </div>
      </div>

      <h2 className="text-xl font-serif text-gray-900 mb-2 text-center">
        {stageName}
      </h2>

      {message && (
        <p className="text-gray-500 text-sm text-center max-w-sm leading-relaxed">
          {message}
        </p>
      )}

      {typeof progress === "number" && (
        <div className="w-64 mt-6">
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#1B3A5C] rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1.5 text-center">
            {progress}% complete
          </p>
        </div>
      )}

      <p className="mt-6 text-xs text-gray-300">
        This may take a moment — please do not close this page.
      </p>
    </div>
  )
}
