---
name: ui-interaction
description: >
  Responsible for all dynamic, conversational, and decision-driven interactions
  within the pipeline UI. Covers how decisions are presented and collected, how
  the orchestrator's plain English responses are displayed, how the user types
  free text, how tradeoffs between alternatives are shown, how confirmations for
  irreversible actions work, how errors and warnings are communicated, how charts
  are captioned and presented, and how the final report is assembled and displayed.
  Works in conjunction with ui-shell which defines the overall layout and structure.
  Trigger when any of the following are mentioned: "decision prompt", "show options",
  "ask the user", "present alternatives", "collect input", "confirm action",
  "display results", "show chart", "free text input", "conversation", "user choice",
  "tradeoff", "explain to user", "display warning", or any situation involving
  dynamic user interaction within the pipeline interface.
---

# UI Interaction Skill

The UI Interaction skill defines how the application talks to the user — and how
the user talks back. The shell (skill 17) provides the frame. This skill provides
the conversation that happens inside it.

Every interaction follows one rule: **the user must always understand what they
are being asked and why.** No decision is presented without context. No action
is irreversible without an explicit confirmation. No technical term appears
without a plain English explanation.

---

## Responsibilities

1. Render decision cards — single choice, multi-choice, and ranked alternatives
2. Display tradeoffs between options clearly and fairly
3. Collect free text input for goal capture and open questions
4. Confirm irreversible actions with explicit modals
5. Render the Orchestrator's plain English narration
6. Caption and present charts from backend agents
7. Handle errors and warnings at each severity level
8. Collect and display privacy decisions
9. Build and display the final pipeline report
10. Handle "why?" follow-up questions — the user can always ask for more detail

---

## The Conversation Display

The Orchestrator's plain English output is the primary voice of the application.
It is rendered as a conversation — not as a form or a technical log.

```jsx
// OrchestratorMessage.jsx
function OrchestratorMessage({ message, animate = true }) {
  return (
    <div
      className={`
        flex gap-4 mb-6
        ${animate ? "animate-in fade-in slide-in-from-bottom-2 duration-300" : ""}
      `}
    >
      {/* Avatar */}
      <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-blue-800
                      flex items-center justify-center">
        <Brain size={18} className="text-white" />
      </div>

      {/* Message bubble */}
      <div className="flex-1 bg-white rounded-2xl rounded-tl-sm
                      border border-gray-100 shadow-sm px-5 py-4">
        <p className="text-sm font-medium text-blue-800 mb-2">Pipeline Guide</p>
        <div className="text-gray-700 text-base leading-relaxed whitespace-pre-line">
          {message}
        </div>
      </div>
    </div>
  )
}
```

---

## Decision Cards

The most common interaction pattern. Used whenever the Orchestrator presents a
choice — cleaning strategy, model selection, scaling method, etc.

### Single Choice Decision

```jsx
function SingleChoiceDecision({ question, options, onConfirm, recommended }) {
  const [selected, setSelected] = useState(recommended || null)

  return (
    <div className="mb-6">
      <p className="text-gray-800 font-medium mb-4 text-base">{question}</p>

      <div className="space-y-3 mb-5">
        {options.map(opt => (
          <button
            key={opt.id}
            onClick={() => setSelected(opt.id)}
            className={`
              w-full text-left p-4 rounded-xl border-2 transition-all duration-150
              ${selected === opt.id
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 bg-white hover:border-gray-300"}
            `}
          >
            <div className="flex items-start gap-3">
              <div className={`
                mt-0.5 w-5 h-5 rounded-full border-2 flex-shrink-0
                flex items-center justify-center
                ${selected === opt.id ? "border-blue-500" : "border-gray-300"}
              `}>
                {selected === opt.id && (
                  <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                )}
              </div>

              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-gray-900">{opt.label}</span>
                  {opt.id === recommended && (
                    <span className="text-xs px-2 py-0.5 bg-amber-100
                                     text-amber-700 rounded-full font-medium">
                      Recommended
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-500 leading-relaxed">
                  {opt.explanation}
                </p>
                {opt.tradeoff && (
                  <p className="text-xs text-gray-400 mt-1.5 italic">
                    Trade-off: {opt.tradeoff}
                  </p>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      <button
        onClick={() => selected && onConfirm(selected)}
        disabled={!selected}
        className="px-6 py-3 bg-blue-800 text-white rounded-xl font-medium
                   hover:bg-blue-900 disabled:opacity-40 disabled:cursor-not-allowed
                   transition-colors duration-150"
      >
        Confirm Choice
      </button>
    </div>
  )
}
```

### Multi-Column Comparison Card

Used when alternatives have multiple comparable properties — e.g. model selection.

```jsx
function ComparisonTable({ title, columns, rows, recommended, onSelect }) {
  const [selected, setSelected] = useState(recommended)

  return (
    <div className="mb-6 overflow-x-auto">
      <p className="font-medium text-gray-800 mb-3">{title}</p>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-2 pr-4 text-gray-500 font-medium">Option</th>
            {columns.map(col => (
              <th key={col} className="text-left py-2 px-3 text-gray-500 font-medium">
                {col}
              </th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr
              key={row.id}
              onClick={() => setSelected(row.id)}
              className={`
                border-b border-gray-100 cursor-pointer transition-colors
                ${selected === row.id ? "bg-blue-50" : "hover:bg-gray-50"}
              `}
            >
              <td className="py-3 pr-4">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-900">{row.label}</span>
                  {row.id === recommended && (
                    <span className="text-xs px-1.5 py-0.5 bg-amber-100
                                     text-amber-700 rounded font-medium">
                      Recommended
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-400 mt-0.5">{row.description}</p>
              </td>
              {columns.map(col => (
                <td key={col} className="py-3 px-3 text-gray-600 text-sm">
                  {row[col]}
                </td>
              ))}
              <td className="py-3 pl-3">
                <div className={`
                  w-5 h-5 rounded-full border-2 flex items-center justify-center
                  ${selected === row.id ? "border-blue-500" : "border-gray-300"}
                `}>
                  {selected === row.id && (
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <button
        onClick={() => selected && onSelect(selected)}
        disabled={!selected}
        className="mt-4 px-6 py-3 bg-blue-800 text-white rounded-xl
                   font-medium hover:bg-blue-900 disabled:opacity-40 transition-colors"
      >
        Confirm Selection
      </button>
    </div>
  )
}
```

---

## Free Text Input

Used for goal capture and any open-ended questions.

```jsx
function FreeTextInput({ label, placeholder, hint, onSubmit, minLength = 10 }) {
  const [value, setValue] = useState("")
  const [error, setError] = useState(null)
  const tooShort = value.trim().length < minLength

  function handleSubmit() {
    if (tooShort) {
      setError(`Please describe your goal in at least ${minLength} characters.`)
      return
    }
    setError(null)
    onSubmit(value.trim())
  }

  return (
    <div className="mb-6">
      <label className="block text-gray-800 font-medium mb-2">{label}</label>
      {hint && <p className="text-sm text-gray-400 mb-3">{hint}</p>}

      <textarea
        value={value}
        onChange={e => { setValue(e.target.value); setError(null) }}
        placeholder={placeholder}
        rows={4}
        className={`
          w-full px-4 py-3 rounded-xl border text-base text-gray-800
          resize-none focus:outline-none focus:ring-2 focus:ring-blue-400
          transition-shadow leading-relaxed
          ${error ? "border-red-300 bg-red-50" : "border-gray-200 bg-white"}
        `}
      />

      {error && <p className="text-sm text-red-600 mt-1.5">{error}</p>}

      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-gray-400">{value.trim().length} characters</span>
        <button
          onClick={handleSubmit}
          disabled={tooShort}
          className="px-5 py-2.5 bg-blue-800 text-white rounded-xl text-sm
                     font-medium hover:bg-blue-900 disabled:opacity-40 transition-colors"
        >
          Continue →
        </button>
      </div>
    </div>
  )
}
```

---

## Irreversible Action Confirmation Modal

Used before: data deletion, session deletion, privacy transformations,
deployment, and any action that cannot be undone.

```jsx
function ConfirmModal({ title, message, consequence, confirmLabel,
                         onConfirm, onCancel }) {
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center
                    justify-center p-4 z-50 animate-in fade-in duration-150">
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6
                      animate-in zoom-in-95 duration-200">
        <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center
                        justify-center mb-4">
          <AlertTriangle size={24} className="text-amber-600" />
        </div>

        <h2 className="text-xl font-serif text-gray-900 mb-2">{title}</h2>
        <p className="text-gray-600 text-sm leading-relaxed mb-3">{message}</p>

        {consequence && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-5">
            <p className="text-red-700 text-sm font-medium">⚠ This cannot be undone:</p>
            <p className="text-red-600 text-sm mt-1">{consequence}</p>
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl border border-gray-200
                       text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-2.5 rounded-xl bg-red-600 text-white
                       text-sm font-medium hover:bg-red-700 transition-colors"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
```

---

## Chart Display with Caption

Every chart is shown with a mandatory plain English caption.
Charts are never displayed without explanation.

```jsx
function ChartDisplay({ imageBlob, caption, altText, expandable = true }) {
  const [expanded, setExpanded] = useState(false)
  const imageUrl = imageBlob ? URL.createObjectURL(imageBlob) : null
  if (!imageUrl) return null

  return (
    <div className="mb-6">
      <div
        className={`
          relative rounded-xl overflow-hidden border border-gray-100
          bg-white shadow-sm transition-all duration-200
          ${expandable ? "cursor-pointer hover:shadow-md" : ""}
          ${expanded ? "shadow-lg" : ""}
        `}
        onClick={() => expandable && setExpanded(!expanded)}
      >
        <img src={imageUrl} alt={altText} className="w-full h-auto block" loading="lazy" />
        {expandable && (
          <button className="absolute top-3 right-3 p-1.5 bg-white/80 rounded-lg
                             text-gray-400 hover:text-gray-700">
            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        )}
      </div>
      {/* Caption always shown */}
      <p className="mt-3 px-1 text-sm text-gray-600 leading-relaxed">{caption}</p>
    </div>
  )
}
```

---

## Privacy Decision Interface

Each sensitive column gets its own card. All decisions must be made before
the pipeline proceeds.

```jsx
function PrivacyDecisionCard({ column, finding, recommended, options, onDecide }) {
  const [selected, setSelected]   = useState(recommended)
  const [confirmed, setConfirmed] = useState(false)

  if (confirmed) {
    return (
      <div className="p-4 bg-green-50 border border-green-200 rounded-xl
                      flex items-center gap-3 mb-3">
        <CheckCircle size={18} className="text-green-600 flex-shrink-0" />
        <div>
          <span className="font-medium text-green-800">'{column}'</span>
          <span className="text-green-700 text-sm ml-2">
            → {options.find(o => o.id === selected)?.label}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white border border-amber-200 rounded-xl p-5 mb-4 shadow-sm">
      <div className="flex items-start gap-3 mb-4">
        <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center
                        justify-center flex-shrink-0 mt-0.5">
          <Lock size={15} className="text-amber-600" />
        </div>
        <div>
          <p className="font-semibold text-gray-900">
            Column:{" "}
            <code className="font-mono text-blue-700 bg-blue-50 px-1.5 py-0.5
                             rounded text-sm">{column}</code>
          </p>
          <p className="text-sm text-gray-600 mt-1 leading-relaxed">{finding}</p>
        </div>
      </div>

      <div className="space-y-2 mb-4">
        {options.map(opt => (
          <button
            key={opt.id}
            onClick={() => setSelected(opt.id)}
            className={`
              w-full text-left px-4 py-3 rounded-xl border transition-all
              ${selected === opt.id
                ? "border-blue-400 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"}
            `}
          >
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-sm font-medium text-gray-900">{opt.label}</span>
              {opt.id === recommended && (
                <span className="text-xs px-1.5 py-0.5 bg-amber-100
                                 text-amber-700 rounded font-medium">
                  Recommended
                </span>
              )}
            </div>
            <p className="text-xs text-gray-500">{opt.tradeoff}</p>
          </button>
        ))}
      </div>

      <button
        onClick={() => { setConfirmed(true); onDecide(column, selected) }}
        className="px-5 py-2 bg-blue-800 text-white text-sm rounded-xl
                   font-medium hover:bg-blue-900 transition-colors"
      >
        Confirm
      </button>
    </div>
  )
}
```

---

## "Why?" Follow-Up Mechanism

The user can always ask for more explanation on any Orchestrator message.

```jsx
function OrchestratorMessageWithFollowUp({ message, stageContext, onFollowUp }) {
  const [open, setOpen]     = useState(false)
  const [question, setQ]    = useState("")
  const [loading, setLoad]  = useState(false)

  async function handleAsk() {
    if (!question.trim()) return
    setLoad(true)
    await onFollowUp(question, stageContext)
    setQ(""); setOpen(false); setLoad(false)
  }

  return (
    <div className="mb-6">
      <div className="flex gap-4">
        <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-blue-800
                        flex items-center justify-center">
          <Brain size={18} className="text-white" />
        </div>
        <div className="flex-1 bg-white rounded-2xl rounded-tl-sm
                        border border-gray-100 shadow-sm px-5 py-4">
          <p className="text-sm font-medium text-blue-800 mb-2">Pipeline Guide</p>
          <div className="text-gray-700 text-base leading-relaxed whitespace-pre-line">
            {message}
          </div>
          <button
            onClick={() => setOpen(!open)}
            className="mt-3 text-sm text-blue-600 hover:text-blue-800
                       flex items-center gap-1.5 transition-colors"
          >
            <HelpCircle size={14} />
            {open ? "Never mind" : "I'd like more detail"}
          </button>
        </div>
      </div>

      {open && (
        <div className="mt-2 ml-13 pl-4 animate-in fade-in slide-in-from-top-1 duration-200">
          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={e => setQ(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleAsk()}
              placeholder="What would you like to know more about?"
              autoFocus
              className="flex-1 px-4 py-2.5 text-sm rounded-xl border border-gray-200
                         focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
            <button
              onClick={handleAsk}
              disabled={!question.trim() || loading}
              className="px-4 py-2.5 bg-blue-700 text-white text-sm rounded-xl
                         disabled:opacity-40 hover:bg-blue-800 transition-colors"
            >
              {loading ? "…" : "Ask"}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
```

---

## Final Report Display

Assembled from all stage `report_section` objects. Rendered as a scrollable
narrative — not a dashboard.

```jsx
function FinalReport({ report, onDownload }) {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2
                        bg-green-50 text-green-700 rounded-full text-sm
                        font-medium mb-4">
          <CheckCircle size={16} /> Pipeline Complete
        </div>
        <h1 className="text-3xl font-serif text-gray-900 mb-2">
          Your Model Report
        </h1>
        <p className="text-gray-500 text-sm">
          A plain English record of every decision made
        </p>
      </div>

      <div className="space-y-6">
        {report.sections.map((section, i) => (
          <div
            key={i}
            className="bg-white rounded-2xl border border-gray-100
                       shadow-sm p-6 animate-in fade-in
                       slide-in-from-bottom-2 duration-300"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-7 h-7 rounded-full bg-blue-800 text-white
                              text-xs font-bold flex items-center
                              justify-center flex-shrink-0">
                {i + 1}
              </div>
              <h2 className="text-xl font-serif text-gray-900">{section.title}</h2>
            </div>

            <p className="text-gray-700 leading-relaxed mb-3">{section.summary}</p>

            {section.decision_made && (
              <div className="bg-gray-50 rounded-xl px-4 py-3 text-sm mb-2">
                <span className="font-medium text-gray-600">Decision: </span>
                <span className="text-gray-700">{section.decision_made}</span>
              </div>
            )}

            {section.alternatives_considered && (
              <div className="bg-gray-50 rounded-xl px-4 py-3 text-sm mb-2">
                <span className="font-medium text-gray-600">Alternatives considered: </span>
                <span className="text-gray-700">{section.alternatives_considered}</span>
              </div>
            )}

            {section.why_this_matters && (
              <p className="text-sm text-gray-500 italic mt-3 leading-relaxed">
                {section.why_this_matters}
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="mt-10 flex justify-center">
        <button
          onClick={onDownload}
          className="flex items-center gap-2 px-6 py-3 bg-blue-800
                     text-white rounded-xl font-medium hover:bg-blue-900
                     transition-colors"
        >
          <Download size={18} /> Download Report as PDF
        </button>
      </div>
    </div>
  )
}
```

---

## Interaction Pattern Reference

| Situation | Component | Key behaviour |
|---|---|---|
| Orchestrator speaks | `OrchestratorMessage` | Animates in, "Tell me more" always available |
| Choose one option | `SingleChoiceDecision` | Recommended pre-selected, tradeoffs visible |
| Compare options | `ComparisonTable` | Side-by-side properties, one click to select |
| Open question | `FreeTextInput` | Min length enforced, submits on Enter |
| Irreversible action | `ConfirmModal` | Consequence stated explicitly, red confirm |
| Agent running | `AgentRunning` (shell) | Informative message, never a blank screen |
| Warning | `AlertBanner` type="warning" | Always actionable |
| Hard stop | `AlertBanner` type="error" | Clear next step shown |
| Chart result | `ChartDisplay` | Caption always shown below, expandable |
| Privacy column | `PrivacyDecisionCard` | All options listed, recommended highlighted |
| Pipeline complete | `FinalReport` | Narrative sections, PDF download |

---

## Animation Principles

- Orchestrator messages: fade + slide in from below, 300ms ease-out
- Decision cards: appear 100ms after the message
- Confirmed decisions: collapse to a summary row, 200ms
- Stage transitions: fade out 150ms, fade in 250ms
- Report sections: stagger in at 60ms intervals
- Never block interaction during animations — they are decorative, not gating

---

## Reference Files

- `references/interaction-flows.md` — full interaction flow diagrams for each stage
