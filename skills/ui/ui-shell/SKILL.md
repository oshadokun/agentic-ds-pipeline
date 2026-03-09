---
name: ui-shell
description: >
  Responsible for building the web interface of the guided data science pipeline
  app. Builds the overall application structure, navigation, layout, progress
  tracking, stage display, and visual design. Always read before building any
  part of the frontend. Covers technology stack selection, component architecture,
  design system, accessibility, responsiveness, and how pipeline stages map to
  UI views. Works in conjunction with the ui-interaction skill which handles the
  dynamic decision and conversation elements. Trigger when any of the following
  are mentioned: "build the UI", "build the frontend", "web interface",
  "application layout", "progress bar", "navigation", "stage view", "dashboard",
  "design the app", or any request to create or modify the visual structure
  of the pipeline application.
---

# UI Shell Skill

The UI Shell is the frame of the application — the structure everything else
lives inside. It defines the technology stack, the visual language, the layout
system, the navigation, and how the 13 pipeline stages are represented to the
user.

The audience is non-technical. The interface must feel calm, guided, and
trustworthy — not like a developer tool. Every design decision should reduce
cognitive load and build confidence.

---

## Design Principles

Before writing a single line of code, commit to these principles:

**1. Calm confidence**
This is not a dashboard for data scientists. It is a guided journey for someone
who may never have built a model before. The interface should feel like a
knowledgeable guide — never overwhelming, never condescending.

**2. One thing at a time**
Never show two stages simultaneously. The user is always in one place. Progress
is visible but not distracting. The next step is always clear.

**3. Progress is earned**
Completed stages feel genuinely done. The progress indicator is not decorative —
it reflects real work. Each completed stage is a small win worth acknowledging.

**4. Data stays private**
The UI should visually reinforce privacy — data never appears in URLs, session
identifiers are not exposed, sensitive column names are handled carefully.

**5. Plain English everywhere**
No technical labels in the navigation. "Loading Your Data" not "Ingestion".
"Checking Data Quality" not "Validation". "Training the Model" not "Fitting".

---

## Technology Stack

**Framework:** React (with Vite for development)
- Component-based — each pipeline stage is a self-contained view
- Fast development iteration
- Strong ecosystem for data display (charts, tables)

**Styling:** Tailwind CSS
- Utility-first — fast to iterate
- Consistent spacing and typography system
- Easy responsive design

**Charts:** Recharts
- React-native chart library
- Clean default aesthetics
- Easy to customise for non-technical display

**State Management:** React Context + useReducer
- Session state held in context
- No external state library needed at this scale

**Backend Communication:** Fetch API
- POST to the local FastAPI backend (session management, pipeline execution)
- WebSocket for long-running stage progress updates

**Icons:** Lucide React

---

## Visual Design System

### Colour Palette

```css
:root {
  /* Primary — deep, trustworthy blue */
  --colour-primary:         #1B3A5C;
  --colour-primary-light:   #2E6099;
  --colour-primary-pale:    #EBF3FB;

  /* Accent — warm amber for actions and highlights */
  --colour-accent:          #D97706;
  --colour-accent-light:    #FEF3C7;

  /* Status colours */
  --colour-success:         #065F46;
  --colour-success-bg:      #ECFDF5;
  --colour-warning:         #92400E;
  --colour-warning-bg:      #FFFBEB;
  --colour-error:           #991B1B;
  --colour-error-bg:        #FEF2F2;

  /* Neutrals */
  --colour-text-primary:    #111827;
  --colour-text-secondary:  #6B7280;
  --colour-text-muted:      #9CA3AF;
  --colour-border:          #E5E7EB;
  --colour-surface:         #F9FAFB;
  --colour-white:           #FFFFFF;

  /* Stage states */
  --colour-stage-complete:  #065F46;
  --colour-stage-active:    #1B3A5C;
  --colour-stage-pending:   #9CA3AF;
}
```

### Typography

```css
/* Heading font — confident, readable, professional */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --font-heading: 'DM Serif Display', Georgia, serif;
  --font-body:    'DM Sans', system-ui, sans-serif;
  --font-mono:    'JetBrains Mono', 'Courier New', monospace;
}

/* Scale */
--text-xs:   0.75rem;    /* 12px — labels, metadata */
--text-sm:   0.875rem;   /* 14px — secondary text */
--text-base: 1rem;       /* 16px — body */
--text-lg:   1.125rem;   /* 18px — lead text */
--text-xl:   1.25rem;    /* 20px — section headings */
--text-2xl:  1.5rem;     /* 24px — page headings */
--text-3xl:  1.875rem;   /* 30px — stage title */
```

### Spacing System
Follow an 8px base grid. All margins, padding, and gaps should be multiples of 8px.
Use Tailwind spacing scale: `p-2` (8px), `p-4` (16px), `p-6` (24px), `p-8` (32px).

### Border Radius
- Small elements (badges, tags): `rounded` (4px)
- Cards and panels: `rounded-xl` (12px)
- Modals and large containers: `rounded-2xl` (16px)

### Shadows
```css
--shadow-sm:  0 1px 3px rgba(0,0,0,0.08);
--shadow-md:  0 4px 16px rgba(0,0,0,0.08);
--shadow-lg:  0 8px 32px rgba(0,0,0,0.12);
```

---

## Application Layout

```
┌─────────────────────────────────────────────────────────┐
│  HEADER — App name + session info + privacy indicator   │
├────────────┬────────────────────────────────────────────┤
│            │                                            │
│  PROGRESS  │           MAIN CONTENT AREA               │
│  SIDEBAR   │                                            │
│            │   Stage title + explanation                │
│  13 stages │   Stage-specific content                   │
│  listed    │   Decision / action area                   │
│  vertically│   Navigation (Back / Continue)             │
│            │                                            │
└────────────┴────────────────────────────────────────────┘
```

**Responsive:**
- Desktop (> 1024px): Sidebar + main content side by side
- Tablet (768–1024px): Sidebar collapses to top progress bar
- Mobile (< 768px): Full-width, progress bar at top, swipeable

---

## Component Architecture

```
App
├── AppShell
│   ├── Header
│   │   ├── AppLogo
│   │   ├── SessionInfo
│   │   └── PrivacyBadge
│   ├── ProgressSidebar
│   │   ├── ProgressHeader (goal summary)
│   │   ├── StageList
│   │   │   └── StageItem (× 13)
│   │   └── SessionActions (save, resume, delete)
│   └── MainContent
│       ├── StageHeader
│       ├── StageBody (dynamic — loads current stage component)
│       └── StageNavigation
│
├── Views (one per stage)
│   ├── GoalCaptureView
│   ├── IngestionView
│   ├── ValidationView
│   ├── EDAView
│   ├── CleaningView
│   ├── FeatureEngineeringView
│   ├── NormalisationView
│   ├── SplittingView
│   ├── TrainingView
│   ├── EvaluationView
│   ├── TuningView
│   ├── ExplainabilityView
│   ├── DeploymentView
│   └── MonitoringView
│
├── Shared Components
│   ├── DecisionCard        ← presents a decision with alternatives
│   ├── ExplanationPanel    ← plain English explanation block
│   ├── ProgressChart       ← charts from EDA/evaluation
│   ├── DataPreviewTable    ← shows first n rows of data
│   ├── StatusBadge         ← complete / in progress / pending
│   ├── AlertBanner         ← warnings, errors, advisories
│   ├── ConfirmModal        ← for irreversible actions
│   └── LoadingState        ← shown during agent execution
│
└── Contexts
    ├── SessionContext      ← holds session.json state
    └── PipelineContext     ← controls stage flow
```

---

## Stage Navigation Labels

These are the plain English labels shown in the sidebar.
Never use technical names in the UI.

```javascript
const STAGE_LABELS = {
  ingestion:           { label: "Load Your Data",          icon: "Upload" },
  validation:          { label: "Check Data Quality",      icon: "ShieldCheck" },
  eda:                 { label: "Explore Your Data",        icon: "BarChart2" },
  cleaning:            { label: "Clean Your Data",          icon: "Sparkles" },
  feature_engineering: { label: "Prepare Features",         icon: "Wrench" },
  normalisation:       { label: "Scale Your Data",          icon: "Sliders" },
  splitting:           { label: "Divide Your Data",         icon: "Scissors" },
  training:            { label: "Train the Model",          icon: "Brain" },
  evaluation:          { label: "Measure Performance",      icon: "Target" },
  tuning:              { label: "Fine-Tune the Model",      icon: "SlidersHorizontal" },
  explainability:      { label: "Understand the Model",     icon: "Lightbulb" },
  deployment:          { label: "Deploy the Model",         icon: "Rocket" },
  monitoring:          { label: "Monitor Performance",      icon: "Activity" }
}
```

---

## Progress Sidebar Component

```jsx
// StageItem.jsx
function StageItem({ stage, status, label, icon, isActive, onClick }) {
  const statusStyles = {
    complete:    "text-green-700 bg-green-50 border-green-200",
    in_progress: "text-blue-800 bg-blue-50 border-blue-300 font-semibold",
    pending:     "text-gray-400 bg-white border-gray-100"
  }

  const statusIcons = {
    complete:    <CheckCircle size={16} className="text-green-600" />,
    in_progress: <Circle size={16} className="text-blue-600 animate-pulse" />,
    pending:     <Circle size={16} className="text-gray-300" />
  }

  return (
    <button
      onClick={status === "complete" ? onClick : undefined}
      disabled={status === "pending"}
      className={`
        w-full flex items-center gap-3 px-4 py-3 rounded-xl border
        transition-all duration-200 text-left text-sm
        ${statusStyles[status]}
        ${isActive ? "ring-2 ring-blue-400 ring-offset-1" : ""}
        ${status === "complete" ? "cursor-pointer hover:shadow-sm" : "cursor-default"}
      `}
    >
      {statusIcons[status]}
      <span className="flex-1">{label}</span>
      {status === "complete" && (
        <ChevronRight size={14} className="text-gray-400" />
      )}
    </button>
  )
}
```

---

## Stage Header Component

Every stage opens with the same pattern — title, plain English explanation,
and what the user will be asked to do.

```jsx
function StageHeader({ title, explanation, whatToExpect }) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center
                        justify-center text-blue-700">
          {/* Stage icon */}
        </div>
        <h1 className="text-2xl font-serif text-gray-900">{title}</h1>
      </div>

      <p className="text-gray-600 text-base leading-relaxed mb-4">
        {explanation}
      </p>

      {whatToExpect && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl
                        px-4 py-3 text-sm text-amber-800">
          <span className="font-medium">What to expect: </span>
          {whatToExpect}
        </div>
      )}
    </div>
  )
}
```

---

## Loading State — During Agent Execution

When a pipeline agent is running, show an informative loading state.
Never show a blank screen or a generic spinner.

```jsx
function AgentRunning({ stageName, message, progress }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-8">
      <div className="relative mb-6">
        <div className="w-16 h-16 rounded-full border-4 border-blue-100
                        border-t-blue-600 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Brain size={24} className="text-blue-600" />
        </div>
      </div>

      <h2 className="text-xl font-serif text-gray-900 mb-2">
        {stageName}
      </h2>
      <p className="text-gray-500 text-sm text-center max-w-sm">
        {message}
      </p>

      {progress && (
        <div className="w-64 mt-6">
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-1 text-center">
            {progress}% complete
          </p>
        </div>
      )}
    </div>
  )
}
```

---

## Alert Banner Component

Used for warnings, advisories, and hard stops.

```jsx
function AlertBanner({ type, title, message, action }) {
  const styles = {
    error:   "bg-red-50 border-red-300 text-red-800",
    warning: "bg-amber-50 border-amber-300 text-amber-800",
    info:    "bg-blue-50 border-blue-300 text-blue-800",
    success: "bg-green-50 border-green-300 text-green-700"
  }
  const icons = {
    error: <XCircle size={20} />,
    warning: <AlertTriangle size={20} />,
    info: <Info size={20} />,
    success: <CheckCircle size={20} />
  }

  return (
    <div className={`flex gap-3 p-4 rounded-xl border ${styles[type]} mb-4`}>
      <div className="flex-shrink-0 mt-0.5">{icons[type]}</div>
      <div className="flex-1">
        {title && <p className="font-semibold text-sm mb-1">{title}</p>}
        <p className="text-sm leading-relaxed">{message}</p>
        {action && (
          <button className="mt-2 text-sm font-medium underline
                             underline-offset-2 hover:no-underline">
            {action.label}
          </button>
        )}
      </div>
    </div>
  )
}
```

---

## Session List View

Shown when no session is active — allows resuming or starting fresh.

```jsx
function SessionListView({ sessions, onResume, onNew }) {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-8">
      <div className="max-w-2xl w-full">
        <div className="mb-10 text-center">
          <h1 className="text-3xl font-serif text-gray-900 mb-3">
            Data Science Pipeline
          </h1>
          <p className="text-gray-500">
            Build, train, and deploy a machine learning model — step by step,
            in plain English.
          </p>
        </div>

        <button
          onClick={onNew}
          className="w-full mb-6 py-4 px-6 bg-blue-800 text-white rounded-2xl
                     font-medium hover:bg-blue-900 transition-colors
                     flex items-center justify-center gap-3 text-lg"
        >
          <Plus size={22} />
          Start a New Pipeline
        </button>

        {sessions.length > 0 && (
          <>
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-gray-200" />
              <span className="text-sm text-gray-400">or continue a previous session</span>
              <div className="flex-1 h-px bg-gray-200" />
            </div>

            <div className="space-y-3">
              {sessions.map(s => (
                <SessionCard key={s.session_id} session={s} onResume={onResume} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function SessionCard({ session, onResume }) {
  return (
    <div
      onClick={() => onResume(session.session_id)}
      className="w-full p-4 bg-white rounded-xl border border-gray-200
                 hover:border-blue-300 hover:shadow-md transition-all
                 cursor-pointer text-left"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-gray-900 truncate">
            {session.goal || "No goal recorded"}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {session.progress} · Last updated{" "}
            {new Date(session.last_updated).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs px-2 py-1 bg-blue-50 text-blue-700
                           rounded-full font-medium">
            {session.last_stage?.replace(/_/g, " ") || "Not started"}
          </span>
          <ChevronRight size={16} className="text-gray-400" />
        </div>
      </div>
    </div>
  )
}
```

---

## Backend API Integration

The frontend communicates with the local FastAPI backend for all pipeline operations.

```javascript
// api.js — centralised API client

const BASE_URL = "http://localhost:8001"  // Pipeline management backend
                                           // (different from model API on 8000)

export const api = {
  // Sessions
  listSessions:   ()             => fetch(`${BASE_URL}/sessions`).then(r => r.json()),
  createSession:  (goal)         => post(`/sessions`, { goal }),
  loadSession:    (id)           => fetch(`${BASE_URL}/sessions/${id}`).then(r => r.json()),
  deleteSession:  (id)           => del(`/sessions/${id}`),

  // Pipeline
  runStage:       (id, stage, decisions) =>
    post(`/sessions/${id}/stages/${stage}/run`, { decisions }),
  getStageResult: (id, stage)    =>
    fetch(`${BASE_URL}/sessions/${id}/stages/${stage}/result`).then(r => r.json()),

  // Files
  uploadFile:     (id, file)     => upload(`/sessions/${id}/data`, file),
  getChart:       (id, path)     =>
    fetch(`${BASE_URL}/sessions/${id}/charts?path=${path}`).then(r => r.blob()),

  // Report
  getReport:      (id)           =>
    fetch(`${BASE_URL}/sessions/${id}/report`).then(r => r.json())
}

function post(path, body) {
  return fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  }).then(r => r.json())
}
```

---

## Accessibility Requirements

- All interactive elements must have clear focus indicators
- Colour is never the only means of conveying status — always pair with an icon or text
- All images and charts must have descriptive alt text
- Font size never below 14px for body text
- Minimum contrast ratio 4.5:1 for all text
- All decisions must be completable without a mouse (keyboard navigable)
- Loading states must be announced to screen readers via aria-live

---

## File Structure

```
frontend/
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── api.js
│   ├── components/
│   │   ├── shell/
│   │   │   ├── AppShell.jsx
│   │   │   ├── Header.jsx
│   │   │   ├── ProgressSidebar.jsx
│   │   │   └── StageNavigation.jsx
│   │   ├── shared/
│   │   │   ├── AlertBanner.jsx
│   │   │   ├── AgentRunning.jsx
│   │   │   ├── DecisionCard.jsx
│   │   │   ├── ExplanationPanel.jsx
│   │   │   ├── DataPreviewTable.jsx
│   │   │   ├── ConfirmModal.jsx
│   │   │   └── StatusBadge.jsx
│   │   └── stages/
│   │       ├── GoalCaptureView.jsx
│   │       ├── IngestionView.jsx
│   │       ├── ValidationView.jsx
│   │       └── ... (one per stage)
│   ├── contexts/
│   │   ├── SessionContext.jsx
│   │   └── PipelineContext.jsx
│   └── styles/
│       └── globals.css
├── public/
├── index.html
├── vite.config.js
└── package.json
```

---

## Reference Files

- `references/stage-views.md` — detailed layout specifications for each of the 13 stage views
- `references/responsive-design.md` — breakpoints, mobile layouts, and touch interactions
