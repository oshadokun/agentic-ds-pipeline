# Responsive Design Reference

## Breakpoints

```css
/* Mobile first */
sm:   640px   /* Large phone / small tablet */
md:   768px   /* Tablet portrait */
lg:   1024px  /* Tablet landscape / small desktop */
xl:   1280px  /* Desktop */
2xl:  1536px  /* Large desktop */
```

## Layout at Each Breakpoint

### Mobile (< 768px)
- No sidebar — progress shown as a compact top bar
- Top bar: current stage name + "Step 3 of 13"
- Tap top bar to expand full stage list as a bottom sheet
- Full-width content area
- Navigation buttons fixed to bottom of screen
- Charts scaled to full width
- Tables horizontally scrollable

### Tablet (768px – 1023px)
- Sidebar collapses to icons-only (40px wide)
- Hover/tap icon to see stage label tooltip
- Content area takes remaining width
- Charts at 100% content width

### Desktop (≥ 1024px)
- Full sidebar (240px wide) always visible
- Content area: max-width 800px, centred in remaining space
- Charts at content width (max 700px)

## Touch Interactions

- All buttons minimum 44×44px touch target
- Swipe left/right on chart carousel (EDA view)
- Pull-to-refresh on monitoring view
- Long press on completed stage to see summary tooltip

## Chart Responsiveness

All charts from the backend are served as PNG images. Display rules:
- Max width: 100% of content area
- Never overflow their container
- On mobile: stacked vertically, never side by side
- Caption always below the chart, never overlapping

## Typography Scaling

```css
/* Headings scale down on mobile */
.stage-title {
  font-size: clamp(1.25rem, 4vw, 1.875rem);
}

/* Body text stays readable */
.body-text {
  font-size: clamp(0.9375rem, 2vw, 1rem);
  line-height: 1.6;
}
```
