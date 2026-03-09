/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ["DM Serif Display", "Georgia", "serif"],
        sans:  ["DM Sans", "system-ui", "sans-serif"],
        mono:  ["JetBrains Mono", "Courier New", "monospace"]
      },
      colors: {
        brand: {
          DEFAULT: "#1B3A5C",
          light:   "#2E6099",
          pale:    "#EBF3FB"
        }
      },
      boxShadow: {
        sm:  "0 1px 3px rgba(0,0,0,0.08)",
        md:  "0 4px 16px rgba(0,0,0,0.08)",
        lg:  "0 8px 32px rgba(0,0,0,0.12)"
      },
      animation: {
        "fade-in":       "fadeIn 0.3s ease-out",
        "slide-up":      "slideUp 0.3s ease-out",
        "scale-in":      "scaleIn 0.2s ease-out"
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" },              to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(8px)" },
                   to:   { opacity: "1", transform: "translateY(0)" } },
        scaleIn: { from: { opacity: "0", transform: "scale(0.95)" },
                   to:   { opacity: "1", transform: "scale(1)" } }
      }
    }
  },
  plugins: []
}
