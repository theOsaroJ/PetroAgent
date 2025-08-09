/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        petro: {
          bg: "#0b1220",
          panel: "#0f172a",
          accent: "#00e0b8",
          text: "#e5f4ff",
          muted: "#9fb6c8"
        }
      }
    },
  },
  plugins: [],
}
