/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fuchsia: {
          500: '#d946ef',
          600: '#c026d3',
        },
        pink: {
          500: '#ec4899',
          400: '#f472b6',
        },
      },
      keyframes: {
        fadeIn: {
          'from': { opacity: '0', transform: 'translateY(10px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        typing: {
          '0%, 60%, 100%': { transform: 'translateY(0)', opacity: '0.5' },
          '30%': { transform: 'translateY(-10px)', opacity: '1' },
        },
      },
      animation: {
        fadeIn: 'fadeIn 0.3s ease-in',
        typing: 'typing 1.4s infinite ease-in-out',
      },
    },
  },
  plugins: [],
}
