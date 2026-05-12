import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000', // Update this to match your Frappe port
        changeOrigin: true,
        secure: false,
        headers: { 'Host': 'mysite2.localhost' },
      }
    }
  }
})
