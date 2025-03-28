import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import path from "path";
// Use dynamic import for vite-plugin-checker
// import checker from 'vite-plugin-checker'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    // Temporarily disable checker due to ESM compatibility issues
    // checker({
    //   typescript: true,
    // }),
  ],
  root: "./",
  publicDir: "public",
  build: {
    outDir: "dist",
  },
  resolve: {
    alias: {
      "@": "/src",
    },
  },
  server: {
    port: 3000,
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
