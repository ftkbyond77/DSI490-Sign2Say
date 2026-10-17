import { defineConfig } from "vite";

// dev: `npm run dev` proxies /api to a local reverse-api (uvicorn on :8091); production: nginx serves dist/ and proxies /api
export default defineConfig({
  server: { port: 5181, proxy: { "/api": "http://127.0.0.1:8091", "/avatars": "http://127.0.0.1:8091" } },
  build: { target: "es2022", chunkSizeWarningLimit: 900, assetsInlineLimit: 0 },
});
