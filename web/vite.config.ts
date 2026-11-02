import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
export default defineConfig(({ mode }) => ({
  define: { __HOSTED_DEMO__: JSON.stringify(mode === "hosted") },
  plugins: [
    react(),
    {
      name: "public-demo-artifact",
      generateBundle() {
        if (mode !== "hosted") return;
        const source = JSON.parse(
          readFileSync(
            new URL("../runs/demo/console.json", import.meta.url),
            "utf8",
          ),
        );
        // Publish the same fields as /api/experiment, never the truth records.
        const data = Object.fromEntries(
          [
            "dataset",
            "projection",
            "policy",
            "flows",
            "metrics",
            "windows",
          ].map((key) => [key, source[key]]),
        );
        if (!Array.isArray(data.flows) || !data.flows.length)
          throw Error("Missing demo flows");
        this.emitFile({
          type: "asset",
          fileName: "demo/console.json",
          source: JSON.stringify(data),
        });
      },
    },
  ],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
}));
