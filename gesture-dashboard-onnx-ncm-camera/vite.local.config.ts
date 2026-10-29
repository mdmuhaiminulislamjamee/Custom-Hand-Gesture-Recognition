import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';

// Windows-local development intentionally omits the Sites/Cloudflare worker
// plugins. The production `vite.config.ts` retains them for a future hosted
// build, while this configuration avoids launching workerd for localhost.
export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  plugins: [vinext()],
  // Vite 8 uses Rolldown for dependency pre-bundling. On memory-constrained
  // Windows PCs that optimizer can fail before the local server is ready.
  // This app's browser dependencies are ESM, so serve them directly instead.
  optimizeDeps: {
    noDiscovery: true,
    include: [],
  },
  environments: {
    client: { optimizeDeps: { noDiscovery: true, include: [] } },
    rsc: { optimizeDeps: { noDiscovery: true, include: [] } },
    ssr: { optimizeDeps: { noDiscovery: true, include: [] } },
  },
  server: {
    host: '127.0.0.1',
    port: 3200,
    strictPort: true,
    watch: {
      ignored: ['**/.venv/**', '**/.runtime/**', '**/feedback/**', '**/models/**', '**/build/**', '**/.dart_tool/**'],
    },
  },
});
