import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '..', 'IRIS_');
  return {plugins: [react()], server: {port: Number(env.IRIS_FRONTEND_PORT || 5187), strictPort: true,
    proxy: {'/api': `http://127.0.0.1:${env.IRIS_PORT || 8014}`}}, build: {outDir: 'dist'}};
});
