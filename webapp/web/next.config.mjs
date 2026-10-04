/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // dev only (npm run dev without the gateway): forward /api and /tts to local services
  async rewrites() {
    if (process.env.NODE_ENV === "production") return [];
    return [
      { source: "/api/:path*", destination: `${process.env.SIGN_API ?? "http://localhost:8000"}/:path*` },
      { source: "/tts/:path*", destination: `${process.env.TTS_API ?? "http://localhost:8001"}/:path*` },
    ];
  },
};
export default nextConfig;
