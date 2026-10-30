/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // dev only (npm run dev without the gateway): forward the APIs to local services, the same paths the gateway serves
  //   /api → sign-api · /tts → tts · /t2s → reverse-api /api · /avatars → reverse-api (character files)
  async rewrites() {
    if (process.env.NODE_ENV === "production") return [];
    const t2s = process.env.T2S_API ?? "http://localhost:8091";
    return [
      { source: "/api/:path*", destination: `${process.env.SIGN_API ?? "http://localhost:8000"}/:path*` },
      { source: "/tts/:path*", destination: `${process.env.TTS_API ?? "http://localhost:8001"}/:path*` },
      { source: "/t2s/:path*", destination: `${t2s}/api/:path*` },
      { source: "/avatars/:path*", destination: `${t2s}/avatars/:path*` },
    ];
  },
};
export default nextConfig;
