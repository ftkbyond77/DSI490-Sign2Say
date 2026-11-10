import "./landing.css";
import Closing from "@/components/landing/Closing";
import Hero from "@/components/landing/Hero";
import Nav from "@/components/landing/Nav";
import Services from "@/components/landing/Services";
import SignDemo from "@/components/landing/SignDemo";
import SmoothScroll from "@/components/landing/SmoothScroll";

// Landing page: ① hero → ② looping demo clip (Human → Extracting → Predicting → Speech) → ③ one row per service → ④ closing.
export default function Home() {
  return (
    <div className="lp">
      <SmoothScroll />
      <Nav />
      <main>
        <Hero />
        <SignDemo />
        <Services />
        <Closing />
      </main>
    </div>
  );
}
