"use client";
// liquid-logo (paper-design/liquid-logo, now shipped as the LiquidMetal shader): the hand mark rendered as moving liquid chrome.
import { LiquidMetal } from "@paper-design/shaders-react";

export default function LiquidMark({ size = 128, speed = 0.6 }: { size?: number; speed?: number }) {
  return (
    <LiquidMetal
      image="/landing/hand-mask.svg" style={{ width: size, height: size }}
      colorBack="#00000000" colorTint="#2aafb3" shape="none"
      repetition={2.4} softness={0.55} shiftRed={0.25} shiftBlue={0.35} distortion={0.12} contour={0.55} angle={60}
      speed={speed} scale={0.9} fit="contain"
    />
  );
}
