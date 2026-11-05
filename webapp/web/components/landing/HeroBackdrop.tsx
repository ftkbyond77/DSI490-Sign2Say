"use client";
// ShaderGradient (ruucm/shadergradient) as the hero backdrop: a slow water plane in the brand teal and white.
import { ShaderGradient, ShaderGradientCanvas } from "@shadergradient/react";

export default function HeroBackdrop({ animate }: { animate: boolean }) {
  return (
    <ShaderGradientCanvas style={{ position: "absolute", inset: 0 }} pixelDensity={1} fov={45} pointerEvents="none" lazyLoad={false}>
      <ShaderGradient
        control="props" type="waterPlane" animate={animate ? "on" : "off"}
        color1="#2aafb3" color2="#9fe6e8" color3="#dff7f7"
        uSpeed={0.18} uStrength={3.2} uDensity={1.2} uFrequency={0} uAmplitude={0}
        brightness={1.15} reflection={0.12} lightType="3d" envPreset="city" grain="off"
        cAzimuthAngle={170} cPolarAngle={72} cDistance={4.4} cameraZoom={1}
        positionX={0} positionY={0.9} positionZ={-0.3} rotationX={45} rotationY={0} rotationZ={0}
      />
    </ShaderGradientCanvas>
  );
}
