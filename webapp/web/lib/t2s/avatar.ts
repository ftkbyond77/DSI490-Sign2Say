// The signing avatar: VRM loading, half-body framing, and applying one pose (local quaternions of normalized humanoid bones +
// VRoid face morphs). The motion arrays come from the API in the model frame of the VRM 0.x rig (see t2s/motion/rig.py), already
// fitted to the chosen character's body, so a value is written to the normalized bone as is (VRM 1.0 characters: mirrored x / z).
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

export type Pose = { q: Float32Array; m: Float32Array }; // q [B*4] (x,y,z,w), m [M]

// the motion names thumbs the VRM 0.x way (Proximal · Intermediate · Distal); three-vrm uses the VRM 1.0 names for every model
// (it renames VRM 0.x thumbs on load), so the same three joints are Metacarpal · Proximal · Distal here (t2s/motion/rig.py TO_VRM1)
const TO_VRM1: Record<string, string> = {
  leftThumbProximal: "leftThumbMetacarpal", leftThumbIntermediate: "leftThumbProximal",
  rightThumbProximal: "rightThumbMetacarpal", rightThumbIntermediate: "rightThumbProximal",
};

// VRoid face morph → standard VRM expression, for characters that do not carry VRoid's Fcl_* morphs
const FACE_EXPR: Record<string, string> = {
  Fcl_EYE_Close_L: "blinkLeft", Fcl_EYE_Close_R: "blinkRight", Fcl_MTH_A: "aa", Fcl_MTH_I: "ih", Fcl_MTH_U: "ou", Fcl_MTH_O: "oh",
  Fcl_BRW_Surprised: "surprised", Fcl_BRW_Angry: "angry", Fcl_BRW_Sorrow: "sad", Fcl_MTH_Fun: "happy",
};

export class Avatar {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(24, 1, 0.05, 20);
  vrm: VRM | null = null;
  private bones: (THREE.Object3D | null)[] = [];
  private morphTargets: { mesh: THREE.Mesh; idx: number }[][] = [];
  private lookTarget = new THREE.Object3D();
  private host: HTMLElement;
  private ro: ResizeObserver;
  framing: "half" | "close" = "half";

  constructor(host: HTMLElement) {
    this.host = host;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x8a8a9a, 1.6);
    const key = new THREE.DirectionalLight(0xffffff, 1.9);
    key.position.set(0.6, 1.8, 1.6);
    const rim = new THREE.DirectionalLight(0xdfe8ff, 0.8);
    rim.position.set(-1.2, 1.6, -1.0);
    this.scene.add(hemi, key, rim, this.lookTarget);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
  }

  /** download + prepare a VRM without showing it (the current character stays on stage meanwhile) */
  async fetch(url: string, onProgress?: (f: number) => void): Promise<VRM> {
    const loader = new GLTFLoader();
    loader.register((p) => new VRMLoaderPlugin(p));
    const gltf = await loader.loadAsync(url, (e) => onProgress && e.total && onProgress(e.loaded / e.total));
    const vrm = gltf.userData.vrm as VRM;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(vrm); // scene faces the camera (+Z); normalized bones stay in the model frame
    vrm.scene.traverse((o) => ((o as THREE.Mesh).frustumCulled = false));
    return vrm;
  }

  /** put a fetched VRM on stage (the previous one is removed and its GPU memory freed); call bind() right after */
  show(vrm: VRM) {
    const old = this.vrm;
    this.scene.add(vrm.scene);
    this.vrm = vrm;
    if (vrm.lookAt) vrm.lookAt.target = this.lookTarget;
    // hair / cloth springs start from the on-stage (rotated) pose, not from where the scene sat before it was attached
    vrm.scene.updateMatrixWorld(true);
    vrm.springBoneManager?.reset();
    // VRM 1.0 faces +Z in its own frame: the motion (VRM 0.x frame) is mirrored about Y by negating x / z of every rotation
    this.v1 = vrm.meta?.metaVersion === "1";
    // framing follows the body: head height relative to the library avatar (AvatarSample_C, head joint 1.561 m)
    const head = vrm.humanoid.getNormalizedBoneNode("head");
    this.bodyScale = head ? THREE.MathUtils.clamp(head.getWorldPosition(new THREE.Vector3()).y / 1.561, 0.4, 2.5) : 1;
    this.frame();
    if (old) {
      this.scene.remove(old.scene);
      VRMUtils.deepDispose(old.scene);
    }
  }

  async load(url: string, onProgress?: (f: number) => void) {
    this.show(await this.fetch(url, onProgress));
  }

  private v1 = false;
  private bodyScale = 1;
  private expr: (string | null)[] = [];

  /** bind the bone / morph names of a motion payload (again after a character change) */
  bind(boneNames: string[], morphNames: string[]) {
    const h = this.vrm!.humanoid;
    this.bones = boneNames.map((n) => h.getNormalizedBoneNode((TO_VRM1[n] ?? n) as never));
    const meshes: THREE.Mesh[] = [];
    this.vrm!.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && m.morphTargetDictionary) meshes.push(m);
    });
    this.morphTargets = morphNames.map((name) =>
      meshes.flatMap((mesh) => (mesh.morphTargetDictionary![name] !== undefined ? [{ mesh, idx: mesh.morphTargetDictionary![name] }] : [])),
    );
    // characters without VRoid face morphs: drive the nearest standard VRM expression instead
    const em = this.vrm!.expressionManager;
    this.expr = morphNames.map((name, i) => {
      const e = FACE_EXPR[name];
      return this.morphTargets[i].length === 0 && e && em?.getExpression(e) ? e : null;
    });
  }

  apply(p: Pose) {
    const s = this.v1 ? -1 : 1;
    for (let b = 0; b < this.bones.length; b++) {
      const n = this.bones[b];
      if (n) n.quaternion.set(s * p.q[b * 4], p.q[b * 4 + 1], s * p.q[b * 4 + 2], p.q[b * 4 + 3]);
    }
  }

  /** after vrm.update(): write face morphs (expression manager resets its own binds every update) */
  applyMorphs(m: Float32Array) {
    for (let i = 0; i < this.morphTargets.length; i++)
      for (const t of this.morphTargets[i]) t.mesh.morphTargetInfluences![t.idx] = m[i];
    const em = this.vrm?.expressionManager;
    if (em) for (let i = 0; i < this.expr.length; i++) if (this.expr[i]) em.setValue(this.expr[i]!, m[i]); // applied on the next update
  }

  update(dt: number) {
    this.lookTarget.position.copy(this.camera.position);
    this.vrm?.update(dt);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /** half-body framing: waist → just above the head, the signing space (hands up to ~0.35 m in front / to the sides) inside */
  frame() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    const aspect = w / h;
    this.camera.aspect = aspect;
    const k = this.bodyScale;
    const top = (this.framing === "half" ? 1.80 : 1.74) * k, bottom = (this.framing === "half" ? 0.98 : 1.18) * k;
    // metres of width that must stay visible (elbows / hands out to the side); a portrait phone keeps the signing space only
    const needW = (this.framing === "half" ? 1.0 : 0.72) * (aspect < 0.8 ? 0.8 : 1.0) * k;
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const distH = (top - bottom) / 2 / Math.tan(fov / 2);
    const distW = needW / 2 / (Math.tan(fov / 2) * aspect);
    const dist = Math.max(distH, distW);
    const cy = (top + bottom) / 2;
    this.camera.position.set(0, cy + 0.02, dist);
    this.camera.lookAt(0, cy, 0);
    this.camera.updateProjectionMatrix();
  }

  resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.frame();
  }

  dispose() {
    this.ro.disconnect();
    this.renderer.dispose();
  }
}

/** a picker thumbnail for a character whose file carries none: head and shoulders, rendered offscreen once (the API keeps it) */
export async function headshot(vrm: VRM, w = 160, h = 200): Promise<Blob> {
  const r = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  r.setPixelRatio(1);
  r.setSize(w, h, false);
  r.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(0.6, 1.8, 1.6);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a9a, 1.6), key, vrm.scene);
  vrm.update(0);
  vrm.scene.updateMatrixWorld(true);
  const head = vrm.humanoid.getNormalizedBoneNode("head")?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3(0, 1.5, 0);
  const cam = new THREE.PerspectiveCamera(28, w / h, 0.05, 10);
  const y = head.y + 0.08;                       // face in the middle, shoulders at the bottom edge (arms are still in T-pose)
  cam.position.set(head.x, y, head.z + 0.75);
  cam.lookAt(head.x, y - 0.01, head.z);
  r.render(scene, cam);
  const blob = await new Promise<Blob>((res, rej) => r.domElement.toBlob((b) => (b ? res(b) : rej(new Error("no image"))), "image/png"));
  scene.remove(vrm.scene);
  r.dispose();
  r.forceContextLoss();
  return blob;
}
