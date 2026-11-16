// Asset preview: preview.glb (visual + collision layers, one node per MuJoCo body) + joint sliders.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const sanitize = (n) => n.replace(/\s/g, "_").replace(/[\[\]\.:\/]/g, "");

export class AssetPreview {
  constructor(el) { this.el = el; }

  async load(url, asset) {
    const gltf = await new GLTFLoader().loadAsync(url);
    const W = this.el.clientWidth, H = this.el.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(W, H);
    this.el.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d10);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x5a6070, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(2, 5, 3);
    this.scene.add(sun);
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;                    // MuJoCo Z-up -> three.js Y-up
    this.world.add(gltf.scene);
    this.scene.add(this.world);
    this.layers = { vis: [], col: [] };
    gltf.scene.traverse((o) => {
      if (o.name.endsWith("|vis")) this.layers.vis.push(o);
      if (o.name.endsWith("|col")) this.layers.col.push(o);
      if (o.isMesh) {
        o.material = o.material.clone();
        o.material.side = THREE.DoubleSide;
        if (o.name.toLowerCase().startsWith("floor")) { o.material.vertexColors = false; o.material.color.set(0x2a2f38); }
      }
    });
    for (const g of this.layers.col) {
      const room = g.parent && g.parent.name === "body~world";
      g.traverse((o) => {
        if (!o.isMesh || room) return;
        o.material.transparent = true; o.material.opacity = 0.55; o.material.depthWrite = false;
        o.material.vertexColors = false; o.material.color.set(0x9b6cf0);
      });
      g.visible = room;
    }
    // joints: child bodies named "<asset>/<asset>_<child>" in the preview scene
    this.joints = [];
    const byName = new Map();
    gltf.scene.traverse((o) => byName.set(o.name, o));
    for (const j of asset?.articulation?.joints || []) {
      const key = sanitize(`body~${asset.name}/${asset.name}_${j.child}`);
      const node = byName.get(key) || [...byName.values()].find((o) => o.name.startsWith("body~") && o.name.endsWith(sanitize(`_${j.child}`)));
      if (node) this.joints.push({ ...j, node, p0: node.position.clone(), q0: node.quaternion.clone() });
    }
    this.world.updateMatrixWorld(true);
    const b = new THREE.Box3();                         // frame the furniture, not the 10 m floor
    gltf.scene.traverse((o) => { if (o.name.startsWith("body~") && o.name !== "body~world" && o.parent === gltf.scene) b.expandByObject(o); });
    if (b.isEmpty()) b.setFromObject(this.world);
    const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.y, s.z);
    this.camera = new THREE.PerspectiveCamera(40, W / H, 0.01, 50);
    this.camera.position.set(c.x + r * 1.3, c.y + r * 0.9, c.z + r * 1.4);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(c);
    this.controls.enableDamping = true;
    new ResizeObserver(() => {
      const w = this.el.clientWidth, h = this.el.clientHeight;
      this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    }).observe(this.el);
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }

  setLayer(name, on) {
    for (const g of this.layers[name]) if (!(g.parent && g.parent.name === "body~world")) g.visible = on;
  }

  setJoint(i, q) {
    const j = this.joints[i];
    if (!j) return;
    if (j.type === "slide") j.node.position.set(j.p0.x + q * j.axis[0], j.p0.y + q * j.axis[1], j.p0.z + q * j.axis[2]);
    else {                       // hinge (cabinet doors): rotate about the joint axis at the child body origin
      const ax = new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize();
      j.node.quaternion.copy(j.q0).multiply(new THREE.Quaternion().setFromAxisAngle(ax, q));
    }
  }

  dispose() { this.renderer?.setAnimationLoop(null); this.renderer?.dispose(); this.renderer?.domElement.remove(); }
}
