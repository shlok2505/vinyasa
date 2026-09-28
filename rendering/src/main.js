import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';


// --------------------------------------------------
// SCENE
// --------------------------------------------------

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x20252b);


// --------------------------------------------------
// CAMERA
// --------------------------------------------------

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);

camera.position.set(4.5, 2.8, 4.8);
camera.lookAt(0, 1, -0.5);


// --------------------------------------------------
// RENDERER
// --------------------------------------------------

const renderer = new THREE.WebGLRenderer({
  antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

document.body.appendChild(renderer.domElement);


// --------------------------------------------------
// ORBIT CONTROLS
// --------------------------------------------------

const orbitControls = new OrbitControls(
  camera,
  renderer.domElement
);

orbitControls.target.set(0, 1, -0.6);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.08;

orbitControls.minDistance = 2;
orbitControls.maxDistance = 10;
orbitControls.maxPolarAngle = Math.PI / 2.05;


// --------------------------------------------------
// TRANSFORM CONTROLS
// --------------------------------------------------

const transformControls = new TransformControls(
  camera,
  renderer.domElement
);

transformControls.setMode('translate');
transformControls.setSize(0.8);

scene.add(transformControls.getHelper());


// Disable camera movement while dragging furniture
transformControls.addEventListener(
  'dragging-changed',
  (event) => {
    orbitControls.enabled = !event.value;
  }
);


// --------------------------------------------------
// ROOM DIMENSIONS
// --------------------------------------------------

const roomWidth = 10;
const roomDepth = 10;
const roomHeight = 10;
const wallThickness = 0.15;


// --------------------------------------------------
// MATERIALS
// --------------------------------------------------

const floorMaterial = new THREE.MeshStandardMaterial({
  color: 0xa88f72,
  roughness: 0.92,
  metalness: 0
});

const wallMaterial = new THREE.MeshStandardMaterial({
  color: 0xded6c9,
  roughness: 0.96,
  metalness: 0
});


// --------------------------------------------------
// ROOM CREATION HELPER
// --------------------------------------------------

function createBox(
  width,
  height,
  depth,
  material,
  position
) {
  const geometry = new THREE.BoxGeometry(
    width,
    height,
    depth
  );

  const mesh = new THREE.Mesh(
    geometry,
    material
  );

  mesh.position.set(
    position.x,
    position.y,
    position.z
  );

  mesh.castShadow = true;
  mesh.receiveShadow = true;

  scene.add(mesh);

  return mesh;
}


// --------------------------------------------------
// FLOOR
// --------------------------------------------------

createBox(
  roomWidth,
  0.1,
  roomDepth,
  floorMaterial,
  {
    x: 0,
    y: -0.05,
    z: 0
  }
);


// --------------------------------------------------
// BACK WALL
// --------------------------------------------------

createBox(
  roomWidth,
  roomHeight,
  wallThickness,
  wallMaterial,
  {
    x: 0,
    y: roomHeight / 2,
    z: -roomDepth / 2
  }
);


// --------------------------------------------------
// LEFT WALL
// --------------------------------------------------

createBox(
  wallThickness,
  roomHeight,
  roomDepth,
  wallMaterial,
  {
    x: -roomWidth / 2,
    y: roomHeight / 2,
    z: 0
  }
);


// --------------------------------------------------
// RIGHT WALL
// --------------------------------------------------

createBox(
  wallThickness,
  roomHeight,
  roomDepth,
  wallMaterial,
  {
    x: roomWidth / 2,
    y: roomHeight / 2,
    z: 0
  }
);


// --------------------------------------------------
// LIGHTING
// --------------------------------------------------

const ambientLight = new THREE.AmbientLight(
  0xfff4e6,
  1.5
);

scene.add(ambientLight);


// Main sunlight
const sunlight = new THREE.DirectionalLight(
  0xfff1d6,
  3.5
);

sunlight.position.set(3, 6, 4);
sunlight.castShadow = true;

sunlight.shadow.mapSize.width = 2048;
sunlight.shadow.mapSize.height = 2048;

sunlight.shadow.camera.left = -6;
sunlight.shadow.camera.right = 6;
sunlight.shadow.camera.top = 6;
sunlight.shadow.camera.bottom = -6;

sunlight.shadow.camera.near = 0.1;
sunlight.shadow.camera.far = 20;

sunlight.shadow.bias = -0.0005;

scene.add(sunlight);


// Warm ceiling light
const ceilingLight = new THREE.PointLight(
  0xffd6a3,
  2.5,
  8
);

ceilingLight.position.set(
  0,
  roomHeight - 0.15,
  0
);

ceilingLight.castShadow = true;
ceilingLight.shadow.mapSize.width = 1024;
ceilingLight.shadow.mapSize.height = 1024;

scene.add(ceilingLight);


// Cool fill light
const fillLight = new THREE.PointLight(
  0xb8d9ff,
  1.2,
  6
);

fillLight.position.set(
  -2,
  2,
  -2
);

scene.add(fillLight);


// Ceiling fixture
const fixtureGeometry = new THREE.CylinderGeometry(
  0.25,
  0.25,
  0.08,
  32
);

const fixtureMaterial = new THREE.MeshStandardMaterial({
  color: 0x222222,
  roughness: 0.5,
  metalness: 0.4
});

const ceilingFixture = new THREE.Mesh(
  fixtureGeometry,
  fixtureMaterial
);

ceilingFixture.position.set(
  0,
  roomHeight - 0.08,
  0
);

ceilingFixture.castShadow = true;

scene.add(ceilingFixture);


// --------------------------------------------------
// FURNITURE COLLECTION
// --------------------------------------------------

const selectableObjects = [];

let selectedObject = null;


// --------------------------------------------------
// GLB SOFA LOADING
// --------------------------------------------------

// --------------------------------------------------
// SCENE OBJECTS — loaded from src/data/scene.json
// --------------------------------------------------

const gltfLoader = new GLTFLoader();

function resolveModelUrl(modelPath) {
  if (!modelPath) {
    return modelPath;
  }

  // Vite serves files from rendering/public/ at the site root.
  // Older reconstruction entries used "public/model/sofa.glb".
  if (modelPath.startsWith('public/')) {
    return `/${modelPath.slice('public/'.length)}`;
  }

  if (modelPath.startsWith('/')) {
    return modelPath;
  }

  return `/${modelPath}`;
}

async function loadSceneObjects() {
  const response = await fetch('/src/data/scene.json');
  const scene_data = await response.json();

  for (const entry of scene_data.objects) {
    if (entry.type !== 'model') {
      continue; // skip non-model entries (e.g. placeholder "box" test data)
    }

    loadModelEntry(entry);
  }
}

function loadModelEntry(entry) {
  const modelUrl = resolveModelUrl(entry.model);

  gltfLoader.load(
    modelUrl,

    (gltf) => {
      const object = gltf.scene;
      object.name = entry.id;

      object.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material.needsUpdate = true;
          }
        }
      });

      // No rescaling here — Part 2's convert_to_glb() already
      // normalized this mesh to entry.dimensions at generation time.
      // Rescaling again would conflict with that.

      // Position and rotate directly from the merged scene entry
      // (these values trace back to object_manifest.json from Part 1).
      object.position.set(
        entry.position.x,
        entry.position.y,
        entry.position.z
      );

      if (entry.rotation) {
        object.rotation.set(
          THREE.MathUtils.degToRad(entry.rotation.x),
          THREE.MathUtils.degToRad(entry.rotation.y),
          THREE.MathUtils.degToRad(entry.rotation.z)
        );
      }

      scene.add(object);
      selectableObjects.push(object);

      console.log(`Loaded and positioned: ${entry.id}`);
    },

    (progress) => {
      if (progress.total > 0) {
        const percentage = (progress.loaded / progress.total) * 100;
        console.log(`Loading ${entry.id}: ${percentage.toFixed(1)}%`);
      }
    },

    (error) => {
      console.error(`Failed to load ${modelUrl}:`, error);
    }
  );
}

loadSceneObjects();
// --------------------------------------------------
// FIND SELECTABLE PARENT
// --------------------------------------------------

function getSelectableParent(object) {
  let current = object;

  while (current) {
    if (selectableObjects.includes(current)) {
      return current;
    }

    current = current.parent;
  }

  return null;
}


// --------------------------------------------------
// CLICK TO SELECT FURNITURE
// --------------------------------------------------

const raycaster = new THREE.Raycaster();

const mouse = new THREE.Vector2();

window.addEventListener(
  'pointerdown',
  (event) => {
    // Ignore clicks on transform controls
    if (event.target !== renderer.domElement) {
      return;
    }

    mouse.x =
      (event.clientX / window.innerWidth) * 2 - 1;

    mouse.y =
      -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(
      mouse,
      camera
    );

    const intersections =
      raycaster.intersectObjects(
        selectableObjects,
        true
      );

    if (intersections.length === 0) {
      selectedObject = null;
      transformControls.detach();

      return;
    }

    const clickedObject =
      intersections[0].object;

    selectedObject =
      getSelectableParent(clickedObject);

    if (selectedObject) {
      transformControls.attach(
        selectedObject
      );

      transformControls.setMode('rotate');

      console.log(
        'Selected object:',
        selectedObject.name
      );
    }
  }
);


// --------------------------------------------------
// KEYBOARD CONTROLS
// --------------------------------------------------

window.addEventListener(
  'keydown',
  (event) => {
    if (!selectedObject) {
      return;
    }

    if (event.key.toLowerCase() === 'g') {
      transformControls.setMode('translate');
    }

    if (event.key.toLowerCase() === 'r') {
      transformControls.setMode('rotate');
    }

    if (event.key.toLowerCase() === 's') {
      transformControls.setMode('scale');
    }

    if (event.key === 'Escape') {
      transformControls.detach();
      selectedObject = null;
    }
  }
);


// --------------------------------------------------
// KEEP FURNITURE INSIDE ROOM
// --------------------------------------------------

transformControls.addEventListener(
  'objectChange',
  () => {
    if (!transformControls.object) {
      return;
    }

    const object = transformControls.object;

    // Do not allow furniture to move vertically
    if (transformControls.getMode() === 'translate') {
      object.position.y = 0;
    }

    const box = new THREE.Box3().setFromObject(
      object
    );

    const objectSize = new THREE.Vector3();

    box.getSize(objectSize);

    const halfWidth = objectSize.x / 2;
    const halfDepth = objectSize.z / 2;

    const minX =
      -roomWidth / 2 +
      halfWidth +
      wallThickness;

    const maxX =
      roomWidth / 2 -
      halfWidth -
      wallThickness;

    const minZ =
      -roomDepth / 2 +
      halfDepth +
      wallThickness;

    const maxZ =
      roomDepth / 2 -
      halfDepth -
      wallThickness;

    object.position.x = THREE.MathUtils.clamp(
      object.position.x,
      minX,
      maxX
    );

    object.position.z = THREE.MathUtils.clamp(
      object.position.z,
      minZ,
      maxZ
    );
  }
);


// --------------------------------------------------
// RESIZE
// --------------------------------------------------

window.addEventListener(
  'resize',
  () => {
    camera.aspect =
      window.innerWidth / window.innerHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(
      window.innerWidth,
      window.innerHeight
    );
  }
);


// --------------------------------------------------
// ANIMATION LOOP
// --------------------------------------------------

function animate() {
  requestAnimationFrame(animate);

  orbitControls.update();

  renderer.render(
    scene,
    camera
  );
}

animate();