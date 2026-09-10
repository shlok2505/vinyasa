import * as THREE from 'three';
import sceneData from './data/scene.json';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

camera.position.set(5, 5, 5);
camera.lookAt(2, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);

document.body.appendChild(renderer.domElement);

const controls = new TransformControls(camera, renderer.domElement);
scene.add(controls.getHelper());

const grid = new THREE.GridHelper(10, 10);
scene.add(grid);

sceneData.objects.forEach((objectData) => {
    const geometry = new THREE.BoxGeometry(
        objectData.dimensions.width,
        objectData.dimensions.height,
        objectData.dimensions.depth
    );

    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });

    const object = new THREE.Mesh(geometry, material);

    object.position.set(
        objectData.position.x,
        objectData.position.y,
        objectData.position.z
    );
    object.rotation.set(
        THREE.MathUtils.degToRad(objectData.rotation?.x ?? 0),
        THREE.MathUtils.degToRad(objectData.rotation?.y ?? 0),
        THREE.MathUtils.degToRad(objectData.rotation?.z ?? 0)
    );
    if (objectData.id === "object_01") {
        controls.attach(object);
    }
    scene.add(object);
});

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

animate();