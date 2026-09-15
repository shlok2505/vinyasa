# Vinyasa Object Reconstruction

## Goal

Convert one object image into a usable GLB model.

## Current experiment

Input object: Sofa

## Input

reconstruction/input/sofa.jpg

## Expected output

reconstruction/output/sofa.glb

## Pipeline

Image
→ AI 3D reconstruction
→ Raw mesh
→ Mesh cleanup
→ Scale normalization
→ GLB export