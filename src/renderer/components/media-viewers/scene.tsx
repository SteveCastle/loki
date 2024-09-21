/* eslint-disable react/no-unknown-property */
import React, { Suspense } from 'react';
import { ScaleModeOption } from 'settings';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

function useFileProtocol(path: string) {
  const electronPath = window.electron.url.format({
    protocol: 'gsm',
    pathname: path,
  });
  return electronPath;
}

type Props = {
  path: string;
  scaleMode?: ScaleModeOption;
  settable?: boolean;
  coverSize?: { width: number; height: number };
  playSound?: boolean;
  handleLoad?: React.ReactEventHandler<HTMLImageElement | HTMLVideoElement>;
  showControls?: boolean;
  mediaRef?: React.RefObject<HTMLVideoElement>;
  initialTimestamp?: number;
  startTime?: number;
  orientation: 'portrait' | 'landscape' | 'unknown';
  onTimestampChange?: (timestamp: number) => void;
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
};

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return (
    <group scale={0.2}>
      <primitive object={scene} />
    </group>
  );
}

function ObjScene({ objUrl }: { objUrl: string }) {
  return (
    <Canvas
      camera={{ position: [0, 0, 10], fov: 50 }}
      style={{ width: '100vw', height: '100vh' }}
    >
      <ambientLight intensity={1} />
      <pointLight position={[10, 10, 10]} />
      <Suspense fallback={null}>
        <Model url={objUrl} />
      </Suspense>
      <OrbitControls />
    </Canvas>
  );
}

export default function Scene({
  path,
  scaleMode,
  settable,
  coverSize,
  playSound,
  handleLoad,
  showControls,
  mediaRef,
  initialTimestamp,
  startTime,
  orientation,
  onTimestampChange,
  cache,
}: Props) {
  return (
    <div className="Scene">
      <h1>Scene</h1>
      <ObjScene objUrl={path} />
    </div>
  );
}
