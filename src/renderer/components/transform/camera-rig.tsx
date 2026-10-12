import React, { useRef, useState } from 'react';
import { CAMERA_AZIMUTHS, cameraViewLabel, snapAzimuth, type CameraElevation } from './intents';

// The "New angle" camera placement: a top-down orbit (drag the camera around
// the subject, snapping to 45° steps) and a side-on height arc (high / eye /
// low). The subject sits in the middle facing where the photo was taken from
// (the ghost camera at the bottom), and never moves: only the camera does.
//
// Orbit geometry: azimuth 0 is the original camera (bottom of the ring) and
// grows toward the right of the original frame, i.e. counter-clockwise on
// screen: 90 = the right of the ring, 180 = the top (directly behind).

const SIZE = 236;
const C = SIZE / 2;
const R = 82;

function orbitPoint(az: number, r = R): [number, number] {
  const t = (az * Math.PI) / 180;
  return [C + r * Math.sin(t), C + r * Math.cos(t)];
}

/** Pointer position → azimuth (unsnapped degrees, 0…360). */
function pointerAzimuth(svg: SVGSVGElement, clientX: number, clientY: number): number {
  const box = svg.getBoundingClientRect();
  const x = ((clientX - box.left) / box.width) * SIZE - C;
  const y = ((clientY - box.top) / box.height) * SIZE - C;
  return ((Math.atan2(x, y) * 180) / Math.PI + 360) % 360;
}

/** A view cone from the camera at (x, y) spanning `half` either side of the target point. */
function conePath(x: number, y: number, tx: number, ty: number, half: number): string {
  const len = Math.hypot(tx - x, ty - y) || 1;
  const px = (-(ty - y) / len) * half;
  const py = ((tx - x) / len) * half;
  return `M${x} ${y} L${tx + px} ${ty + py} L${tx - px} ${ty - py} Z`;
}

/** A small camera glyph at (x, y), lens pointing at the subject. */
function CameraGlyph({ x, y, ghost }: { x: number; y: number; ghost?: boolean }) {
  // Rotate so the lens (drawn pointing up, -y) faces the centre.
  const angle = (Math.atan2(C - y, C - x) * 180) / Math.PI + 90;
  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`} className={ghost ? 'ts-rig-cam ghost' : 'ts-rig-cam'}>
      <rect x={-10} y={-5} width={20} height={13} rx={3} />
      <path d="M-5 -5 L-3 -11 H3 L5 -5" />
    </g>
  );
}

function OrbitView({ azimuth, onChange }: { azimuth: number; onChange: (az: number) => void }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const shown = drag ?? azimuth;
  const [cx, cy] = orbitPoint(shown);
  const [sx, sy] = orbitPoint(azimuth);

  const move = (e: React.PointerEvent) => {
    if (drag === null || !svgRef.current) return;
    const raw = pointerAzimuth(svgRef.current, e.clientX, e.clientY);
    setDrag(raw);
    const snapped = snapAzimuth(raw);
    if (snapped !== azimuth) onChange(snapped);
  };
  const start = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const raw = pointerAzimuth(svgRef.current, e.clientX, e.clientY);
    setDrag(raw);
    onChange(snapAzimuth(raw));
  };
  const end = () => setDrag(null);
  const key = (e: React.KeyboardEvent) => {
    // Arrow keys walk the orbit: right/up toward the right of the frame.
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onChange(snapAzimuth(azimuth + 45));
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onChange(snapAzimuth(azimuth - 45));
    else if (e.key === 'Home') onChange(0);
    else if (e.key === 'End') onChange(180);
    else return;
    e.preventDefault();
  };

  return (
    <svg
      ref={svgRef}
      className={`ts-rig-svg${drag !== null ? ' dragging' : ''}`}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      onPointerDown={start}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      role="slider"
      tabIndex={0}
      aria-label="Camera position around the subject"
      aria-valuemin={0}
      aria-valuemax={315}
      aria-valuenow={azimuth}
      aria-valuetext={CAMERA_AZIMUTHS.find((a) => a.az === azimuth)?.label}
      onKeyDown={key}
    >
      <circle className="ts-rig-ring" cx={C} cy={C} r={R} />
      {/* The view cone from the (snapped) camera to the subject. */}
      <path className="ts-rig-cone" d={conePath(sx, sy, C, C, 22)} />
      {CAMERA_AZIMUTHS.map((a) => {
        const [x, y] = orbitPoint(a.az);
        const [lx, ly] = orbitPoint(a.az, R + (a.az === 0 ? 26 : 17));
        return (
          <g key={a.az} className={`ts-rig-stop${a.az === azimuth ? ' active' : ''}`}>
            <circle cx={x} cy={y} r={3.5} />
            {a.az % 90 === 0 && (
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle">
                {a.az === 0 ? 'original' : a.az === 180 ? 'behind' : a.az === 90 ? 'R' : 'L'}
              </text>
            )}
          </g>
        );
      })}
      {/* The subject, top-down, facing the original camera (down). */}
      <g className="ts-rig-subject">
        <ellipse cx={C} cy={C} rx={18} ry={7} />
        <circle cx={C} cy={C} r={6} />
        <path d={`M${C - 3.5} ${C + 6} L${C} ${C + 12} L${C + 3.5} ${C + 6}`} />
      </g>
      {azimuth !== 0 && <CameraGlyph x={C} y={C + R} ghost />}
      <CameraGlyph x={cx} y={cy} />
    </svg>
  );
}

const ELEVATIONS: Array<{ id: CameraElevation; deg: number; label: string }> = [
  { id: 'high', deg: 45, label: 'High' },
  { id: 'eye', deg: 0, label: 'Eye level' },
  { id: 'low', deg: -30, label: 'Low' },
];

const SW = 170;
const SH = SIZE;
// The arc is centred on the subject's chest.
const PX = 120;
const PY = 128;
const SR = 82;

function heightPoint(deg: number): [number, number] {
  const t = (deg * Math.PI) / 180;
  return [PX - SR * Math.cos(t), PY - SR * Math.sin(t)];
}

function HeightView({ elevation, onChange }: { elevation: CameraElevation; onChange: (e: CameraElevation) => void }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState(false);
  const pick = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * SW;
    const y = ((clientY - box.top) / box.height) * SH;
    const deg = (Math.atan2(PY - y, PX - x) * 180) / Math.PI;
    let best = ELEVATIONS[0];
    ELEVATIONS.forEach((e) => {
      if (Math.abs(e.deg - deg) < Math.abs(best.deg - deg)) best = e;
    });
    if (best.id !== elevation) onChange(best.id);
  };
  const idx = ELEVATIONS.findIndex((e) => e.id === elevation);
  const cur = ELEVATIONS[idx] || ELEVATIONS[1];
  const [cx, cy] = heightPoint(cur.deg);
  const key = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') onChange(ELEVATIONS[Math.max(0, idx - 1)].id);
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') onChange(ELEVATIONS[Math.min(ELEVATIONS.length - 1, idx + 1)].id);
    else return;
    e.preventDefault();
  };
  return (
    <svg
      ref={svgRef}
      className={`ts-rig-svg ts-rig-side${dragging ? ' dragging' : ''}`}
      viewBox={`0 0 ${SW} ${SH}`}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        setDragging(true);
        pick(e.clientX, e.clientY);
      }}
      onPointerMove={(e) => dragging && pick(e.clientX, e.clientY)}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      role="slider"
      tabIndex={0}
      aria-label="Camera height"
      aria-valuemin={-30}
      aria-valuemax={45}
      aria-valuenow={cur.deg}
      aria-valuetext={cur.label}
      onKeyDown={key}
    >
      <path className="ts-rig-ring" d={`M${heightPoint(52)[0]} ${heightPoint(52)[1]} A${SR} ${SR} 0 0 0 ${heightPoint(-36)[0]} ${heightPoint(-36)[1]}`} />
      <line className="ts-rig-ground" x1={8} y1={PY + 52} x2={SW - 8} y2={PY + 52} />
      <path className="ts-rig-cone" d={`M${cx} ${cy} L${PX} ${PY - 34} L${PX} ${PY + 50} Z`} />
      {ELEVATIONS.map((e) => {
        const [x, y] = heightPoint(e.deg);
        return (
          <g key={e.id} className={`ts-rig-stop${e.id === elevation ? ' active' : ''}`}>
            <circle cx={x} cy={y} r={3.5} />
            <text x={x - 12} y={y} textAnchor="end" dominantBaseline="middle">
              {e.label === 'Eye level' ? 'eye' : e.label.toLowerCase()}
            </text>
          </g>
        );
      })}
      {/* Standing subject, side-on. */}
      <g className="ts-rig-subject">
        <circle cx={PX} cy={PY - 26} r={7} />
        <path d={`M${PX} ${PY - 19} V${PY + 14} M${PX} ${PY + 14} L${PX - 7} ${PY + 52} M${PX} ${PY + 14} L${PX + 7} ${PY + 52} M${PX} ${PY - 12} L${PX - 9} ${PY + 10} M${PX} ${PY - 12} L${PX + 9} ${PY + 10}`} />
      </g>
      <CameraGlyph x={cx} y={cy} />
    </svg>
  );
}

export default function CameraRig({
  azimuth,
  elevation,
  onChange,
  thumbUrl,
}: {
  azimuth: number;
  elevation: CameraElevation;
  onChange: (next: { cameraAzimuth?: number; cameraElevation?: CameraElevation }) => void;
  /** The photo being re-shot, shown as "the original view". */
  thumbUrl?: string;
}) {
  const same = azimuth === 0 && elevation === 'eye';
  return (
    <div className="ts-rig">
      <div className="ts-rig-views">
        <figure>
          <OrbitView azimuth={azimuth} onChange={(az) => onChange({ cameraAzimuth: az })} />
          <figcaption>From above · drag the camera around</figcaption>
        </figure>
        <figure>
          <HeightView elevation={elevation} onChange={(e) => onChange({ cameraElevation: e })} />
          <figcaption>From the side · drag up or down</figcaption>
        </figure>
      </div>
      <div className="ts-rig-readout">
        {thumbUrl && <img src={thumbUrl} alt="" className="ts-rig-thumb" />}
        <div>
          <strong className={same ? 'warn' : undefined}>{cameraViewLabel(azimuth, elevation)}</strong>
          <small>
            {same
              ? 'This is where the photo was taken from: move the camera.'
              : 'The subject keeps its spot and pose; only the camera moves.'}
          </small>
        </div>
      </div>
    </div>
  );
}
