import { ScaleModeOption } from 'settings';

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
    <div>
      <h1>Scene</h1>
    </div>
  );
}
