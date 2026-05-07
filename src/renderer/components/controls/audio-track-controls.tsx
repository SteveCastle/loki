import React, { useContext, useCallback } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { clampVolume } from '../../../settings';
import './audio-track-controls.css';

/**
 * Content of the audio popover. Wraps the volume slider and conditionally
 * adds an audio-track picker (when 2+ tracks) and a subtitle on/off toggle
 * (when a sidecar exists). Designed to be rendered inside the existing
 * volumeControlHover container in video-controls.tsx.
 */
export default function AudioTrackControls() {
  const { libraryService } = useContext(GlobalStateContext);

  const volume = useSelector(
    libraryService,
    (state: any) => state.context.settings.volume
  );
  const subtitlesEnabled = useSelector(
    libraryService,
    (state: any) => state.context.settings.subtitlesEnabled
  );
  const tracks = useSelector(
    libraryService,
    (state) => state.context.availableAudioTracks
  );
  const selectedIndex = useSelector(
    libraryService,
    (state) => state.context.selectedAudioTrackIndex
  );
  const subtitle = useSelector(
    libraryService,
    (state) => state.context.availableSubtitle
  );

  const setSetting = useCallback(
    (key: string, value: unknown) => {
      libraryService.send('CHANGE_SETTING', { data: { [key]: value } });
    },
    [libraryService]
  );

  const showTracks = tracks.length >= 2;
  const showSubtitleToggle = subtitle !== null;

  return (
    <div className="audioTrackControls">
      <div className="row volumeRow">
        <span className="rowLabel">Volume</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={volume}
          onChange={(e) => setSetting('volume', clampVolume(parseFloat(e.target.value)))}
          className="volumeSliderHover"
          aria-label="Volume"
        />
        <span className="rowValue">{Math.round(volume * 100)}%</span>
      </div>

      {showTracks && (
        <div className="row trackRow">
          <span className="rowLabel">Audio</span>
          <select
            value={selectedIndex}
            onChange={(e) =>
              libraryService.send({
                type: 'SET_AUDIO_TRACK',
                index: Number(e.target.value),
              })
            }
            aria-label="Audio track"
          >
            {tracks.map((t, i) => {
              const langSuffix = t.language ? ` (${t.language})` : '';
              return (
                <option key={t.id} value={i}>
                  {t.label}
                  {langSuffix}
                </option>
              );
            })}
          </select>
        </div>
      )}

      {showSubtitleToggle && (
        <div className="row subtitleRow">
          <span className="rowLabel">Subtitles</span>
          <button
            className={`toggle ${subtitlesEnabled ? 'on' : 'off'}`}
            onClick={() => setSetting('subtitlesEnabled', !subtitlesEnabled)}
            aria-pressed={subtitlesEnabled}
          >
            {subtitlesEnabled ? 'On' : 'Off'}
          </button>
        </div>
      )}
    </div>
  );
}
